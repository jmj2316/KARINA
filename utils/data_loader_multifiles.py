import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import netCDF4 as nc
from utils.img_utils import reshape_fields
import torch.distributed as dist

def get_data_loader(params, files_pattern, distributed, train):
    dataset = GetDataset(params, files_pattern, train)
    
    # If distributed training is enabled, use DistributedSampler
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    # DataLoader setup
    dataloader = DataLoader(dataset,
                            batch_size=params.batch_size,
                            num_workers=1,
                            shuffle=(not distributed and train),  # Shuffle only for non-distributed training
                            sampler=sampler if train else None,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


def is_leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    return False    


class GetDataset(Dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.crop_size_x = params.crop_size_x
        self.crop_size_y = params.crop_size_y
        self.roll = params.roll
        self._get_files_stats()
        print("Files Paths:", self.files_paths)  # Add this print statement
        #self.channel_weights = [1, 0.5, 0.8, 1, 1, 0.7, 1, 0.7, 1]
        self.orography = params.orography
        self.add_noise = params.add_noise if train else False

        try:
            self.normalize = params.normalize
        except:
            self.normalize = True #by default turn on normalization if not specified in config

        if self.orography:
            self.orography_path = params.orography_path

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.nc")
        def custom_sort(path):
            # Extract year and lag values from the filename
            year = int(re.search(r'(\d{4})\.nc', path).group(1))
            lag = int(re.search(r'lag(\d+)_', path).group(1))
            return (year, lag)
    
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        with nc.Dataset(self.files_paths[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            #original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2]#just get rid of one of the pixels
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        self.precip_files = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
        logging.info("Delta t: {} hours".format(1*self.dt))
        logging.info("Including {} hours of past history in training at a frequency of {} hours".format(1*self.dt*self.n_history, 1*self.dt))

    def _open_file(self, year_idx):
        if self.files_paths[year_idx] is None:
            raise ValueError(f"No file path found for year index: {year_idx}")
        #print(f"[DEBUG] Opening .nc file: {self.files_paths[year_idx]}")
        _file = nc.Dataset(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields']
        if self.orography:
            _orog_file = nc.Dataset(self.orography_path, 'r')
            self.orography_field = _orog_file['orog']
    
    def year_from_idx(self, year_idx):
    # Extract the year from the filepath
        filepath = self.files_paths[year_idx]
        year = int(filepath.split('_')[-1].split('.')[0])
        return year
    
    def __len__(self):
        return self.n_samples_total - self.n_years - 1

    def __getitem__(self, global_idx):
        year_idx = int(global_idx/self.n_samples_per_year)  # which year we are on

        # Determine if the current year is a leap year
        if is_leap_year(self.year_from_idx(year_idx)): 
            max_samples_current_year = 366
            self.n_samples_per_year = 366 
            # If you still want to keep this debug line, you can modify it similarly.
            # print(f"[DEBUG] Year {self.year_from_idx(year_idx)} is detected as a leap year.")
        else:
            max_samples_current_year = 365
            self.n_samples_per_year = 365
        local_idx = int(global_idx % max_samples_current_year)  # which sample in that year we are on - determines indices for centering

        # Write the debug statement to a dedicated log file
        rank = dist.get_rank()
        with open(f"log_gpu_{rank}.txt", "a") as log_file:
            log_file.write(f"[DEBUG] Accessing global_idx={global_idx}, year_idx={year_idx}, local_idx={local_idx}\n")

        y_roll = np.random.randint(0, 144) if self.train else 0  # roll image in y direction

        # Open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)

            # Ensure we're at least self.dt * n_history timesteps into the prediction.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history
    
        # If we're on the last image of the year, shift to the first image of the next year.
        if local_idx >= max_samples_current_year - self.dt:
                # If we're on the last .nc file, skip the last sample
            if year_idx == self.n_years - 1:
                    #print(f"[DEBUG] Skipping last sample from the last .nc file: global_idx={global_idx}, local_idx={local_idx}")
                raise IndexError(f"Skipping last sample from the last .nc file: global_idx={global_idx}, local_idx={local_idx}")
            else:
                year_idx += 1
                    # Adjust for next year's samples
                if is_leap_year(self.year_from_idx(year_idx)):
                    max_samples_current_year = 366
                else:
                    max_samples_current_year = 365
                local_idx = self.dt * self.n_history
                    #print(f"[DEBUG] Transitioning to year_idx={year_idx} and local_idx={local_idx}")
                self._open_file(year_idx)

            step = self.dt

        else:
            inp_local_idx = local_idx
            tar_local_idx = local_idx
    
            # if we are on the last image in a year, raise an error
            if tar_local_idx >= self.n_samples_per_year - self.dt:
                raise IndexError("The last sample is not allowed.")
            else:
                step = self.dt
    
            if year_idx == 0:
                lim = 1458
                local_idx = local_idx % lim
                inp_local_idx = local_idx + 2
                tar_local_idx = local_idx
                step = 0 if tar_local_idx >= lim - self.dt else self.dt
                    
        if self.train and self.roll:
            y_roll = random.randint(0, self.img_shape_y)
        else:
            y_roll = 0

        if self.orography:
            orog = self.orography_field[0:144]
        else:
            orog = None

        if self.train and (self.crop_size_x or self.crop_size_y):
            rnd_x = random.randint(0, self.img_shape_x-self.crop_size_x)
            rnd_y = random.randint(0, self.img_shape_y-self.crop_size_y)
        else:
            rnd_x = 0
            rnd_y = 0
                
        fields = self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels]
        
        #for i, weight in enumerate(self.channel_weights):
        #    fields[:, i] *= weight
        if year_idx >= len(self.files):
            print(f"[DEBUG] Error Condition 1:")
            print(f"year_idx: {year_idx}, len(self.files): {len(self.files)}, global_idx: {global_idx}")
            raise IndexError(f"year_idx ({year_idx}) exceeds the number of years available.")

    # Check time index bounds:
        if local_idx + step >= self.n_samples_per_year:
            print(f"[DEBUG] Error Condition 2:")
            print(f"local_idx: {local_idx}, step: {step}, self.n_samples_per_year: {self.n_samples_per_year}, max_samples_current_year: {max_samples_current_year}")
            raise IndexError(f"Time index (local_idx + step = {local_idx + step}) exceeds available time samples for the year.")

    # Check channel index bounds:
        if max(self.out_channels) >= self.n_in_channels:
            print(f"[DEBUG] Error Condition 3:")
            print(f"max(self.out_channels): {max(self.out_channels)}, self.n_in_channels: {self.n_in_channels}")
            raise IndexError(f"Channel index ({max(self.out_channels)}) exceeds available channels.")

        inp = reshape_fields(self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels], 'inp', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog, self.add_noise)
        
        tar = reshape_fields(self.files[year_idx][local_idx + step, self.out_channels], 'tar', self.crop_size_x, self.crop_size_y, rnd_x, rnd_y, self.params, y_roll, self.train, self.normalize, orog)

        # Print inp and tar for debugging
        #print("Input (inp) Shape:", inp.shape, "Min:", inp.min(), "Max:", inp.max())
        #print("Target (tar) Shape:", tar.shape, "Min:", tar.min(), "Max:", tar.max())
    
        return inp, tar
        
# This file utilizes methods adapted from NVIDIA FourCastNet for data processing.
# Original FourCastNet code can be found at https://github.com/NVlabs/FourCastNet
# We thank the NVIDIA FourCastNet team for making their code available for use.
