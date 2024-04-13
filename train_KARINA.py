import os
import time
import numpy as np
import argparse
import torch
import cProfile
import re
import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.karina import KARINA
import wandb
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch
from utils.darcy_loss import LpLoss
import matplotlib.pyplot as plt
from collections import OrderedDict
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from torch.nn.utils import clip_grad_norm_
import random
from torch.optim import AdamW

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # If you are using CuDNN, the below two lines can also help with reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer():
  def count_parameters(self):
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

  def __init__(self, params, world_rank):
    self.max_gradient_norm = 1.0 
    self.params = params
    self.world_rank = world_rank
    self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    if params.log_to_wandb:
      wandb.init(config=params, name=params.name, group=params.group, project=params.project, entity=params.entity)

    logging.info('rank %d, begin data loader init'%world_rank)
    self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(params, params.train_data_path, dist.is_initialized(), train=True)
    self.valid_data_loader, self.valid_dataset = get_data_loader(params, params.valid_data_path, dist.is_initialized(), train=False)
    self.loss_obj = LpLoss()
    logging.info('rank %d, data loader initialized'%world_rank)

    params.crop_size_x = self.valid_dataset.crop_size_x
    params.crop_size_y = self.valid_dataset.crop_size_y
    params.img_shape_x = self.valid_dataset.img_shape_x
    params.img_shape_y = self.valid_dataset.img_shape_y

    if params.nettype == 'karina':
      self.model = KARINA(params).to(self.device) 
    else:
      raise Exception("not implemented")

    if self.params.enable_nhwc:
      # NHWC: Convert model to channels_last memory format
      self.model = self.model.to(memory_format=torch.channels_last)

    if params.log_to_wandb:
      wandb.watch(self.model)

    self.optimizer = AdamW(self.model.parameters(), lr=params.lr)

    if params.enable_amp == True:
      self.gscaler = amp.GradScaler()

    if dist.is_initialized():
      self.model = DistributedDataParallel(self.model,
                                           device_ids=[params.local_rank],
                                           output_device=[params.local_rank],find_unused_parameters=True)

    self.iters = 0
    self.startEpoch = 0
    if params.resuming:
      logging.info("Loading checkpoint %s"%params.best_checkpoint_path)
      self.restore_checkpoint(params.best_checkpoint_path)
        
    self.epoch = self.startEpoch

    if params.scheduler == 'ReduceLROnPlateau':
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
    elif params.scheduler == 'CosineAnnealingLR':
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.max_epochs, last_epoch=self.startEpoch-1)
    else:
      self.scheduler = None

    '''if params.log_to_screen:
      logging.info(self.model)'''
    if params.log_to_screen:
      logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

  def train(self):
    if self.params.log_to_screen:
      logging.info("Starting Training Loop...")

    best_valid_loss = 1.e6
    for epoch in range(self.startEpoch, self.params.max_epochs):
      if dist.is_initialized():
        self.train_sampler.set_epoch(epoch)
#        self.valid_sampler.set_epoch(epoch)

      start = time.time()
      tr_time, data_time, train_logs = self.train_one_epoch()
      valid_time, valid_logs = self.validate_one_epoch()
      if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
        valid_weighted_rmse = self.validate_final()

      if self.params.scheduler == 'ReduceLROnPlateau':
        self.scheduler.step(valid_logs['valid_loss'])
      elif self.params.scheduler == 'CosineAnnealingLR':
        self.scheduler.step()
        if self.epoch >= self.params.max_epochs:
          logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
          exit()

      if self.params.log_to_wandb:
        for pg in self.optimizer.param_groups:
          lr = pg['lr']
        wandb.log({'lr': lr})

      if self.world_rank == 0:
        if self.params.save_checkpoint:
          #checkpoint at the end of every epoch
          self.save_checkpoint(self.params.checkpoint_path)
          if valid_logs['valid_loss'] <= best_valid_loss:
            #logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
            self.save_checkpoint(self.params.best_checkpoint_path)
            best_valid_loss = valid_logs['valid_loss']

      if self.params.log_to_screen:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        #logging.info('train data time={}, train step time={}, valid step time={}'.format(data_time, tr_time, valid_time))
        logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))
#        if epoch==self.params.max_epochs-1 and self.params.prediction_type == 'direct':
#          logging.info('Final Valid RMSE: Z500- {}. T850- {}, 2m_T- {}'.format(valid_weighted_rmse[0], valid_weighted_rmse[1], valid_weighted_rmse[2]))

  def train_one_epoch(self):
    self.epoch += 1
    tr_time = 0
    data_time = 0
    self.model.train()

    for i, data in enumerate(self.train_data_loader, 0):
        self.iters += 1
        data_start = time.time()
        inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)

        if self.params.enable_nhwc:
            inp = inp.to(memory_format=torch.channels_last)
            tar = tar.to(memory_format=torch.channels_last)

        if 'residual_field' in self.params.target:
            tar -= inp[:, 0:tar.size()[1]]
        data_time += time.time() - data_start
        tr_start = time.time()

        self.model.zero_grad()

        if self.params.enable_amp:
            with torch.cuda.amp.autocast():
                # Normal forward pass
                gen = self.model(inp)
                loss = self.loss_obj(gen, tar)

                # Gradient scaling
                self.gscaler.scale(loss).backward()
                self.gscaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
        else:
            # Normal training without AMP
            gen = self.model(inp)
            loss = self.loss_obj(gen, tar)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient_norm)
            self.optimizer.step()

        tr_time += time.time() - tr_start

    # Collecting logs for the current training epoch
    logs = {'loss': loss.item()}

    if dist.is_initialized():
        for key in sorted(logs.keys()):
            log_tensor = torch.tensor(logs[key]).to(self.device)  # Convert scalar to tensor and send to the correct device
            dist.all_reduce(log_tensor)
            logs[key] = log_tensor.item() / dist.get_world_size()

    if self.params.log_to_wandb:
        wandb.log(logs, step=self.epoch)
        
    print(f"GPU {dist.get_rank()} - Epoch {self.epoch} - Batch {i} - Loss: {loss.item()}")

    return tr_time, data_time, logs

  def validate_one_epoch(self):
    self.model.eval()
    n_valid_batches = 9  # Consider adjusting this based on your validation set size and needs
    valid_loss = 0.0
    valid_l1 = 0.0
    valid_steps = 0

    valid_start = time.time()

    with torch.no_grad():
        for i, data in enumerate(self.valid_data_loader, 0):
            if i >= n_valid_batches:  # Limit the number of batches for validation if needed
                break
            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)

            gen = self.model(inp)
            batch_loss = self.loss_obj(gen, tar)
            valid_loss += batch_loss.item()
            valid_l1 += torch.nn.functional.l1_loss(gen, tar).item()

            valid_steps += 1

    # Average the validation metrics over all validated batches
    valid_loss_avg = valid_loss / valid_steps
    valid_l1_avg = valid_l1 / valid_steps

    valid_time = time.time() - valid_start

    logs = {'valid_loss': valid_loss_avg, 'valid_l1': valid_l1_avg}

    if self.params.log_to_wandb:
        wandb.log(logs, step=self.epoch)

    return valid_time, logs


  def validate_final(self):
    self.model.eval()
    # Assuming n_valid_batches is set to cover the whole validation dataset or a specific portion you're interested in.
    n_valid_batches = int(self.valid_dataset.n_patches_total / self.valid_dataset.n_patches)
    valid_weighted_rmse = torch.zeros((self.params.N_out_channels), dtype=torch.float32, device=self.device)
    
    if self.params.normalization == 'minmax':
        raise Exception("minmax normalization not supported")
    elif self.params.normalization == 'zscore':
        mult = torch.as_tensor(np.load(self.params.global_stds_path)[0, self.params.out_channels, 0, 0]).to(self.device)
    
    total_weighted_rmse = 0
    count = 0

    with torch.no_grad():
        for i, data in enumerate(self.valid_data_loader):
            if i >= n_valid_batches:
                break
            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)
            if 'residual_field' in self.params.target:
                tar -= inp[:, 0:tar.size()[1]]

            gen = self.model(inp)
            for c in range(self.params.N_out_channels):
                if 'residual_field' in self.params.target:
                    weighted_rmse = weighted_rmse_torch((gen[:, c] + inp[:, c]), (tar[:, c] + inp[:, c]), self.device)
                else:
                    weighted_rmse = weighted_rmse_torch(gen[:, c], tar[:, c], self.device)
                valid_weighted_rmse[c] += weighted_rmse
            count += 1

    # Average the RMSE across all batches and channels
    valid_weighted_rmse /= count

    # Apply normalization if necessary
    if self.params.normalization == 'zscore':
        valid_weighted_rmse = mult * valid_weighted_rmse

    return valid_weighted_rmse.cpu().numpy()  # Convert to CPU and numpy for further processing or analysis


  def load_model_wind(self, model_path):
    if self.params.log_to_screen:
      logging.info('Loading the wind model weights from {}'.format(model_path))
    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.params.local_rank))
    if dist.is_initialized():
      self.model_wind.load_state_dict(checkpoint['model_state'])
    else:
      new_model_state = OrderedDict()
      model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
      for key in checkpoint[model_key].keys():
          if 'module.' in key: # model was stored using ddp which prepends module
              name = str(key[7:])
              new_model_state[name] = checkpoint[model_key][key]
          else:
              new_model_state[key] = checkpoint[model_key][key]
      self.model_wind.load_state_dict(new_model_state)
      self.model_wind.eval()

  def save_checkpoint(self, checkpoint_path, model=None):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """
 
    if not model:
      model = self.model

    torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

  def restore_checkpoint(self, checkpoint_path):
     #We intentionally require a checkpoint_dir to be passed
     #   in order to allow Ray Tune to use this function 
    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.params.local_rank))
    try:
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.param_groups[0]['initial_lr'] = self.params.lr
    except:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            new_state_dict[name] = val 
        self.model.load_state_dict(new_state_dict)
    self.iters = checkpoint['iters']
    self.startEpoch = checkpoint['epoch']
    if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
  set_seed(777)
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='01', type=str)
  parser.add_argument("--yaml_config", default='./config/KARINA.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  parser.add_argument("--enable_amp", action='store_true')
  parser.add_argument("--epsilon_factor", default = 0, type = float)
  parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')


  args = parser.parse_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)
  params['epsilon_factor'] = args.epsilon_factor

  params['world_size'] = 4
  if 'WORLD_SIZE' in os.environ:
    params['world_size'] = int(os.environ['WORLD_SIZE'])
    print(f"World size: {params['world_size']}")

  world_rank = 0
  local_rank = 0
  if params['world_size'] > 1:
    dist.init_process_group(backend='nccl',
                            init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    args.gpu = local_rank
    world_rank = dist.get_rank()
    params['global_batch_size'] = params.batch_size
    params['batch_size'] = int(params.batch_size//params['world_size'])

  torch.cuda.set_device(local_rank)
  torch.backends.cudnn.benchmark = True

  # Set up directory
  expDir = "/home/jmj2316/KARINA_eccv_no_padding_yes_SE" 
  if  world_rank==0:
    if not os.path.isdir(expDir):
      os.makedirs(expDir, exist_ok=True)
      os.makedirs(os.path.join(expDir, 'KARINA/'))

  params['experiment_dir'] = os.path.abspath(expDir)
  params['checkpoint_path'] = os.path.join(expDir, 'KARINA/ckpt.tar')
  params['best_checkpoint_path'] = os.path.join(expDir, 'KARINA/best_ckpt.tar')

  # Do not comment this line out please:
  args.resuming = True if os.path.isfile(params.checkpoint_path) else False

  params['resuming'] = args.resuming
  params['local_rank'] = local_rank
  params['enable_amp'] = args.enable_amp

  # this will be the wandb name
#  params['name'] = args.config + '_' + str(args.run_num)
#  params['group'] = "era5_wind" + args.config
  params['name'] = args.config + '_' + str(args.run_num)
  params['group'] = "era5_precip" + args.config
  params['project'] = "ERA5_precip"
  params['entity'] = "flowgan"
  if world_rank==0:
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
    logging_utils.log_versions()
    params.log()

  params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
  params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

  params['in_channels'] = np.array(params['in_channels'])
  params['out_channels'] = np.array(params['out_channels'])
  if params.orography:
    params['N_in_channels'] = len(params['in_channels']) +1
  else:
    params['N_in_channels'] = len(params['in_channels'])

  params['N_out_channels'] = len(params['out_channels'])

  if world_rank == 0:
    hparams = ruamelDict()
    yaml = YAML()
    for key, value in params.params.items():
      hparams[str(key)] = str(value)
    with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
      yaml.dump(hparams,  hpfile )

  trainer = Trainer(params, world_rank)
  trainer.train()
  logging.info('DONE ---- rank %d'%world_rank)

# This file utilizes methods adapted from NVIDIA FourCastNet for data processing.
# Original FourCastNet code can be found at https://github.com/NVlabs/FourCastNet
# We thank the NVIDIA FourCastNet team for making their code available for use.
