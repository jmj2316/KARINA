import os
import logging
import subprocess

_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

def config_logger(log_level=logging.INFO):
    logging.basicConfig(format=_format, level=log_level)

def log_to_file(logger_name=None, log_level=logging.INFO, log_filename='tensorflow.log'):
    if not os.path.exists(os.path.dirname(log_filename)):
        os.makedirs(os.path.dirname(log_filename))

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_format))
    log.addHandler(fh)

def log_versions():
    import torch

    logging.info('--------------- Versions ---------------')
    
    try:
        branch_name = subprocess.check_output(['git', 'branch'], stderr=subprocess.STDOUT).strip()
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).strip()
        logging.info('git branch: ' + str(branch_name))
        logging.info('git hash: ' + str(git_hash))
    except subprocess.CalledProcessError as e:
        logging.error('Git command failed: ' + str(e))
    except Exception as e:
        logging.error('Unexpected error: ' + str(e))
    
    logging.info('Torch: ' + str(torch.__version__))
    logging.info('----------------------------------------')
