# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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

# This file utilizes methods adapted from NVIDIA FourCastNet for data processing.
# Original FourCastNet code can be found at https://github.com/NVlabs/FourCastNet
# We thank the NVIDIA FourCastNet team for making their code available for use.
