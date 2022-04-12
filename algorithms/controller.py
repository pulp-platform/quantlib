# 
# controller.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

__all__ = ['Controller']


class Controller(object):

    def __init__(self):
        # controller base class does not do any initialization
        pass

    @staticmethod
    def get_modules(*args, **kwargs):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step_pre_training_epoch(self, *args, **kwargs):
        """Update the quantization hyper-parameters before the training step of the current epoch."""
        raise NotImplementedError

    def step_pre_training_batch(self, *args, **kwargs):
        """Update the quantization hyper-parameters before the training mini-batch."""
        pass

    def step_pre_validation_epoch(self, *args, **kwargs):
        """Update the quantization hyper-parameters before the validation step of the current epoch."""
        raise NotImplementedError

