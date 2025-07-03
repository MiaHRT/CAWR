# Copyright 2025 Ranting Hu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.distributions import Normal
import torch
import math
from numbers import Real
from torch.distributions.utils import broadcast_all
from numbers import Number


class Huber(Normal):
    def __init__(self, loc, scale, validate_args=None):
        super(Huber, self).__init__(loc=loc, scale=scale, validate_args=validate_args)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        
        # Huber Loss is a loss function first proposed by Huber in 1964, 
        # behaves as the L2 norm when close to the origin and transitions to the L1 norm when far from the origin, 
        # see https://link.springer.com/chapter/10.1007/978-1-4612-4380-9_35 for more details.
        kappa = 0.2
        errors = value - self.loc
        huber_loss = torch.where(
            errors.abs() <= kappa, errors ** 2, kappa * (2 * errors.abs() - kappa)
        )
        return -huber_loss / (2 * var)


class Skew(Normal):
    def __init__(self, loc, scale, validate_args=None):
        super(Skew, self).__init__(loc=loc, scale=scale, validate_args=validate_args)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        
        return torch.log((torch.exp(-((value - self.loc) ** 2) / (var)) + 1/(torch.abs(value - self.loc)/self.scale + 1)) / self.scale)


class Flat(Normal):
    def __init__(self, loc, scale, validate_args=None):
        super(Flat, self).__init__(loc=loc, scale=scale, validate_args=validate_args)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        
        return torch.log(torch.exp(-((value - self.loc) ** 2) / (var/2))/torch.sqrt(var) + 0.5)
        
