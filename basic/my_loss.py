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
        
