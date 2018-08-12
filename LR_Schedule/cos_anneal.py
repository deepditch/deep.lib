import torch
import torch.nn as nn
import torch.optim as optim
from LR_Schedule.lr_scheduler import _LRScheduler, _OnBatchLRScheduler
import session
import math

class CosAnneal(_OnBatchLRScheduler):
    def __init__(self, T_max, lr_min=0, iteration=0, T_mult=1):
        self.lr_min = lr_min       
        self.T_cur = iteration
        self.T_max = T_max
        self.T_mult = T_mult
        super().__init__(iteration)

    def get_lr(self):
        cos_out = (1 + math.cos(math.pi * (self.T_cur % self.T_max) / self.T_max))  
        self.T_cur += 1
        return [self.lr_min + .5 * (base_lr - self.lr_min) * cos_out for base_lr in self.base_lrs]
