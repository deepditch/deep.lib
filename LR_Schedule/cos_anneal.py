import torch
import torch.nn as nn
import torch.optim as optim
from LR_Schedule.lr_scheduler import _LRScheduler, _OnBatchLRScheduler
import session
import math

class CosAnneal(_OnBatchLRScheduler):
    '''Cosine annealing learning rate schedule'''

    def __init__(self, T_max, lr_min=0, iteration=0, T_mult=1):
        self.lr_min = lr_min       
        self.T_max = T_max
        self.init_T_max = T_max
        self.T_mult = T_mult
        self.T_cur = iteration
        super().__init__(iteration)

    def get_lr(self):        
        if self.T_cur == self.T_max: 
            self.T_max *= self.T_mult
            self.T_cur = 0
    
        cos_out = (1 + math.cos(math.pi * (self.T_cur / self.T_max)))
            
        self.T_cur += 1
        
        if type(self.lr_min) is list or type(self.lr_min) is tuple:
            return [lr_min + .5 * (base_lr - lr_min) * cos_out for lr_min, base_lr in zip(self.lr_min, self.base_lrs)]
        else:
            return [self.lr_min + .5 * (base_lr - self.lr_min) * cos_out for base_lr in self.base_lrs]

    def sub_reset(self):
        self.T_cur = 0
        self.T_max = self.init_T_max
