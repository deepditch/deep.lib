import torch
import torch.nn as nn
import torch.optim as optim
from LR_Schedule.lr_scheduler import _LRScheduler, _OnBatchLRScheduler
import session
import math

class CosAnneal(_OnBatchLRScheduler):
    def __init__(self, T_max, eta_min=0, iteration=0):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(iteration)

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.iteration / self.T_max)) / 2
                for base_lr in self.base_lrs]
