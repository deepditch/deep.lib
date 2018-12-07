import torch
import torch.nn as nn
import torch.optim as optim
from LR_Schedule.lr_scheduler import _LRScheduler, _OnBatchLRScheduler
import session
import math

class Cyclical(_OnBatchLRScheduler):
    '''Cyclical learning rate schedule'''

    def __init__(self, cycle_len, div=4, cut_div=8, iteration=0, momentums=(32,5)):
        self.cycle_len = cycle_len
        self.cycle_iter = 0
        self.cycle_count = 0
        self.div = div
        self.cut_div = cut_div
        if momentums is not None:
            self.moms = momentums
        super().__init__(iteration)

    def get_lr(self):        
        cut_pt = self.cycle_len//self.cut_div

        if self.cycle_iter > cut_pt:
            pct = 1 - (self.cycle_iter - cut_pt) / (self.cycle_len - cut_pt)
        else: pct = self.cycle_iter / cut_pt
        
        res = [base_lr * (1 + pct * (self.div - 1)) / self.div for base_lr in self.base_lrs]

        self.cycle_iter += 1
        if self.cycle_iter==self.cycle_len:
            self.cycle_iter = 0
            self.cycle_count += 1

        return res

    def should_get_mom(self): return False

    def get_mom(self):
        cut_pt = self.cycle_len//self.cut_div
        if self.cycle_iter > cut_pt:
            pct = (self.cycle_iter - cut_pt)/(self.cycle_len - cut_pt)
        else: pct = 1 - self.cycle_iter/cut_pt

        res = self.moms[1] + pct * (self.moms[0] - self.moms[1])

        return res

    def sub_reset(self):
        self.cycle_iter = 0
        self.cycle_count = 0
