import torch
import torch.nn as nn
import torch.optim as optim
from LR_Schedule.lr_scheduler import _LRScheduler, _OnBatchLRScheduler
import session
import math
import numpy as np

class LearningRateDecay(_OnBatchLRScheduler):
    '''Learning rate decay schedule'''

    def __init__(self, iterations, intervals=[1/3, 1/3, 1/3], lrs=[1e-3, 1e-4, 1e-5], iteration=0):
        super().__init__(iteration)
        self.iterations = iterations       
        self.intervals = intervals
        self.lrs = lrs

        assert np.sum(intervals) == 1

        self.thresholds = [0]
        self.idx = 0

        for i, interval in enumerate(intervals):
            if self.thresholds[i] >= self.iteration: self.idx == i
            self.thresholds.append(self.thresholds[i] + self.iterations*interval)

    def get_lr(self):        
        if self.idx < len(self.thresholds)-2 and self.thresholds[self.idx+1] < self.iteration:
            self.idx += 1

        return self.lrs[self.idx]