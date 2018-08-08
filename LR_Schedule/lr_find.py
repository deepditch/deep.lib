import torch
import torch.nn as nn
import torch.optim as optim
from LR_Schedule.lr_scheduler import _LRScheduler, _OnBatchLRScheduler
import matplotlib.pyplot as plt
from session import *

class LRFindScheduler(_OnBatchLRScheduler):
    def __init__(self, num_examples, start_lr=1e-5, end_lr=10):
        super(LRFindScheduler, self).__init__()       
        self.losses = []
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_examples = num_examples

    def on_train_begin(self, session):
        super(LRFindScheduler, self).on_train_begin(session)
        self.multipliers = [(self.end_lr/base_lr)/self.num_examples for base_lr in self.base_lrs]
        self.lrs = []

    def on_train_end(self, session):
        self.set_lr(self.base_lrs)
    
    def get_lr(self):
        new_lr = [base_lr * mult * self.iteration for base_lr, mult in zip(self.base_lrs, self.multipliers)]
        self.lrs.append(new_lr)
        return new_lr

    def on_batch_end(self, session, loss):
        self.losses.append(loss)


def lr_find(session, data, start_lr=1e-5, end_lr=10):
    scheduler = LRFindScheduler(len(data), start_lr, end_lr)
    tmp_session = Session(session.model, session.criterion, session.optimizer, scheduler)
    tmp_session.train(data, 1)

    # Plot the results 
    fig, ax = plt.subplots()

    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate (log scale)")

    for lr in [*zip(*scheduler.lrs)]: # Matrix transposition
        ax.plot(lr, scheduler.losses)

    ax.set_xscale('log')