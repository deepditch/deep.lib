import torch
import torch.nn as nn
import torch.optim as optim
from lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt
import session

class LRFindScheduler(_LRScheduler):
    def __init__(optimizer, num_examples, end_lr = 10):
        super(LRFindScheduler, self).__init__(optimizer)
        self.multipliers = [(end_lr/base_lr)/num_examples for base_lr in self.base_lrs]
        self.lrs = []
    
    def get_lr(self):
        new_lr = [base_lr * mult * self.last_epoch for base_lr, mult in zip(self.base_lrs, self.multipliers)]
        self.lrs.append(new_lr)
        return new_lr


def lr_find(session, dataLoader):
    schedule = LRFindScheduler(session.optimizer, len(dataLoader)) 
    losses = []   
    for input, label in dataLoader:
        schedule.step()
        losses.append(session.step(input, label))

    # Plot the results 
    fig, ax = plt.subplots()

    ax.ylabel("validation loss")
    ax.xlabel("learning rate (log scale)")

    for lr in list(map(list, zip(*schedule.lrs))): # Matrix transposition
        ax.plot(lr, losses)

    ax.xscale('log')