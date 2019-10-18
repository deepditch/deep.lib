import torch
import torch.nn as nn
import torch.optim as optim
from LR_Schedule.lr_scheduler import _LRScheduler, _OnBatchLRScheduler
import matplotlib.pyplot as plt
from session import Session, TrainingSchedule
import math
import util

class LRFindScheduler(_OnBatchLRScheduler):
    '''Class representing the learning rate find algorithm. Linearly increases learning rate while training. 
    This learning rate scheduler will cause loss to diverge.
    '''

    def __init__(self, num_examples, start_lr=None, end_lr=10):
        super(LRFindScheduler, self).__init__()       
        self.losses = []
        self.end_lr = end_lr
        self.num_examples = num_examples
        self.start_lr = start_lr

    def on_train_begin(self, session):
        super(LRFindScheduler, self).on_train_begin(session)
        if self.start_lr is not None: self.base_lrs = util.listify(self.start_lr, session.optimizer.param_groups)
        self.end_lr = util.listify(self.end_lr, session.optimizer.param_groups)
        self.multipliers = [(end_lr/base_lr)**(1/self.num_examples) for base_lr, end_lr in zip(self.base_lrs, self.end_lr)]
        self.lrs = []
        self.best=40
    
    def get_lr(self):
        new_lr = [base_lr * mult ** self.iteration for base_lr, mult in zip(self.base_lrs, self.multipliers)]
        self.lrs.append(new_lr)
        return new_lr

    def on_batch_end(self, session, lossMeter, output, label):
        self.losses.append(lossMeter.debias)
        if (math.isnan(lossMeter.debias) or lossMeter.debias > self.best*4 or lossMeter.debias is float('nan')):
            session.stop()
        if (lossMeter.debias<self.best and self.iteration>10): self.best=lossMeter.debias

    def plot(self, iterations=None):
        if(iterations is None): iterations = self.iteration

        fig, ax_loss = plt.subplots()   
        ax_lr = ax_loss.twinx()

        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Loss", color='g')

        ax_lr.set_yscale("log")
        ax_lr.set_ylabel("Learning Rate (Log Scale)", color='b')

        ax_loss.plot(range(iterations), self.losses, 'g-')
    
        for lr in [*zip(*self.lrs)]: # Matrix transposition
            ax_lr.plot(range(iterations), lr, 'b-')


def lr_find(session, data, start_lr=None, end_lr=10):
    """Duplicates session and runs one epoch over data while linearly increasing the learning rate at each step.
    Plots the loss. This method can be used to find a learning rate by choosing a learning rate from the plot where
    the loss is decreasing.
    
    Arguments:
        session {Session} -- Any session 
        data {Dataloader} -- A training dataset
    
    Keyword Arguments:
        start_lr {int or list} -- Learning rate(s) to start from (default: {None})
        end_lr {int or list} -- Learning rate(s) to increase to (default: {10})
    """

    session.save('temp')
    lr_scheduler = LRFindScheduler(len(data), start_lr, end_lr)
    schedule = TrainingSchedule(data, [lr_scheduler])
    session.train(schedule, 1)
    lr_scheduler.plot()
    session.load('temp')
