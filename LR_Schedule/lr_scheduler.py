import torch
import torch.nn as nn
import torch.optim as optim
from callbacks import TrainCallback
import matplotlib.pyplot as plt
import copy

class _LRScheduler(TrainCallback):
    def __init__(self, iteration=0):      
        self.iteration = iteration

    def get_lr(self): raise NotImplementedError 
    def sub_reset(self): raise NotImplementedError

    def reset(self): 
        self.sub_reset()
        self.iteration = 0

    def on_train_begin(self, session):
        self.session = session
        if self.iteration == 0:
            for group in self.session.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(self.session.optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.base_lrs = list(map(lambda group: group['initial_lr'], self.session.optimizer.param_groups))

    def step(self, iteration=None):
        if iteration is not None:
            self.iteration = iteration    
        self.session.set_lr(self.get_lr())
        self.iteration += 1

    def plot(self, iterations=None):
        cp = copy.deepcopy(self)
        if(iterations is None): iterations = cp.iteration
        cp.reset()
        if cp.base_lrs is None: cp.base_lrs = [1]

        lrs = []

        for i in range(iterations):
            lrs.append(cp.get_lr())
            cp.iteration += 1

        fig, ax = plt.subplots()

        ax.set_ylabel("learning rate")
        ax.set_xlabel("iterations")

        for lr in [*zip(*lrs)]: # Matrix transposition
            ax.plot(range(iterations), lr)


class _OnEpochLRScheduler(_LRScheduler):
    def __init__(self, iteration=0):
        super().__init__(iteration)

    def on_epoch_begin(self, session):
        self.step()


class _OnBatchLRScheduler(_LRScheduler):
    def __init__(self, iteration=0):
        super().__init__(iteration)

    def on_batch_begin(self, session):
        self.step()