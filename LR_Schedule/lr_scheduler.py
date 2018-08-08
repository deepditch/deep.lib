import torch
import torch.nn as nn
import torch.optim as optim
from callbacks import TrainCallback
import matplotlib.pyplot as plt

class _LRScheduler(TrainCallback):
    def __init__(self, iteration=0):      
        self.iteration = iteration

    def on_train_begin(self, session):
        self.optimizer = session.optimizer
        if self.iteration == 0:
            for group in self.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError 

    def set_lr(self, lrs):
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def step(self, iteration=None):
        if iteration is not None:
            self.iteration = iteration    
        self.set_lr(self.get_lr())
        self.iteration += 1

    def plot(self, iterations=None):
        save_iter = self.iteration
        self.iteration = 0

        if(iterations is None): iterations = save_iter

        lrs = []

        for i in range(iterations):
            lrs.append(self.get_lr())
            self.iteration += 1

        fig, ax = plt.subplots()

        ax.set_ylabel("learning rate")
        ax.set_xlabel("iterations")

        for lr in [*zip(*lrs)]: # Matrix transposition
            ax.plot(range(iterations), lr)

        # ax.set_xscale('log')

        self.iteration = save_iter


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