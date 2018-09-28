import torch
import torch.nn as nn
import torch.optim as optim
from callbacks import TrainCallback
import matplotlib.pyplot as plt

class _LRScheduler(TrainCallback):
    '''An abstract class representing a learning rate schedule'''

    def __init__(self, iteration=0):      
        '''
        Keyword Arguments:
            iteration {int} -- The iteration the lr_scheduler is started on. 
                Can be increased from 0 to start in the middle of a schedule (default: {0})
        '''

        self.iteration = iteration

    def get_lr(self): 
        '''Gets the next learning rate in the schedule
        
        Raises:
            NotImplementedError -- Sub classes must implement this method
        '''

        raise NotImplementedError 

    def sub_reset(self): 
        '''Resets internal state of the schedule
        
        Raises:
            NotImplementedError -- Sub classes must implement this method. If the sub class has no internal state, can pass
        '''

        raise NotImplementedError

    def reset(self): 
        '''Resets the schedule'''

        self.sub_reset()
        self.iteration = 0

    def on_train_begin(self, session):
        '''Extracts the initial learning rate from the session
        
        Arguments:
            session {Session} -- A training session
        
        Raises:
            KeyError -- If the schedular was initialized with iteration != 0, then the session must 
                have the 'initial_lr' key already set
        '''

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
        '''Increments self.iteration and updates the learning rate
        
        Keyword Arguments:
            iteration {int} -- Can be used to arbitrarily jump to any point in the schedule (default: {None})
        '''

        if iteration is not None:
            self.iteration = iteration    
        self.session.set_lr(self.get_lr())
        self.iteration += 1

    def plot(self, iterations=None):
        '''Plots learning rate against iterations
        
        Keyword Arguments:
            iterations {int} -- The number of iterations to plot. If none, plots for self.iterations (default: {None})
        '''

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
    '''An abstract class that represents a learning rate that is updated after each epoch'''

    def __init__(self, iteration=0):
        super().__init__(iteration)

    def on_epoch_begin(self, session):
        self.step()


class _OnBatchLRScheduler(_LRScheduler):
    '''An abstract class that represents a learning rate that is updated after each batch'''

    def __init__(self, iteration=0):
        super().__init__(iteration)

    def on_batch_begin(self, session):
        self.step()