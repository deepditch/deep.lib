import copy
import pickle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from deeplib.callbacks import TrainCallback

class _LRScheduler(TrainCallback):
    '''An abstract class representing a learning rate schedule'''

    def __init__(self, iteration=0): 
        '''
        Keyword Arguments:
            iteration {int} -- The iteration the lr_scheduler is started on. 
                Can be increased from 0 to start in the middle of a schedule (default: {0})
        '''

        self.iteration = iteration

    def state_dict(self):
        return pickle.dumps({k: self.__dict__[k] for k in set(list(self.__dict__.keys())) - set(["session"])})

    def get_lr(self): 
        '''Gets the next learning rate in the schedule
        
        Raises:
            NotImplementedError -- Sub classes must implement this method
        '''

        raise NotImplementedError 

    def should_get_mom(self): return False

    def get_mom(self): pass

    def sub_reset(self): 
        '''Resets internal state of the schedule
        
        Raises:
            NotImplementedError -- Sub classes must implement this method. If the sub class has no internal state, can pass
        '''

        raise NotImplementedError

    def reset(self): 
        '''Resets the schedule'''

        self.sub_reset()
        self.base_lrs = None
        self.iteration = 0

    def on_train_begin(self, session, *args, **kwargs):
        '''Extracts the initial learning rate from the session
        
        Arguments:
            session {Session} -- A training session
        
        Raises:
            KeyError -- If the schedular was initialized with iteration != 0, then the session must 
                have the 'initial_lr' key already set
        '''

        if self.iteration == 0:
            for group in session.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(session.optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.base_lrs = list(map(lambda group: group['initial_lr'], session.optimizer.param_groups))

    def step(self, session, iteration=None):
        '''Increments self.iteration and updates the learning rate
        
        Keyword Arguments:
            iteration {int} -- Can be used to arbitrarily jump to any point in the schedule (default: {None})
        '''

        if iteration is not None:
            self.iteration = iteration    
        session.set_lr(self.get_lr())
        if self.should_get_mom():
            session.set_mom(self.get_mom())
        self.iteration += 1

    def plot(self, iterations=None, lrs=[1]):
        '''Plots learning rate against iterations
        
        Keyword Arguments:
            iterations {int} -- The number of iterations to plot. If none, plots for self.iterations (default: {None})
        '''

        cp = copy.deepcopy(self)
        if iterations is None: iterations = cp.iteration
        cp.reset()
        if cp.base_lrs is None: cp.base_lrs = lrs

        lrs = []
        moms = []

        for i in range(iterations):
            lrs.append(cp.get_lr())
            moms.append(cp.get_mom())
            cp.iteration += 1

        fig, ax_lr = plt.subplots()   
        ax_mom = ax_lr.twinx()

        ax_mom.set_xlabel("Iteration")
        ax_mom.set_ylabel("Momentum", color='g')
        ax_lr.set_ylabel("Learning Rate", color='b')

        ax_mom.plot(range(iterations), moms, 'g-')
    
        for lr in [*zip(*lrs)]: # Matrix transposition
            ax_lr.plot(range(iterations), lr, 'b-')

class _OnEpochLRScheduler(_LRScheduler):
    '''An abstract class that represents a learning rate that is updated after each epoch'''

    def __init__(self, iteration=0):
        super().__init__(iteration)

    def on_epoch_begin(self, session, *args, **kwargs):
        self.step(session)

class _OnBatchLRScheduler(_LRScheduler):
    '''An abstract class that represents a learning rate that is updated after each batch'''

    def __init__(self, iteration=0):
        super().__init__(iteration)

    def on_batch_begin(self, session, *args, **kwargs):
        self.step(session)

class TorchLRScheduleCallback(TrainCallback):
    def __init__(self, schedule_fn, *args, **kwargs):
        self.schedule_fn = schedule_fn
        self.args = args
        self.kwargs = kwargs

    def on_train_begin(self, session, schedule, cb_dict): 
        self.schedule = self.schedule_fn(session.optimizer, *self.args, **self.kwargs)

    def state_dict(self): return self.schedule.state_dict()

    def load_state_dict(self, state_dict): self.schedule.load_state_dict()

    def register_metric(self): return 'Learning Rate'

    def on_epoch_begin(self, session, schedule, cb_dict): cb_dict['Learning Rate'] = self.schedule.get_last_lr()

class TorchOnBatchLRScheduleCallback(TorchLRScheduleCallback):    
    def on_batch_end(self, session, *args, **kwargs):
        self.schedule.step()

class TorchOnEpochLRScheduleCallback(TorchLRScheduleCallback):    
    def on_epoch_end(self, session, *args, **kwargs):
        self.schedule.step()