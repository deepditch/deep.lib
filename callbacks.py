import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm_notebook as tqdm, tnrange
import session as sess
from torch.autograd import Variable
import numpy as np
import os
import psutil
from tqdm import tqdm


class TrainCallback:
    def on_train_begin(self, session): pass
    def on_epoch_begin(self, session): pass
    def on_batch_begin(self, session): pass
    def on_batch_end(self, session, lossMeter, output, label): pass
    def on_epoch_end(self, session, lossMeter): pass
    def on_train_end(self, session): pass
        

class Saver(TrainCallback):
    def __init__(self, dir):
        self.dir = dir

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.best = 0
        self.epoch = 0

    def on_epoch_end(self, session, lossMeter):
        self.epoch += 1
        session.save('model.%d' % self.epoch)

MEGA = 10 ** 6
MEGA_STR = ' ' * MEGA

class MemoryProfiler(TrainCallback):  
    def print_profile(self, prefix):
        process = psutil.Process(os.getpid())
        total, available, percent, used, free = psutil.virtual_memory()
        total, available, used, free = total / MEGA, available / MEGA, used / MEGA, free / MEGA
        proc = process.memory_info()[1] / MEGA
        tqdm.write('process = %.2f total = %.2f available = %.2f used = %.2f free = %.2f percent = %.2f' % (proc, total, available, used, free, percent))

    def on_train_begin(self, session): self.print_profile("on_train_begin")
    # def on_epoch_begin(self, session): self.print_profile("on_epoch_begin")
    # def on_batch_begin(self, session): self.print_profile("on_batch_begin")
    # def on_batch_end(self, session, lossMeter): self.print_profile("on_batch_end")
    def on_epoch_end(self, session, lossMeter): self.print_profile("on_epoch_end")
    def on_train_end(self, session): self.print_profile("on_train_end")


class LossLogger(TrainCallback):
    def on_epoch_end(self, session, lossMeter): 
        tqdm.write(f"Training Loss: {lossMeter.debias}")