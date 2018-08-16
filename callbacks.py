import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm_notebook as tqdm, tnrange
import session as sess
from torch.autograd import Variable
import numpy as np

class TrainCallback:
    def on_train_begin(self, session): pass
    def on_epoch_begin(self, session): pass
    def on_batch_begin(self, session): pass
    def on_batch_end(self, session, lossMeter): pass
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




