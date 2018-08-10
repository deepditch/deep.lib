import torch
import torch.nn as nn
import torch.optim as optim
from LR_Schedule.lr_scheduler import *
import callbacks
from torch.autograd import Variable
from tqdm import tqdm_notebook as tqdm, tnrange

USE_GPU = torch.cuda.is_available()
def to_gpu(x, *args, **kwargs):
    '''puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x


class TrainModel():
    def __init__(self, model):
        self.model = model
        self.was_training = model.training

    def __enter__(self):
        self.model.train()

    def __exit__(self, type, value, traceback):
        self.model.train(mode=self.was_training)


class EvalModel():
    def __init__(self, model):
        self.model = model
        self.was_training = model.training

    def __enter__(self):
        self.model.eval()

    def __exit__(self, type, value, traceback):
        self.model.train(mode=self.was_training)


class LossMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0
        self.raw_avg = 0
        self.interpolated_avg = 0
        self.debias = 0
        self.sum = 0
        self.count = 0
        self.batches = 0

    def update(self, loss, n=1):
        self.loss = loss
        self.sum += loss * n
        self.count += n
        self.batches += 1
        self.raw_avg = self.sum / self.count

        # When training on a large dataset, this average weights later batches higher than earlier batches
        self.interpolated_avg = self.interpolated_avg * .98 + loss * (1-.98)
        self.debias = self.interpolated_avg / (1 - .98**self.batches)


class Session():
    def __init__(self, model, criterion, optimizer):
        self.model = to_gpu(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.training = False
    
    def step(self, input, label):
        input = Variable(to_gpu(input))
        label = Variable(to_gpu(label))
        self.optimizer.zero_grad()              # Clear past gradent
        outputs = self.model(input)             # Forward pass
        loss = self.criterion(outputs, label)   # Calculate loss
        loss.backward()                         # Calculate new gradient
        self.optimizer.step()                   # Update model parameters
        return loss.data.tolist()[0]            # Return loss value

    def train(self, schedule):
        self.training = True
        lossMeter = LossMeter()
        with TrainModel(self.model):
            for cb in schedule.callbacks: cb.on_train_begin(self)       
            for epoch in tqdm(range(schedule.epochs), desc="Epochs"):
                if not self.training: break
                for cb in schedule.callbacks: cb.on_epoch_begin(self)
                running_loss = 0
                for input, label in tqdm(schedule.data, desc="Steps", leave=False):
                    if not self.training: break
                    for cb in schedule.callbacks: cb.on_batch_begin(self)
                    step_loss = self.step(input, label)         
                    lossMeter.update(step_loss, label.shape[0])
                    for cb in schedule.callbacks: cb.on_batch_end(self, lossMeter)
                for cb in schedule.callbacks: cb.on_epoch_end(self, lossMeter)      
            for cb in schedule.callbacks: cb.on_train_end(self)    

    def stop_training(self):
        self.training = False
    

class TrainingSchedule():
    def __init__(self, data, epochs, callbacks=[]):
        self.data = data
        self.epochs = epochs
        self.iterations = len(self.data) * self.epochs
        self.callbacks = callbacks

    def add_callback(self, callback):
        self.callbacks.append(callback)



    

