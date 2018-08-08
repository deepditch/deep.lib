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


class Session():
    def __init__(self, model, criterion, optimizer, scheduler, callbacks=[]):
        self.model = to_gpu(model)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.callbacks = callbacks
        self.add_callback(self.scheduler)

    # Train on a single batch
    def step(self, input, label):
        input = Variable(to_gpu(input))
        label = Variable(to_gpu(label))

        self.optimizer.zero_grad()

        for cb in self.callbacks: cb.on_batch_begin(self)

        outputs = self.model(input)
        loss = self.criterion(outputs, label)
        loss.backward()
        self.optimizer.step()

        for cb in self.callbacks: cb.on_batch_end(self, loss.data.tolist()[0])

        return loss.data.tolist()[0]

    # Train for one pass over the data
    def epoch(self, data):
        for cb in self.callbacks: cb.on_epoch_begin(self)

        loss = 0

        for input, label in tqdm(data, desc="Steps", leave=False):
            loss += self.step(input, label) 

        for cb in self.callbacks: cb.on_epoch_end(self, loss/len(data))

    def train(self, data, epochs):
        with TrainModel(self.model):
            for cb in self.callbacks: cb.on_train_begin(self)       

            for epoch in tqdm(range(epochs), desc="Epochs"):
                self.epoch(data)       

            for cb in self.callbacks: cb.on_train_end(self)

    def add_callback(self, callback):
        self.callbacks.append(callback)
    

class TrainingSchedule():
    def __init__(self, data, epochs, callbacks=[]):
        self.data = data
        self.epochs = epochs
        self.iterations = len(self.data) * self.epochs
        self.callbacks = callbacks


    def add_callback(self, callback):
        self.callbacks.append(callback)


    def train(self, session):  
        with TrainModel(session.model):
            for cb in self.callbacks: cb.on_train_begin(session)

            for epoch in tqdm(range(epochs), desc="Epochs"):

                for cb in self.callbacks: cb.on_epoch_begin(self)

                for input, label in tqdm(self.data, desc="Steps", leave=False): 
                    for cb in self.callbacks: cb.on_batch_begin(self)
                    session.step(input, label)
                    for cb in self.callbacks: cb.on_batch_end(self)

                for cb in self.callbacks: cb.on_epoch_end(self)

            for cb in self.callbacks: cb.on_train_end(session)

    

