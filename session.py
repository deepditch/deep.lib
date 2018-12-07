import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import callbacks
from tqdm import tqdm_notebook as tqdm, tnrange
import os
import util


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
    def __init__(self, model, criterion, optim_fn, lrs=1e-3, weight_decay=0):
        self.model = util.to_gpu(model)
        self.criterion = criterion    
        self.optim_fn = optim_fn
        param_arr = [{'params':layer.parameters(), 'lr':0} for layer in self.model.children()]
        self.optimizer = self.optim_fn(param_arr, weight_decay=weight_decay)
        self.set_lr(lrs)
        self.running = False

    def save(self, name):
        if not name.endswith('.ckpt.tar'): name += '.ckpt.tar'
        state = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
        }
        torch.save(state, name)

    def load(self, name):
        if not name.endswith('.ckpt.tar'): name += '.ckpt.tar' 
        checkpoint = torch.load(name)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def freeze_to(self, layer_index):
        layers = list(self.model.children())
        for l in layers: 
            for param in l.parameters():
                param.requires_grad = False

        for l in layers[layer_index:]:
            for param in l.parameters():
                param.requires_grad = True
                
    def freeze(self):
        self.freeze_to(-1)

    def unfreeze(self):
        self.freeze_to(0)

    def set_lr(self, lrs):
        lrs = util.listify(lrs, self.optimizer.param_groups)
        if len(lrs) != len(self.optimizer.param_groups):
            raise ValueError("Size Mismatch: Expected lrs of length {} but got {}".format(len(self.optimizer.param_groups), len(lrs)))

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def forward(self, input):
        input = Variable(util.to_gpu(input))
        return self.model(input)

    def step(self, input, label):    
        self.optimizer.zero_grad()                                  # Clear past gradent                                         
        outputs = self.forward(input)                               # Forward pass
        loss = self.criterion(outputs, Variable(util.to_gpu(label)))     # Calculate loss
        loss.backward()                                             # Calculate new gradient
        self.optimizer.step()                                       # Update model parameters
        return loss.data                                            # Return loss value

    def run(self, schedule, epochs):
        self.running = True
        lossMeter = LossMeter()
        for cb in schedule.callbacks: cb.on_train_begin(self)       
        for epoch in tqdm(range(epochs), desc="Epochs"):
            if not self.running: break
            for cb in schedule.callbacks: cb.on_epoch_begin(self)
            running_loss = 0
            for input, label, *_ in tqdm(schedule.data, desc="Steps", leave=False):
                if not self.running: break
                for cb in schedule.callbacks: cb.on_batch_begin(self)
                step_loss = self.step(input, label)         
                lossMeter.update(step_loss, label.shape[0])
                for cb in schedule.callbacks: cb.on_batch_end(self, lossMeter)
            for cb in schedule.callbacks: cb.on_epoch_end(self, lossMeter)      
        for cb in schedule.callbacks: cb.on_train_end(self)   

    def train(self, schedule, epochs):      
        with TrainModel(self.model):
            self.run(schedule, epochs)

    def stop(self):
        self.running = False
    

class TrainingSchedule():
    def __init__(self, data, callbacks=[]):
        self.data = data
        self.callbacks = callbacks

    def add_callback(self, callback):
        self.callbacks.append(callback)
