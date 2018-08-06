import torch
import torch.nn as nn
import torch.optim as optim
import lr_scheduler
import callbacks

class Session():
    def __init__(model, criterion, optimizer, scheduler):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.callbacks = []

    # Train on a single batch
    def step(self, input, label):
        input = input.to(self.device)
        label = label.to(self.device)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = self.model.forward(input)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

        for cb in self.callbacks: cb.on_batch_end(self)


    # Train for one pass over the data
    def epoch(self, dataLoader):
        for cb in self.callbacks: cb.on_epoch_begin()

        self.scheduler.step()

        for input, label in dataLoader:
            self.step(input, label) 

        for cb in self.callbacks: cb.on_epoch_end()


    def train(self, dataLoader, epochs):
        for cb in self.callbacks: cb.on_train_begin()

        for epoch in range(epochs):
            self.epoch(dataLoader)       

        for cb in self.callbacks: cb.on_train_end()
    
