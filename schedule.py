import torch
from torch.util.data import DataLoader
import util
import callbacks
from tqdm.notebook import tqdm
import os
import time
import pickle
import callbacks
import session
from threading import Thread


class Logger(StatelessTrainCallback):
  def __init__(self, metrics):
    self.metrics = metrics  
    self.widths = [len("Epoch")] + [min(7, len(key)) for key in self.metrics]
    self.format_string = "| " + " | ".join([f"\{{ :<{self.metrics}.4f\}}" for key in self.metrics]) + " |"

  def divider(char="-"):
    return "+" + "+".join([char * (width+2) for width in self.widths]) + "+"

  def on_train_begin(self, session, schedule, cb_dict, *args, **kwargs):
    print(self.divider())
    print(self.format_string.format(["Epoch"] + self.metrics))
    print(self.divider("="))

  def on_epoch_end(self, session, schedule, cb_dict, *args, **kwargs): 
    metrics = [schedule.epoch] + [val for key, val in cb_dict.items if key in self.metrics]
    print(self.format_string.format(metrics))
    print(self.divider())


class TrainingSchedule():
    def __init__(self, dataloader: DataLoader, num_epochs: int, callbacks=[]: list[TrainCallback]):
        self.dataloader = dataloader
        self.callbacks = callbacks

        self.metrics = []

        for cb in self.callbacks:
            new_metric = cb.register_metric()

            if new_metric:
                self.metrics += cb.register_metric()

        if len(self.metrics) > 0:
            self.callbacks.append(Logger(self.metrics))

        self.num_epochs = num_epochs

    def data(self):
        for data in tqdm(self.dataloader, desc=f"Epoch {self.epoch+1}", leave=False):
            self.iteration += 1
            yield data

    def __iter__(self):
        for i in trange(self.epoch, self.num_epochs, initial=self.epoch, total=self.num_epochs):
            self.epoch = i
            yield i

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def state_dict(self):
        callbacks_state = [callback.state_dict() for callback in self.callbacks]
        return pickle.dumps({'callbacks': callbacks_state, 'cb_dict': self.cb_dict, 'epoch': self.epoch, 'iteration': self.iteration})

    def load_state_dict(self, state_dict):
        state_dict = pickle.loads(state_dict)

        for cb, cb_state_dict in zip(self.callbacks, state_dict['callbacks']): 
            cb.load_state_dict(cb_state_dict)

        self.epoch = state_dict['epoch']
        self.iteration = state_dict['iteration']
        self.cb_dict = state_dict['cb_dict']

    def on_train_begin(self, session):        
        for cb in self.callbacks:
            cb.on_train_begin(session, self, self.cb_dict)

    def on_epoch_begin(self, session):              
        for cb in self.callbacks:
            cb.on_epoch_begin(session, self, self.cb_dict)

    def on_batch_begin(self, session, input, label): 
        for cb in self.callbacks:
            cb.on_batch_begin(session, self, self.cb_dict, input, label)

    def on_batch_end(self, session, step_loss, input, output, label): 
        for cb in self.callbacks:
            cb.on_batch_end(session, self, self.cb_dict, step_loss, input, output, label)
            
    def on_epoch_end(self, session): 
        for cb in self.callbacks:
            cb.on_epoch_end(session, self, self.cb_dict)

    def on_train_end(self, session): 
        for cb in self.callbacks:
            cb.on_train_end(session, self, self.cb_dict)