import os
import time
import pickle
from tqdm.notebook import tqdm, trange

import torch
from torch.utils.data import DataLoader

import deeplib.util as util
from deeplib.callbacks import StatelessTrainCallback


class Logger(StatelessTrainCallback):
  def __init__(self, metrics):
    self.metrics = metrics  
    self.widths = [len("Epoch")] + [max(7, len(key)) for key in self.metrics]
    self.printed_header = False

  def divider(self, char="-", sep="+"):
    return sep + sep.join([char * (width+2) for width in self.widths]) + sep

  def format_column(self, columns: list):
    formats = ["{:>" + str(width) + (".4f" if not isinstance(col, (str, int, bool)) else "") + "}" for width, col in zip(self.widths, columns)]
    return ("| " + " | ".join(formats)  + " |").format(*columns)

  def on_train_begin(self, session, schedule, cb_dict, *args, **kwargs):
    session.add_meta("Training Log", self.format_column(["Epoch"] + self.metrics) + "\n" + self.divider(sep="|"))

  def print_header(self):
    print(self.divider())
    print(self.format_column(["Epoch"] + self.metrics))
    print(self.divider("="))

  def on_epoch_end(self, session, schedule, cb_dict, *args, **kwargs): 
    if not self.printed_header:
      self.print_header()
      self.printed_header = True

    columns = [schedule.epoch] + [val for key, val in cb_dict.items() if key in self.metrics]
    metrics_string = self.format_column(columns)

    print(metrics_string)
    print(self.divider())

    session.append_meta("Training Log", "\n" + metrics_string)


class TrainingSchedule():
    def __init__(self, dataloader: DataLoader, num_epochs: int, callbacks):
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epoch = 0
        self.iteration = 0
        self.callbacks = callbacks
        self.cb_dict = {}
        self.metrics = []

        for cb in self.callbacks:
            new_metric = cb.register_metric()

            if new_metric is None: continue

            if not isinstance(new_metric, list):
                new_metric = [new_metric]
            
            self.metrics += new_metric

        print(self.metrics)

        if len(self.metrics) > 0:
            self.callbacks.append(Logger(self.metrics))

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