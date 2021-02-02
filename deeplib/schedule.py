import os
import time
import pickle
from tqdm.notebook import tqdm, trange

import torch
from torch.utils.data import DataLoader

import deeplib.util as util
import deeplib.callbacks
import deeplib.session
import deeplib.schedule


class Logger(deeplib.callbacks.StatelessTrainCallback):
    """A training callback used to log training statistics to the console. 
    Reads statistics from a TrainingSchedule's callback dictionary (`TrainingSchedule.cb_dict`).
    """
    def __init__(self, metrics: list):
        """
        Args:
            metrics (list[str]): List of metrics to log. Each metric is a key used to read from the TrainingSchedule's callback dictionary.
        """
        self.metrics = metrics
        self.widths = [len("Epoch")] + [max(7, len(key))
                                        for key in self.metrics]
        self.printed_header = False

    def divider(self, char="-", sep="+"):
        return sep + sep.join([char * (width+2) for width in self.widths]) + sep

    def format_column(self, columns: list):
        formats = ["{:>" + str(width) + (".4f" if not isinstance(col, (str, int, bool))
                                         else "") + "}" for width, col in zip(self.widths, columns)]
        return ("| " + " | ".join(formats) + " |").format(*columns)

    def on_train_begin(self, session, schedule, cb_dict, *args, **kwargs):
        session.add_meta("Training Log", self.format_column(
            ["Epoch"] + self.metrics) + "\n" + self.divider(sep="|"))
        cb_dict["print-width"] = sum(self.widths) + (len(self.widths) * 3) + 1

    def on_train_end(self, *args, **kwargs):
        tqdm.write(self.divider())

    def print_header(self):
        tqdm.write(self.divider())
        tqdm.write(self.format_column(["Epoch"] + self.metrics))
        tqdm.write(self.divider("="))

    def on_epoch_end(self, session, schedule, cb_dict, *args, **kwargs):
        if not self.printed_header:
            self.print_header()
            self.printed_header = True

        columns = [schedule.epoch+1] + [cb_dict[key] if key in cb_dict and cb_dict[key] is not None else "None" for key in self.metrics]
        metrics_string = self.format_column(columns)

        tqdm.write(metrics_string)

        session.append_meta("Training Log", "\n" + metrics_string)


class TrainingSchedule():
    """This class is used as a container for a torch.utils.data.DataLoader and a list of training callbacks. 
    The training schedule is passed to the `session.train` method.
    Each training callback has access to a shared callback dictionary (`TrainingSchedule.cb_dict`). 
    Callbacks can coordinate by reading and modifying this dictionary. For example, one callback might write a 
    training statistic to the callback dictionary and a second callback might read that statistic and log to the console.
    """
    def __init__(self, dataloader: torch.utils.data.DataLoader, num_epochs: int, callbacks: list):
        """Initialize a training schedule

        Args:
            dataloader (torch.utils.data.DataLoader): A dataloader used to iterate the training set
            num_epochs (int): Number of epochs to train for
            callbacks (list[deeplib.callbacks.TrainCallback]): List of training callbacks
        """
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.epoch = 0
        self.iteration = 0
        self.callbacks = callbacks
        self.cb_dict = {}
        self.metrics = []

        for cb in self.callbacks:
            new_metric = cb.register_metric()

            if new_metric is None:
                continue

            if not isinstance(new_metric, list):
                new_metric = [new_metric]

            self.metrics += new_metric

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

    def on_batch_begin(self, session, *args, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_begin(session, self, *args, **kwargs)

    def on_batch_end(self, session, step_loss, output, *args, **kwargs):
        for cb in self.callbacks:
            cb.on_batch_end(session, self, self.cb_dict,
                            step_loss, output, *args, **kwargs)

    def on_before_optim(self, session, step_loss, output, *args, **kwargs):
        for cb in self.callbacks:
            cb.on_before_optim(session, self, self.cb_dict, step_loss, output, *args, **kwargs)

    def on_epoch_end(self, session):
        for cb in self.callbacks:
            cb.on_epoch_end(session, self, self.cb_dict)

    def on_train_end(self, session):
        for cb in self.callbacks:
            cb.on_train_end(session, self, self.cb_dict)
