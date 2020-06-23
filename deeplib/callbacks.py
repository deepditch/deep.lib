import os
import math
import time
import psutil
import pickle
import numpy as np
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter


class TrainCallback:
    def on_train_begin(self, session, schedule, cb_dict): pass
    def on_epoch_begin(self, session, schedule, cb_dict): pass
    def on_batch_begin(self, session, schedule, cb_dict, input, label): pass
    def on_batch_end(self, session, schedule, cb_dict, loss, input, output, label): pass
    def on_epoch_end(self, session, schedule, cb_dict): pass
    def on_train_end(self, session, schedule, cb_dict): pass
    def state_dict(self): return pickle.dumps(self.__dict__)
    def load_state_dict(self, state_dict): self.__dict__.update(pickle.loads(state_dict)) 
    def register_metric(self): pass


class StatelessTrainCallback(TrainCallback):
  def state_dict(self): return None
  def load_state_dict(self, state_dict): pass 
      

class Saver(TrainCallback):
    def __init__(self, dir):
        self.dir = dir

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.best = 0
        self.epoch = 0

    def on_epoch_end(self, session, *args, **kwargs):
        self.epoch += 1
        session.save('model.%d' % self.epoch)


class SaveBest(TrainCallback):
    def __init__(self, model_path: str, metric_name: str, higher_is_better: bool):
        self.model_path = model_path
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self.best = float("-inf") if higher_is_better else float("inf")

    def on_epoch_end(self, session, schedule, cb_dict, *args, **kwargs):
      if (self.best < cb_dict[self.metric_name] and self.higher_is_better) or (self.best > cb_dict[self.metric_name] and not self.higher_is_better):
        self.best = cb_dict[self.metric_name]
        session.save(self.model_path)
        session.add_meta(f"Best {self.metric_name}", f"{self.best} at epoch {schedule.epoch}")


MEGA = 10 ** 6
MEGA_STR = ' ' * MEGA

class MemoryProfiler(StatelessTrainCallback):  
    def print_profile(self, prefix):
        process = psutil.Process(os.getpid())
        total, available, percent, used, free = psutil.virtual_memory()
        total, available, used, free = total / MEGA, available / MEGA, used / MEGA, free / MEGA
        proc = process.memory_info()[1] / MEGA
        tqdm.write('process = %.2f total = %.2f available = %.2f used = %.2f free = %.2f percent = %.2f' % (proc, total, available, used, free, percent))

    def on_train_begin(self, *args, **kwargs): self.print_profile("on_train_begin")
    def on_epoch_end(self, *args, **kwargs): self.print_profile("on_epoch_end")
    def on_train_end(self, *args, **kwargs): self.print_profile("on_train_end")


class GPUMemoryProfiler(StatelessTrainCallback):
  def print_profile(self):
    stats = torch.cuda.memory_stats()
    print(f"{stats['allocated_bytes.all.current']:<30,} {stats['allocated_bytes.all.peak']:<30,} {torch.cuda.get_device_properties('cuda').total_memory:<30,}")

  def on_epoch_begin(self, *args, **kwargs):
    print(f"{'current': <30} {'peak': <30} {'total': <30}")

  def on_batch_begin(self, *args, **kwargs): 
    self.print_profile()

  def on_batch_end(self, *args, **kwargs): 
    self.print_profile()  


class TrainingLossLogger(TrainCallback):
  def __init__(self, metric_name = "Loss/Train"):
    self.metric_name = metric_name
    self.loss_meter = LossMeter()

  def register_metric(self):
    return self.metric_name

  def on_epoch_begin(self, *args, **kwargs):
    self.loss_meter.reset()

  def on_batch_end(self, session, schedule, cb_dict, loss, *args, **kwargs):
    cb_dict[self.metric_name] = loss
    self.loss_meter.update(loss)

  def on_epoch_end(self, session, schedule, cb_dict): 
    cb_dict[self.metric_name] = self.loss_meter.raw_avg


class TensorboardLogger(StatelessTrainCallback):
    def __init__(self, directory="./runs/", on_batch_metrics=[], on_epoch_metrics=[]):
      self.writer = SummaryWriter(log_dir=directory)
      self.on_batch_metrics = on_batch_metrics
      self.on_epoch_metrics = on_epoch_metrics

    def on_train_begin(self, session, schedule, *args, **kwargs):
      if len(self.on_epoch_metrics) == 0:
        self.on_epoch_metrics = schedule.metrics

    def on_batch_end(self, session, schedule, cb_dict, loss, input, output, label):
      if self.on_batch_metrics == []:
        self.writer.add_scalar('Loss/train', loss, schedule.iteration)
      else:
        for metric in self.on_batch_metrics:
          if metric in cb_dict and cb_dict[metric] is not None:
            self.writer.add_scalar(metric, cb_dict[metric], schedule.iteration)

    def on_epoch_end(self, session, schedule, cb_dict):
      for metric in self.on_epoch_metrics:
        if metric in cb_dict and cb_dict[metric] is not None:
          self.writer.add_scalar(metric, cb_dict[metric], schedule.iteration)

    def __del__(self):
        self.writer.close()


class OptimizerStepper(TrainCallback):
  def __init__(self, optimizer):
    super().__init__()
    self.optimizer = optimizer

  def state_dict(self): return self.optimizer.state_dict()
  def load_state_dict(self, state_dict): self.optimizer.load_state_dict(state_dict)

  def on_epoch_end(self, *args, **kwargs):
    self.optimizer.step()
    self.optimizer.zero_grad()


class Checkpoint(StatelessTrainCallback):
  def __init__(self, ckpt_file, interval=5*60, reset=False):
      self.ckpt_file = ckpt_file
      if not self.ckpt_file.endswith('.ckpt.tar'): self.ckpt_file += '.ckpt.tar'
      self.interval = interval
      self.reset = reset

  def on_train_begin(self, session, *args, **kwargs):
      if os.path.exists(self.ckpt_file) and not self.reset: 
        print("--- LOADING CHECKPOINT ---")
        session.load(self.ckpt_file)
      self.start_time = time.time()      

  def on_batch_end(self, session, schedule, cb_dict, *args, **kwargs):
      end = time.time()
      elapsed = end - self.start_time

      if elapsed > self.interval:
          self.start_time = end
          session.checkpoint(self.ckpt_file)
          if "print-width" in cb_dict:
            half_width = (cb_dict["print-width"] - 9) / 2
            left = "+" + ("-" * (math.floor(half_width) - 1))
            right = ("-" * (math.ceil(half_width) - 1)) + "+"
            print(left + " CHECKPOINT " + right)
          else:
            print("--- CHECKPOINT ---")