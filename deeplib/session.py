import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import deeplib.util as util
from deeplib.util import *
import deeplib.callbacks

import os
import time
import pickle
from threading import Thread

class Session():
    """A training session is used to train a pytorch model with support for checkpointing and 
    recording training statistics. Can be overriden for custom training behavior.
    """
    def __init__(self, model: torch.nn.Module, criterion: callable, optimizer: torch.optim.Optimizer):
        """
        Args:
            model (torch.nn.Module): The PyTorch model to train. 
            criterion (callable): A callable function that returns the loss. Should take parameters (output, label) where output is the model's output and label is the label return by the torch.utils.data.Dataset
            optimizer (torch.optim.Optimizer): A PyTorch optimizer. The optimizer should already be initialzied with the model's parameters.
        """
        self.model = util.to_gpu(model)
        self.criterion = criterion    
        self.optimizer = optimizer
        self.running = False
        self.schedule = None
        self.meta = {}

    def _save_meta(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        file = os.path.splitext(path)[0] + ".meta.md"

        with open(file, mode="w") as f:
            for key, val in self.meta.items():
                f.write(f"## {key} \n")
                f.write(f"{val} \n\n")

    def _save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            'model': self.model.state_dict()
        }

        torch.save(state, path)

        self._save_meta(path)

    def _checkpoint(self, path):
        state = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'schedule': self.schedule.state_dict() if self.schedule != None else None
        }

        torch.save(state, path)

        self._save_meta(path)

    def get_meta(self, key: str):
        return self.meta[key] if key in self.meta else None

    def add_meta(self, key: str, desc: str):
        if not isinstance(key, str):
            raise TypeError("key must be a string")

        self.meta[key] = str(desc)

    def append_meta(self, key: str, desc: str):
        if not isinstance(key, str):
            raise TypeError("key must be a string")

        if not key in self.meta:
            self.meta[key] = ""

        self.meta[key] += desc

    def save(self, path):
        """Save the session's model. This method will additionally save a markdown file to `${path.basename}.meta.md`
        containing statistics gathered during training.

        Args:
            path (str): File path where the model is saved. Path must exist.
        """
        a = Thread(target=Session._save, args=(self, path))
        a.start()
        a.join()

    def checkpoint(self, path):
        """Save a training checkpoint. The checkpoint will contain the state of the model, optimizer, and training schedule. 
        This method will additionally save a markdown file to `${path.basename}.meta.md` containing statistics gathered during training.

        Args:
            path (str): File path where the checkpoint is saved. Path must exist.
        """
        a = Thread(target=Session._checkpoint, args=(self, path))
        a.start()
        a.join()

    def load(self, path):
        """Load a checkpoint file created from calling either the `Session.save` or `Session.checkpoint` methods

        Args:
            path (str): File path to load
        """
        checkpoint = torch.load(path)
        
        if 'model' in checkpoint: self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'schedule' in checkpoint and self.schedule is not None: self.schedule.load_state_dict(checkpoint['schedule'])
        if 'amp' in checkpoint and checkpoint['amp'] is not None and self.mixed_precision: amp.load_state_dict(checkpoint['amp'])
    
    def load_model(self, name):
        model_dict = torch.load(name, map_location=None)
        self.model.load_state_dict(model_dict)

    def set_lr(self, lrs):
        lrs = util.listify(lrs, self.optimizer.param_groups)
        if len(lrs) != len(self.optimizer.param_groups):
            raise ValueError("Size Mismatch: Expected lrs of length {} but got {}".format(len(self.optimizer.param_groups), len(lrs)))

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def to_device(self, tensor):
        if isinstance(tensor, dict):
            return {key: self.to_device(value) for key, value in tensor.items()}
        elif isinstance(tensor, tuple):
            return tuple(self.to_device(x) for x in tensor)
        elif isinstance(tensor, list):
            return [self.to_device(x) for x in tensor]
        elif isinstance(tensor, torch.Tensor):
            return Variable(util.to_gpu(tensor))
        else: 
            return tensor

    def forward(self, input):
        """A forward pass through the model

        Args:
            input (torch.tensor): The model's input, must be on the same device as the model.

        Returns:
            torch.tensor: The model's output. Must be on the same device as the input.
        """
        if isinstance(input, dict):
            return self.model(**input)
        else:
            return self.model(input)

    def step(self, input: torch.tensor, label: torch.tensor):
        """A single training step. Can be overridden to define custom behavior. This method will receive the output of a single iteration of the dataloader 
        from the deeplib.schedule.TrainingSchedule.

        Args:
            input (torch.tensor): The model's input
            label (torch.tensor): The label corresponding to the input.

        Returns:
            output (torch.tensor): The model's output given the input. Must be on the same device as the input.
            loss (scalar): The loss computed using `this.criterion`.
        """
        output = self.forward(input)             
        loss = self.criterion(output, label)
        return output, loss                         

    def run(self, schedule):
        self.running = True
        self.schedule = schedule

        schedule.on_train_begin(self)       
        
        for epoch in self.schedule:
            if not self.running: break
            
            self.schedule.on_epoch_begin(self)
            
            for item in self.schedule.data():
                if not self.running: break
                
                self.schedule.on_batch_begin(self, *item)

                item = self.to_device(item)
                output, loss = self.step(*item) 

                self.model.zero_grad()     

                loss.backward()  

                self.schedule.on_before_optim(self, loss, output, *item)
                self.optimizer.step()
                self.schedule.on_batch_end(self, loss.data, output, *item)
            
            self.schedule.on_epoch_end(self)      

        self.schedule.on_train_end(self)   

    def train(self, schedule: 'deeplib.schedule.TrainingSchedule'):   
        """Train the session's model using a training schedule.

        Args:
            schedule (deeplib.schedule.TrainingSchedule): The training schedule used to train the session's model.
        """
        with TrainModel(self.model):
            self.run(schedule)

    def stop(self):
        self.running = False
