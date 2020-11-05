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

apex_installed = True

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    apex_installed = False

torch_xla_installed = True

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    torch_xla_installed = False
                

class Session():
    def __init__(self, model, criterion, optim_fn, lrs=1e-3, tpu=False **kwargs):
        self.tpu = tpu

        if self.tpu and not torch_xla_installed: raise ImportError("Please install torch_xla to use tpu training.")

        else: 
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = model.to(self.device)

        self.criterion = criterion    
        self.optim_fn = optim_fn
        param_arr = [{'params':layer.parameters(), 'lr':0} for layer in self.model.children()]
        self.optimizer = self.optim_fn(param_arr, **kwargs) # Initialize with learning rate of 0
        self.set_lr(lrs) # Update learning rate from passed lrs
        self.running = False
        self.mixed_precision = False
        self.tpu = False
        self.schedule = None
        self.meta = {}

    def _save_meta(self, name):
        file = os.path.splitext(name)[0] + ".meta.md"

        with open(file, mode="w") as f:
            for key, val in self.meta.items():
                f.write(f"## {key} \n")
                f.write(f"{val} \n\n")

    def _save(self, name):
        state = {
            'model': self.model.state_dict()
        }

        torch.save(state, name)

        self._save_meta(name)

    def _checkpoint(self, name):
        state = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'amp': amp.state_dict() if self.mixed_precision else None,
            'schedule': self.schedule.state_dict() if self.schedule != None else None
        }

        torch.save(state, name)

        self._save_meta(name)

    def _save_model(self, name):
        state = self.model.state_dict()
        torch.save(state, name)

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

    def save(self, name):
        a = Thread(target=Session._save, args=(self, name))
        a.start()
        a.join()

    def save_model(self, name):
        a = Thread(target=Session._save_model, args=(self, name))
        a.start()
        a.join()

    def checkpoint(self, name):
        a = Thread(target=Session._checkpoint, args=(self, name))
        a.start()
        a.join()

    def load(self, name, map_location=None):
        checkpoint = torch.load(name, map_location=map_location)
        
        if 'model' in checkpoint: self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'schedule' in checkpoint and self.schedule is not None: self.schedule.load_state_dict(checkpoint['schedule'])
        if 'amp' in checkpoint and checkpoint['amp'] is not None and self.mixed_precision: amp.load_state_dict(checkpoint['amp'])
    
    def load_model(self, name):
        model_dict = torch.load(name, map_location=None)
        self.model.load_state_dict(model_dict)

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

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def set_lr(self, lrs):
        lrs = util.listify(lrs, self.optimizer.param_groups)
        if len(lrs) != len(self.optimizer.param_groups):
            raise ValueError("Size Mismatch: Expected lrs of length {} but got {}".format(len(self.optimizer.param_groups), len(lrs)))

        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr

    def set_mom(self, mom):
        if 'betas' in self.optimizer.param_groups[0]:
            for pg in self.optimizer.param_groups: pg['betas'] = (mom, pg['betas'][1])
        else:
            for pg in self.optimizer.param_groups: pg['momentum'] = mom

    def forward(self, input):
        if isinstance(input, dict):
            input = {key: Variable(value.to(self.device)) for key, value in input.items()}  
            return self.model(**input), input
        else:
            input = Variable(input.to(self.device))
            return self.model(Variable(input.to(self.device))), input

    def step(self, input, label):                              
        outputs, input = self.forward(input) 

        if isinstance(label, dict):
            label = {key: Variable(value.to(self.device)) for key, value in label.items()}  
        else:
            label = Variable(label.to(self.device))
                
        loss = self.criterion(outputs, label)

        self.optimizer.zero_grad()     

        if self.mixed_precision:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss: 
                scaled_loss.backward()
        elif self.tpu:
            xm.optimizer_step(optimizer)
        else: 
            loss.backward()  

        self.optimizer.step()

        return loss.data, input, outputs, label                                 

    def run(self, schedule):
        with TrainModel(self.model):
            self.running = True
            self.schedule = schedule

            schedule.on_train_begin(self)       
            
            for epoch in self.schedule:
                if not self.running: break
                
                self.schedule.on_epoch_begin(self)
                
                for input, label, *_ in self.schedule.data():
                    if not self.running: break
                    self.schedule.on_batch_begin(self, input, label)
                    step_loss, input, output, label = self.step(input, label)  
                    self.schedule.on_batch_end(self, step_loss, input, output, label)
                
                self.schedule.on_epoch_end(self)      

            self.schedule.on_train_end(self)   

    @staticmethod
    def xmp_run_wrapper(idx, self, schedule): 
        self.device = xm.xla_device()
        self.run(schedule)

    def train(self, schedule):
        if not self.tpu: self.run(schedule)
        else: xmp.spawn(Session.xmp_run_wrapper, args=(self, schedule), nprocs=8, start_method='fork')

    def stop(self):
        self.running = False

    def to_fp16(self):
        if not apex_installed: raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use MixedPrecision training.")
        self.mixed_precision = True
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
