import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import util
import callbacks
from tqdm.notebook import tqdm
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
    def __init__(self, model, criterion, optim_fn, lrs=1e-3, log=True, reset=False, **kwargs):
        self.model = util.to_gpu(model)
        self.criterion = criterion    
        self.optim_fn = optim_fn
        param_arr = [{'params':layer.parameters(), 'lr':0} for layer in self.model.children()]
        self.optimizer = self.optim_fn(param_arr, **kwargs) # Initialize with learning rate of 0
        self.set_lr(lrs) # Update learning rate from passed lrs
        self.running = False
        self.log = log
        self.epoch = 0
        self.reset = reset
        self.mixed_precision = False
        self.schedule = None
        self.meta = {}

    def _save_meta(self, name):
        file = os.path.splitext(name)[0] + ".meta.md"

        with open(file, mode="w") as f:
            for key, val in self.meta.items():
                f.write(f"## {key} \n")
                f.write(f"{val} \n")


    def _save(self, name):
        if not name.endswith('.ckpt.tar'): name += '.ckpt.tar'

        state = {
            'model': self.model.state_dict(),
            'mixed_precision': self.mixed_precision
        }

        torch.save(state, name)

        self._save_meta(name)

    def _checkpoint(self, name):
        if not name.endswith('.ckpt.tar'): name += '.ckpt.tar'

        state = {
            'model': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'amp': amp.state_dict() if self.mixed_precision else None,
            'schedule': self.schedule.state_dict() if self.schedule != None else None,
            'epoch': self.epoch,
            'mixed_precision': self.mixed_precision
        }

        torch.save(state, name)

        self._save_meta(name)

    def add_meta(self, key: str, desc: str):
        if not isinstance(key, str):
            raise TypeError("key must be a string")

        self.meta[key] = str(desc)

    def save(self, name):
        a = Thread(target=Session._save, args=(self, name))
        a.start()
        a.join()

    def checkpoint(self, name):
        a = Thread(target=Session._checkpoint, args=(self, name))
        a.start()
        a.join()

    def load(self, name, map_location=None):
        if not name.endswith('.ckpt.tar'): name += '.ckpt.tar' 
        checkpoint = torch.load(name, map_location=map_location)
        
        if 'model' in checkpoint: self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint: self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint: self.epoch = checkpoint['epoch']
        if 'schedule' in checkpoint and self.schedule is not None: self.schedule.load_state_dict(checkpoint['schedule'])
        if 'mixed_precision' in checkpoint: self.mixed_precision = checkpoint['mixed_precision']
        if 'amp' in checkpoint and checkpoint['amp'] is not None: amp.load_state_dict(checkpoint['amp'])

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
        return self.model(Variable(util.to_gpu(input)))

    def step(self, input, label):                                   
        outputs = self.forward(input) 

        if isinstance(label, dict):
            label = {key: Variable(value) for key, value in label.items()}  
        else:
            label = Variable(util.to_gpu(label))     
        loss = self.criterion(outputs, label)

        if self.mixed_precision:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss: scaled_loss.backward()
        else: loss.backward()  

        self.optimizer.step()                                       
        self.optimizer.zero_grad()                                        
        return loss.data, outputs                                 

    def run(self, schedule, checkpoint_file=None, reset=False, ckpt_interval=5*60):
        self.running = True
        self.schedule = schedule

        if checkpoint_file != None and not checkpoint_file.endswith('.ckpt.tar'): checkpoint_file += '.ckpt.tar'
        if checkpoint_file != None and os.path.exists(checkpoint_file) and not reset: 
            print("--- LOADING CHECKPOINT ---")
            self.load(checkpoint_file)
            if self.mixed_precision: self.to_fp16()

        lossMeter = LossMeter()

        start = time.time()

        for cb in schedule.callbacks: cb.on_train_begin(self)       
        
        for epoch in tqdm(range(schedule.epochs), desc="Epochs", initial=self.epoch):
            if not self.running: break
            
            for cb in schedule.callbacks: cb.on_epoch_begin(self)
            
            for input, label, *_ in tqdm(schedule.data, desc="Steps", leave=False):
                if not self.running: break
                for cb in schedule.callbacks: cb.on_batch_begin(self)
                step_loss, outputs = self.step(input, label)  
                if self.log: lossMeter.update(util.to_cpu(step_loss), input.shape[0])
                for cb in schedule.callbacks: cb.on_batch_end(self, lossMeter, outputs, label)
            
            for cb in schedule.callbacks: cb.on_epoch_end(self, lossMeter)      

            self.epoch += 1

            if checkpoint_file != None: 
                end = time.time()
                elapsed = end - start

                if elapsed > ckpt_interval:
                    start = end
                    self.checkpoint(checkpoint_file)
                    print("\n--- CHECKPOINT ---")

        for cb in schedule.callbacks: cb.on_train_end(self)   

    def train(self, schedule, checkpoint_file=None, reset=False, ckpt_interval=5*60):      
        with TrainModel(self.model):
            self.run(schedule, checkpoint_file, reset)

    def stop(self):
        self.running = False

    def to_fp16(self):
        if not apex_installed: raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use MixedPrecision training.")
        self.mixed_precision = True
        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')


class TrainingSchedule():
    def __init__(self, data, epochs, callbacks=[]):
        self.data = data
        self.callbacks = callbacks
        self.epochs = epochs

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def state_dict(self):
        callbacks_dict = [callback.state_dict() for callback in self.callbacks]
        return pickle.dumps({'callbacks': callbacks_dict, 'epochs': self.epochs})

    def load_state_dict(self, serialized_state_dict):
        state_dict = pickle.loads(serialized_state_dict)
        for cb, cb_state_dict in zip(self.callbacks, state_dict['callbacks']): cb.load_state_dict(cb_state_dict)
        self.epochs = state_dict['epochs']
