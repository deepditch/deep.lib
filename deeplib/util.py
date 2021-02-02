import torch
from collections import Iterable

USE_GPU = torch.cuda.is_available()

def to_gpu(x, *args, **kwargs):
    '''puts pytorch variable to gpu, if cuda is available and USE_GPU is set to true. '''
    return x.cuda(*args, **kwargs) if USE_GPU else x

def to_cpu(x, *args, **kwargs):
    return x.cpu(*args, **kwargs) if USE_GPU else x

def listify(x, y):
    if not isinstance(x, Iterable): x=[x]
    n = y if type(y)==int else len(y)
    if len(x)==1: x = x * n
    return x

def partition(x, sz):
    return [x[i:i+4] for i in range(0, len(x), 4)]

def mask(arr, indices):
    return [arr[i] for i in indices]

def bn2float(module:torch.nn.Module)->torch.nn.Module:
    "If `module` is batchnorm don't use half precision."
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): module.float()
    for child in module.children(): bn2float(child)
    return module

def model2half(model:torch.nn.Module)->torch.nn.Module:
    "Convert `model` to half precision except the batchnorm layers."
    return bn2float(model.half())

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
