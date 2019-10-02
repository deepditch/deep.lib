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