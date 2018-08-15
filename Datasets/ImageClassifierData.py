import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision
from torchvision import datasets
import pandas as pd
import cv2
import numpy as np
import itertools
import os

def open_image(fn):
    """ Opens an image using OpenCV given the file path.

    Arguments:
        fn: the file path of the image

    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn) and not str(fn).startswith("http"):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn) and not str(fn).startswith("http"):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            if str(fn).startswith("http"):
                req = urllib.urlopen(str(fn))
                image = np.asarray(bytearray(req.read()), dtype="uint8")
                im = cv2.imdecode(image, flags).astype(np.float32)/255
            else:
                im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e


class ClassifierData():
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __getitem__(self, key):
        return self.dataloaders[key]


class ImageDataset(Dataset):
    def __init__(self, files, labels, transforms):
        self.files = files
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        file, label = self.files[i], self.labels[i]
        return (self.transforms(open_image(file)), label)


def from_paths(path, batch_size, transforms):
    path = Path(path)
    image_datasets = {dir: datasets.ImageFolder(path/dir, transform) for dir, transform in transforms.items()}
    dataloaders = {dir: torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4) for dir, data in image_datasets.items()}
    return ClassifierData(dataloaders)


def make_n_hot_labels(labels):
    classes = sorted(list(set(itertools.chain.from_iterable(labels))))
    label2idx = {v:k for k,v in enumerate(classes)}
    n_hot_labels = [[0] * len(classes) for l in labels]    
    for i, l in enumerate(labels):
        for classname in l:
            n_hot_labels[i][label2idx[classname]] = 1

    return n_hot_labels, classes


def parse_csv_data(csv_file):
    path = Path(csv_file)
    df = pd.read_csv(path, dtype=str)
    files = df[df.columns[0]]
    labels = df[df.columns[1]]
    return files, labels


def from_csv(dir, csv_file, batch_size, transforms):
    files, labels = parse_csv_data(csv_file)
    labels = [l.split(' ') for l in labels]
    files = [os.path.join(dir, file) for file in files]
    n_hot_lables, classes = make_n_hot_labels(labels)
    dataset = ImageDataset(files, n_hot_lables, transforms)
    dataloaders = {'train': torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)}
    return ClassifierData(dataloaders)