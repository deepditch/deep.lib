import os
import numpy as np
import itertools
from pathlib import Path
import cv2
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
from torch import randperm


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
    def __init__(self, datasets, batch_size, shuffle=True, num_workers=4):
        self.datasets = datasets
        self.dataloaders = {
            key: torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) 
            for key, data in datasets.items() }

    def __getitem__(self, key):
        return self.dataloaders[key]


class PartitionedData(ClassifierData):
    def __init__(self, dataset, batch_size, partition_dict={'train': .8, 'valid': .2}, shuffle=True, num_workers=4):
        """A class representing a set of torch dataloaders
        
        Arguments:
            dataset {Dataset} -- Dictionary containing 1 or more torch Datasets
            batch_size {int} -- Number of items returned by a dataloader each iteration
        
        Keyword Arguments:
            partition_dict {dict} - Dictionary describing how to partition the data
            shuffle {bool} -- Set to true to reshuffle the data (default: {True})
            num_workers {int} -- How many subprocesses to load data (default: {4})
        """
        self.master_dataset = dataset
        super().__init__(partition_data(dataset, partition_dict), batch_size, shuffle, num_workers)     


def partition_data(dataset, partition_dict):
    if sum(partition_dict.values()) != 1:
        raise ValueError("Percentages must add up to 1")

    indicies = np.random.permutation(len(dataset))
    datasets = {}
    start = 0
    end = 0
    for key, percentage in partition_dict.items():
        end = int(len(dataset) * percentage) + start
        datasets[key] = Subset(dataset, indicies[start:end])
        start = end

    return datasets


class Subset(Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ImageDataset(Dataset):
    def __init__(self, files, labels, transform):
        self.files = files
        self.labels = labels
        self.transform = transform
    
    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        file, label = self.files[i], self.labels[i]
        x, y = self.transform(open_image(file), label)
        return x, y


def from_paths(path, batch_size, transforms):
    path = Path(path)
    image_datasets = {dir: datasets.ImageFolder(path/dir, transform) for dir, transform in transforms.items()}
    return ClassifierData(image_datasets, batch_size)


def parse_csv_data(csv_file):
    """ Parse a CSV file into inputs and labels

    Arguments:
        csv_file {str} -- Path where csv file is located
    
    Returns:
        (str, str) -- first and second columns in the csv file
    """

    path = Path(csv_file)
    df = pd.read_csv(path, dtype=str)
    xs = df[df.columns[0]]
    ys = df[df.columns[1]]
    return xs, ys


def from_csv(dir, csv_file, batch_size, transform):
    """Create image ClassifierData from a csv file. CSV file should be formatted as (filename, label) where filename is unique
    
    Arguments:
        dir {str} -- Folder containing image files
        csv_file {str} -- Location of csv file
        batch_size {int} -- Number of items returned by a dataloader each iteration
        transform {Transform} -- Transforms to be applied to data
    
    Returns:
        ClassifierData
    """

    dir = Path(dir)
    files, labels = parse_csv_data(csv_file)
    labels = [l.split(' ') for l in labels]
    files = [dir/file for file in files]
    n_hot_lables, classes = make_n_hot_labels(labels)
    dataset = ImageDataset(files, n_hot_lables, transform)
    return PartitionedData(dataset, batch_size)


def make_n_hot_labels(labels):
    classes = sorted(list(set(itertools.chain.from_iterable(labels))))
    label2idx = {v:k for k,v in enumerate(classes)}
    n_hot_labels = [np.zeros((len(classes),), dtype=np.int_) for l in labels]     
    for i, l in enumerate(labels):
        for classname in l:
            n_hot_labels[i][label2idx[classname]] = 1

    return n_hot_labels, classes