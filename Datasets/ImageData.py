import os
import numpy as np
from pathlib import Path
import cv2
import pandas as pd
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets

import Datasets.ModelData as md
import Datasets.ClassifierData as ClassifierData


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


class ImageDataset(Dataset):
    def __init__(self, files, labels, transform, balanced=False):
        self.files = files
        self.labels = labels
        self.transform = transform

        if(balanced):              
            counts = np.zeros(len(labels[0]))
            N = 0
            for label in labels:
                counts += label
                N += np.sum(label)

            class_weights = N / counts

            image_weights = np.zeros(len(labels))

            for idx, label in enumerate(labels):
                image_weights[idx] = np.sum(np.compress(label, class_weights))

            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(image_weights, len(image_weights))
        else:
            self.sampler = None
    
    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        file, label = self.files[i], self.labels[i]
        label = deepcopy(label) # Transformations happen in place. Need to deepcopy original label
        x, y = self.transform(open_image(file), label)
        
        if isinstance(y, md.StructuredLabel): ## Ignore the label types
            label = {part[2]: part[0] for part in y}
        else:
            label = y

        meta = {'file': str(file)}
        return x, label, meta


def from_paths(path, batch_size, transforms):
    path = Path(path)
    image_datasets = {dir: datasets.ImageFolder(path/dir, transform) for dir, transform in transforms.items()}
    return md.ModelData(image_datasets, batch_size)


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


def from_csv(dir, csv_file, batch_size, train_transforms, val_trainsforms):
    """Create image ModelData from a csv file. CSV file should be formatted as (filename, label) where filename is unique
    
    Arguments:
        dir {str} -- Folder containing image files
        csv_file {str} -- Location of csv file
        batch_size {int} -- Number of items returned by a dataloader each iteration
        train_transforms {Transform} -- Transforms to be applied to train data
        val_trainsforms {Transform} -- Transforms to be applied to validation data
    
    Returns:
        ModelData
    """

    dir = Path(dir)
    files, labels = parse_csv_data(csv_file)
    labels = [l.split(' ') for l in labels]
    files = [dir/file for file in files]
    n_hot_labels, classes = ClassifierData.make_n_hot_labels(labels)
    i_dict = md.make_partition_indices(len(n_hot_labels), {'train': .8, 'valid': .2})
    datasets = {
        'train': ImageDataset(np.array(files)[i_dict['train']], np.array(n_hot_labels)[i_dict['train']], train_transforms, balanced=True),
        'valid': ImageDataset(np.array(files)[i_dict['valid']], np.array(n_hot_labels)[i_dict['valid']], val_trainsforms)
    }  
    return md.ModelData(datasets, batch_size)
