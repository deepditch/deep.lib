import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from enum import Enum

class ModelData():
    def __init__(self, datasets, batch_size, shuffle=True, num_workers=4):
        self.datasets = datasets
        self.dataloaders = {
            key: torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=num_workers, sampler=data.sampler) 
            for key, data in datasets.items() }

    def __getitem__(self, key):
        return self.dataloaders[key]

    def __setitem__(self, key, value):
        self.dataloaders[key] = value


class Subset(Dataset):
    """ Subset of a dataset at specified indices.

    Arguments:
        dataset {Dataset} -- The whole Dataset
        indices {sequence} -- Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class PartitionedData(ModelData):
    def __init__(self, dataset, batch_size, partition_dict={'train': .8, 'valid': .2}, shuffle=True, num_workers=4):
        """A class for partitioning a dataset
        
        Arguments:
            dataset {Dataset} -- A torch dataset
            batch_size {int} -- Number of items returned by a dataloader each iteration
        
        Keyword Arguments:
            partition_dict {dict} -- Dictionary describing how to partition the data
            shuffle {bool} -- Set to true to reshuffle the data (default: {True})
            num_workers {int} -- How many subprocesses to load data (default: {4})
        """
        self.master_dataset = dataset
        i_dict = make_partition_indices(partition_dict)
        datasets = {key: Subset(dataset, indicies) for key, indicies in i_dict.items()}
        super().__init__(datasets, batch_size, shuffle, num_workers)     


def make_partition_indices(n, partition_dict):
    if sum(partition_dict.values()) != 1:
        raise ValueError("Percentages must add up to 1")

    i_arr = np.random.permutation(n) # Index array: Array containing a random permutation of integers from 0 through (n-1)
    indices = {}
    start, end = 0, 0
    for key, percentage in partition_dict.items():
        end = int(n * percentage) + start
        indices[key] = i_arr[start:end]
        start = end

    return indices


class LabelType(Enum):
    CATEGORY = 1
    COORDINATE = 2 # List of concatinated coordinates: [x,y] * n
    BOUNDING_BOX = 3 # List of concatinated bounding boxes: [center_x, center_y, width, height] * n
    NA = 4


class StructuredLabel(list):
    '''Wrapper class representing a structured label where portions of the label are classified with a LabelType
    '''

    def __init__(self, label=[]):
        '''initialize a StructuredLabel object
        
        Arguments:
            label {list of tuple} -- a list of tuples where each tuple is of the format (label, LabelType)
        '''
        super().__init__(label)

        # TODO: Add label format verification