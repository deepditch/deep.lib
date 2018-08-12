import torch
from pathlib import Path
import torchvision
from torchvision import datasets

class ClassifierData():
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __getitem__(self, key):
        return self.dataloaders[key]


class ImageClassifierData(ClassifierData):
    def __init__(dataloaders):
        super(ImageClassifierData, self).__init__(dataloaders)

    def denorm(x):
        return x


def from_paths(path, batch_size, transforms):
    path = Path(path)
    image_datasets = {dir: datasets.ImageFolder(path/dir, transform) for dir, transform in transforms.items()}
    dataloaders = {dir: torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4) for dir, data in image_datasets.items()}
    return ClassifierData(dataloaders)
    
