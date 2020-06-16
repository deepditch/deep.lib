import numpy as np
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import resnet
from Models.selective_sequential import *
from Loss.triplet_regularized import *
from session import *
from LR_Schedule.lr_decay import LearningRateDecay
from LR_Schedule.cos_anneal import CosAnneal
from LR_Schedule.lr_find import lr_find
from callbacks import *
from validation import *
import Datasets.ImageData as ImageData
from Transforms.ImageTransforms import *
import util
from session import LossMeter, EvalModel
from Layers.flatten import Flatten
from pathlib import Path
from callbacks import TrainCallback
from Models.ShakeDrop.shake_pyramidnet import ShakePyramidNet
from autoaugment import CIFAR10Policy
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default='/media/drake/MX500/Datasets/cifar-10/train')
args = parser.parse_args()


tfms = transforms.Compose(
  [transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), 
    CIFAR10Policy(), 
    transforms.ToTensor(), 
    Cutout(n_holes=1, length=16),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

fulltrainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True,
                                        download=True, 
                                        transform=tfms)

testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False,
                                       download=True, 
                                       transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                       ]))

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

def make_train_loader(percentage=1, batch_size=256, dataset=fulltrainset):
    """
    Percentage: what percentage of the dataset should we use
    """
    total_num_examples = len(dataset)
    partial_num_examples = math.floor(total_num_examples * percentage)

    partial_dataset = torch.utils.data.dataset.Subset(dataset, np.arange(partial_num_examples))
    loader = torch.utils.data.DataLoader(partial_dataset, batch_size=batch_size, shuffle=True)

    return loader, partial_dataset

trainloader, trainset = make_train_loader(percentage=1, batch_size=128)


select = ['encoder', 'out']

class Identity(nn.Module):
  def forward(self, x): return x

def make_model():
    encoder = ShakePyramidNet(depth=110)
    encoder.fc_out = Identity()

    model = SelectiveSequential(
        select,
        {
        'encoder': encoder,
        'out': nn.Linear(286, 10),
        })

    return model


accuracies = {}

for param in [0, .05, .1, .15, .2, .25]:
    criterion = TripletRegularizedMultiMarginLoss(param, .5, select)
    sess = Session(make_model(), criterion, optim.AdamW, 1e-4)
    num_epochs = 128
    validator = TripletRegularizedLossValidator(testloader, select, CustomOneHotAccuracy, model_file=f"best-{param}", tensorboard_dir=f"./runs/lambda{param}")
    lr_scheduler = CosAnneal(len(dataloader), T_mult=2, lr_min=1e-7)
    schedule = TrainingSchedule(trainloader, num_epochs, [lr_scheduler, validator])
    sess.train(schedule, checkpoint_file=f"ckpt-{param}", ckpt_interval=60*60)
    
    accuracies[param] = validator.best_accuracy
    df = pd.DataFrame(accuracies)
    df.to_csv("./search.csv")