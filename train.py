import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from Datasets.RoadDamage import *
from session import *
from validation import *
from LR_Schedule.cos_anneal import *
import argparse
from callbacks import MemoryProfiler
torch.backends.cudnn.benchmark=True

def main(args):
    data = RoadDamageClassifierData(args.data_dir, batch_size=args.batch_size)

    # Define our model using a pretrainind ResNet backbone
    model_ft = models.resnet18(pretrained=True)
    model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Dropout(p=.2),
        nn.Linear(num_ftrs, 8),
        nn.Sigmoid()
    )

    criterion = nn.BCELoss()
    optim_fn = optim.Adam
    sess = Session(model_ft, criterion, optim_fn, 3e-4)

    # Let's freeze all but the final layer i.e. only the last layer's weights are updated during training and the pretrained weights from the ResNet are unchanged
    sess.freeze() 

    # Configure a training schedule. This schedule will decay the learning weight and validate after each epoch. The best model is saved in the model_dir 
    accuracy = NHotAccuracy(8)
    validator = Validator(data['valid'], accuracy, save_best=True, model_dir=args.model_dir)
    lr_scheduler = CosAnneal(len(data['train']), T_mult=2, lr_min=3e-4/100)
    schedule = TrainingSchedule(data['train'], [validator, lr_scheduler])

    # Train for 15 epochs using the schedule
    sess.train(schedule, 15)

    # Now let's unfreeze the ResNet backbone and fine tune it
    sess.unfreeze()

    # Reset our learning rate decay and redefine our schedule
    lrs = [*[1e-4 / 100] * 5, *[1e-4 / 10] * 4, 1e-4]
    lr_scheduler = CosAnneal(len(data['train']), T_mult=2, lr_min=[lr / 100 for lr in lrs])
    schedule = TrainingSchedule(data['train'], [validator, lr_scheduler])

    # This line sets a smaller learning rate for earlier layers in the network
    # The network has 10 layers, 9 belong to the ResNet backbone. The last layer gets 1e-4 as a learning rate
    # The idea here is that we update the highest level feature embeddings faster than the lowest level ones
    # The lowest level features from the pre-trained ResNet are likeley already 'good' for our task  
    sess.set_lr(lrs)

    # And then train for 63 epochs
    sess.train(schedule, 63)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', metavar='Data Directory')
    parser.add_argument('--model_dir', metavar='Model Directory', help='The directory for saving models while training')
    parser.add_argument('--batch_size', type=int, metavar='Batch Size')

    args = parser.parse_args()

    args.batch_size = int(args.batch_size)

    main(args)