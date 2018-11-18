import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models
from Datasets.RoadDamage import *
from session import *
from validation import *
from LR_Schedule.cyclical import *

torch.backends.cudnn.benchmark=True

def main():
    data = RoadDamageClassifierData("C:/fastai/courses/dl2/data/road_damage_dataset/Data")

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

    sess = Session(model_ft, criterion, optim_fn, [*[1e-3] * 9, 1e-2])

    sess.freeze() 

    sess.set_lr(3e-4)

    accuracy = NHotAccuracy(8)
    validator = Validator(data['valid'], accuracy, save_best=True, model_dir="C:/fastai/courses/dl2/data/road_damage_dataset/Models")
    lr_scheduler = Cyclical(len(data['train']) * 24)
    schedule = TrainingSchedule(data['train'], [validator, lr_scheduler])

    sess.train(schedule, 24)

    sess.unfreeze()

    sess.set_lr([*[1e-4 / 1000] * 5, *[1e-4 / 100] * 4, 1e-4])

    sess.train(schedule, 24)

if __name__ == '__main__':
    main()