import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from Datasets.RoadDamage import *
from session import *
from validation import *
from LR_Schedule.cos_anneal import *
import argparse

torch.backends.cudnn.benchmark=True

def main(args):
    data = RoadDamageClassifierData(args.data_dir, batch_size=args.batch_size, partitions=None)

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

    sess.load(args.model_file) 

    accuracy = NHotAccuracy(8)
    validator = Validator(data['valid'], accuracy)

    validator.run(sess)

    for i, c in enumerate(accuracy.confusion):
        print(f'class {i}: {c}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_file', metavar='model_file.ckpt.tar')
    parser.add_argument('--data_dir', metavar='Data Directory')
    parser.add_argument('--batch_size', type=int, metavar='Batch Size')

    args = parser.parse_args()

    args.batch_size = int(args.batch_size)

    main(args)