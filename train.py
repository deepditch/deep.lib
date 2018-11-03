import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import pandas as pd
from PIL import ImageDraw, ImageFont
from collections import namedtuple, OrderedDict
from session import *
from LR_Schedule.cos_anneal import CosAnneal
from LR_Schedule.cyclical import Cyclical
from LR_Schedule.lr_find import lr_find
from callbacks import *
from validation import *
from Vision.ImageHelpers import *
from Vision.SSD import *
from Datasets.RoadDamage import RoadDamageDataset

torch.backends.cudnn.benchmark=True

imsize = 448
batch_size = 32
data, classes, train_tfms, val_tfms, denorm = RoadDamageDataset('../storage/road_damage_data', imsize, batch_size)
num_classes = len(classes) - 1

class StdConv(nn.Module):
    def __init__(self, n_in, n_out, stride=2, drop_p=0.1):
        super().__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm = nn.BatchNorm2d(n_out)
        self.dropout = nn.Dropout(drop_p)
        
    def forward(self, x):
        return self.dropout(self.batch_norm(self.relu(self.conv(x))))
    
def flatten_conv(x,k=1):
    bs,nf,gx,gy = x.size()
    x = x.permute(0,2,3,1).contiguous()
    return x.view(bs,-1,nf//k)

class SSDOut(nn.Module):
    def __init__(self, n_in, k=1):
        super().__init__()
        self.k = k
        self.out_classes = nn.Conv2d(n_in, (num_classes + 1) * self.k, 3, padding=1) # Output for each class + background class
        self.out_boxes = nn.Conv2d(n_in, 4*self.k, 3, padding=1) # Output for bounding boxes
        
    def forward(self, x):
        return [flatten_conv(self.out_classes(x), self.k), F.tanh(flatten_conv(self.out_boxes(x), self.k))] 

class SSDHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.conv_0 = StdConv(512, 256, stride=1)
        self.conv_1 = StdConv(256, 256)
        self.out = SSDOut(256)
        
    def forward(self, x):
        x = self.dropout(F.relu(x))
        x = self.conv_0(x)
        x = self.conv_1(x)
        return self.out(x)
    
def var_from_np(arr, requires_grad=True):
    return Variable(torch.from_numpy(arr), requires_grad=requires_grad)

anc_grids = [4,2,1]

anc_zooms = [1., .7]

anc_ratios = [(1.,1.), (1.,.5), (.5,1.)]

anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]

# print(anchor_scales)

k = len(anchor_scales)

anc_offsets = [1/(o*2) for o in anc_grids]

anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag) for ao, ag in zip(anc_offsets, anc_grids)])

# print(anc_x)

anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag) for ao, ag in zip(anc_offsets, anc_grids)])

# print(anc_y)

anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)

# print(anc_ctrs)

striped_anc_x = np.concatenate([[(i+1)/4-.125 for i in range(4)], [.5, .5, .5, .5]])

striped_anc_y = np.concatenate([[.5, .5, .5, .5], [(i+1)/4-.125 for i in range(4)]])

striped_anc_ctrs = np.stack([striped_anc_x,striped_anc_y], axis=1)

striped_anc_sizes = np.concatenate([[[.25, 1]] * 4, [[1, .25]] * 4])

striped_anchors = np.concatenate([striped_anc_ctrs, striped_anc_sizes], axis=1)

small_striped_anc_x = np.tile(np.concatenate([[.25] * 3, [.5] * 3, [.75] * 3]), 2)
small_striped_anc_x = np.concatenate([small_striped_anc_x, [.125, .125 - .0625, .125 + .0625, .375, .375 - .0625, .375 + .0625, .625, .625 - .0625, .625 + .0625, .875, .875 - .0625, .875 + .0625]])

small_striped_anc_y = np.concatenate([np.tile([.625, .625 - .0625, .625 + .0625], 3), np.tile([.875, .875 - .0625, .875 + .0625], 3)])
small_striped_anc_y = np.concatenate([small_striped_anc_y, [.75] * 12])

small_striped_anc_ctrs = np.stack([small_striped_anc_x, small_striped_anc_y], axis=1)

striped_anc_sizes = np.concatenate([[[.5, .25], [.5, .125], [.5, .125]] * 6, [[.25, .5], [.125, .5], [.125, .5]]* 4])

small_striped_anchors = np.concatenate([small_striped_anc_ctrs, striped_anc_sizes], axis=1)

anc_sizes = np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o, p in anchor_scales])
               for ag in anc_grids])

np_grid_sizes = np.concatenate([np.array([1/ag for i in range(ag*ag) for o, p in anchor_scales])
               for ag in anc_grids])

np_grid_sizes = np.concatenate([[1] * 8, [.5] * 30, np_grid_sizes])

np_anchors = np.concatenate([striped_anchors, small_striped_anchors, np.concatenate([anc_ctrs, anc_sizes], axis=1)])

grid_sizes = var_from_np(np_grid_sizes, requires_grad=False).unsqueeze(1).float()

anchors = var_from_np(np_anchors, requires_grad=False).float()

class SSDOut4by1Stripes(nn.Module):
    def __init__(self, n_in, k=1):
        super().__init__()
        self.k = k
        self.out_classes_1 = nn.Conv2d(n_in, (num_classes + 1) * self.k, (4,1), padding=0) # Output for each class + background class
        self.out_boxes_1 = nn.Conv2d(n_in, 4*self.k, (4,1), padding=0) # Output for bounding boxes
        
        self.out_classes_2 = nn.Conv2d(n_in, (num_classes + 1) * self.k, (1,4), padding=0) # Output for each class + background class
        self.out_boxes_2 = nn.Conv2d(n_in, 4*self.k, (1,4), padding=0) # Output for bounding boxes
        
    def forward(self, x):    
        oc1 = flatten_conv(self.out_classes_1(x), self.k)
        ob1 = F.tanh(flatten_conv(self.out_boxes_1(x), self.k))
        
        oc2 = flatten_conv(self.out_classes_2(x), self.k)
        ob2 = F.tanh(flatten_conv(self.out_boxes_2(x), self.k))
        
        return [torch.cat([oc1, oc2], dim=1), torch.cat([ob1, ob2], dim=1)]
    
class SSDOut2by1Stripes(nn.Module):
    def __init__(self, n_in, k=1):
        super().__init__()
        self.k = k
        self.out_classes_1 = nn.Conv2d(n_in, (num_classes + 1)*self.k, (2,1), padding=0) # Output for each class + background class
        self.out_boxes_1 = nn.Conv2d(n_in, 4*self.k, (2,1), padding=0) # Output for bounding boxes
        
        self.out_classes_2 = nn.Conv2d(n_in, (num_classes + 1)*self.k, (1,2), padding=0) # Output for each class + background class
        self.out_boxes_2 = nn.Conv2d(n_in, 4*self.k, (1,2), padding=0) # Output for bounding boxes
        
    def forward(self, x):    
        oc1 = flatten_conv(self.out_classes_1(x), self.k)
        ob1 = F.tanh(flatten_conv(self.out_boxes_1(x), self.k))
        
        oc2 = flatten_conv(self.out_classes_2(x), self.k)
        ob2 = F.tanh(flatten_conv(self.out_boxes_2(x), self.k))
        
        return [torch.cat([oc1, oc2], dim=1), torch.cat([ob1, ob2], dim=1)]

class SSD_MultiHead(nn.Module):
    def __init__(self, k, bias):
        super().__init__()
        self.drop = nn.Dropout(.4)
        self.sconv0 = StdConv(512,256, stride=2, drop_p=.4)
        self.sconv1 = StdConv(256,256, drop_p=.4)
        self.sconv2 = StdConv(256,256, drop_p=.4)
        self.sconv3 = StdConv(256,256, drop_p=.4)
        self.out1 = SSDOut(256, k)
        self.outStripesLarge = SSDOut4by1Stripes(256, 1)
        self.outStripesSmall = SSDOut2by1Stripes(256, 3)
        self.out2 = SSDOut(256, k)
        self.out3 = SSDOut(256, k)

    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconv0(x)
        x = self.sconv1(x)    
        o1c,o1l = self.out1(x)
        o11c,o11l = self.outStripesLarge(x) 
        o12c,o12l = self.outStripesSmall(x[:,:,:,2:])   
        x = self.sconv2(x)
        o2c,o2l = self.out2(x)
        x = self.sconv3(x)
        o3c,o3l = self.out3(x)
        return [torch.cat([o11c,o12c,o1c,o2c,o3c], dim=1),
                torch.cat([o11l,o12l,o1l,o2l,o3l], dim=1)]
    
model_ft = models.resnet34(pretrained=True)
layers = list(model_ft.children())[0:-2]
layers += [SSD_MultiHead(k, -4.)]
model = nn.Sequential(*list(layers))
criterion = SSDLoss(anchors, grid_sizes, num_classes, imsize)
optim_fn = optim.Adam
sess = Session(model, criterion, optim_fn, [*[1e-3] * 8, 1e-2])

sess.set_lr([*[7e-3 / 2] * 8, 7e-3])

lr_scheduler = Cyclical(len(data['train']) * 32)
accuracy = JaccardAccuracy(anchors, grid_sizes, imsize)
validator = Validator(data['valid'], accuracy)
schedule = TrainingSchedule(data['train'], [lr_scheduler, validator])

sess.train(schedule, 32)

print("Saving Model")

sess.save("Resnet34BottomAnchors")
sess.load("Resnet34BottomAnchors")

print("Done")