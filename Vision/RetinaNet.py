import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
import torchvision.models as models


def init_conv_weights(layer, weights_std=0.01, bias=0):
    '''
    RetinaNet's layer initialization
    :layer
    :
    '''
    nn.init.normal(layer.weight.data, std=weights_std)
    nn.init.constant(layer.bias.data, val=bias)
    return layer


def conv1x1(in_channels, out_channels, **kwargs):
    '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)
    return layer


def conv3x3(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = conv1x1(in_channels, out_channels, **kwargs)
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        # x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels, **kwargs)   
        # self.batch_norm = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        # x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()
        

class FeaturePyramid(nn.Module):
    def __init__(self, resnet, C3_size=512, C4_size=1024, C5_size=2048, feature_size=256):
        super().__init__()

        self.resnet = resnet
        
        self.sfs = [SaveFeatures(self.resnet[-3]), SaveFeatures(self.resnet[-2]) ]

        # upsample C5 to get P5 from the FPN paper
        self.P5_1           = Conv1x1(C5_size, feature_size, stride=1, padding=0)
        self.P5_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2           = Conv3x3(feature_size, feature_size, stride=1, padding=1)
        
        # add P5 elementwise to C4
        self.P4_1           = Conv1x1(C4_size, feature_size, stride=1, padding=0)
        #self.P4_upsampled   = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2           = Conv3x3(feature_size, feature_size, stride=1, padding=1)
        
        # add P4 elementwise to C3
        #self.P3_1 = conv1x1(C3_size, feature_size, stride=1, padding=0)
        #self.P3_2 = conv3x3(feature_size, feature_size, stride=1, padding=1)
        
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = Conv3x3(C5_size, feature_size, stride=2, padding=1)
        
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = Conv3x3(feature_size, feature_size, stride=2, padding=1)


    def forward(self, x):
        C5 = self.resnet(x)
        C4 = self.sfs[1].features
        C3 = self.sfs[0].features
        
        # print(C3.shape, C4.shape, C5.shape)

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)
        
        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x
        #P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        #P3_x = self.P3_1(C3)
        #P3_x = P3_x + P4_upsampled_x
        #P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        
        # print(P3_x.shape, P4_x.shape, P5_x.shape, P6_x.shape, P7_x.shape)
        
        return [P4_x, P5_x, P6_x, P7_x]


class _SubNet(nn.Module):
    def __init__(self, classes, anchors, depth=4, feature_size=256):
        super().__init__()
        self.anchors = anchors
        self.classes = classes
        self.depth = depth
        self.feature_size = feature_size

        self.relu = nn.ReLU(inplace=True)
        # self.down_sample_1 = conv3x3(self.feature_size, self.feature_size, padding=1, stride=2)
        # self.down_sample_2 = conv3x3(self.feature_size, self.feature_size, padding=1, stride=2)
        self.layers = nn.ModuleList([Conv3x3(self.feature_size, self.feature_size, padding=1)] * 4)

    def output_layer(self, x): raise NotImplementedError

    def flatten_conv(self, x):
        bs,nf,gx,gy = x.size()
        x = x.permute(0,2,3,1).contiguous()
        return x.view(bs,-1,nf//self.anchors)

    def forward(self, x):
        # x = self.relu(self.down_sample_1(x))
        # x = self.relu(self.down_sample_2(x))
        for layer in self.layers:
            x = self.relu(layer(x))

        x = self.output_layer(x)

        return self.flatten_conv(x)


class RegressionSubnet(_SubNet):
    def __init__(self, classes, anchors, depth=4, feature_size=256):
        super().__init__(classes, anchors, depth, feature_size)
        self.conv = conv3x3(self.feature_size, 4 * self.anchors, padding=1)
        
    def output_layer(self, x):
        return F.tanh(self.conv(x))


class ClassificationSubnet(_SubNet):
    def __init__(self, classes, anchors, depth=4, feature_size=256):
        super().__init__(classes, anchors, depth, feature_size)
        self.conv = conv3x3(self.feature_size, (1 + self.classes) * self.anchors, padding=1)
        prior = 0.01    
        self.conv.bias.data.fill_(-math.log((1.0-prior)/prior))
        
    def output_layer(self, x):
        return self.conv(x)
    

class RetinaNet(nn.Module):
    def __init__(self, classes, anchors, model_ft=models.resnet50(pretrained=True), C3_size=512, C4_size=1024, C5_size=2048):
        super().__init__()     
        
        layers = list(model_ft.children())[0:-2]
        _resnet = nn.Sequential(*list(layers))
        
        self.feature_pyramid = FeaturePyramid(_resnet, C3_size, C4_size, C5_size)

        self.subnet_boxes = RegressionSubnet(classes, anchors)
        self.subnet_classes = ClassificationSubnet(classes, anchors)

    def forward(self, x):
        boxes = []
        classes = []

        features = self.feature_pyramid(x)

        # how faster to do one loop
        boxes = [self.subnet_boxes(feature) for feature in features]
        classes = [self.subnet_classes(feature) for feature in features]

        return [torch.cat(classes, 1), torch.cat(boxes, 1)]

