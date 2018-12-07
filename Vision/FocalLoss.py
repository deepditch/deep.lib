import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import util

def one_hot_embedding(labels, num_classes):
    ret = torch.eye(num_classes)[labels.data.cpu().long()]
    return ret

class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, label):
        target = one_hot_embedding(label, self.num_classes + 1) # +1 for background
        target = util.to_gpu(Variable(target[:,1:].contiguous())) # Ignore background and send to GPU
        pred = outputs[:,1:] # Get the models predictions (no background)
        weight = self.get_weight(pred, target)
        return F.binary_cross_entropy_with_logits(pred, target, weight, size_average=False)
    
    def get_weight(self, x, t):
        alpha, gamma = 0.25, 2
        p = x.sigmoid()
        pt = p * t + (1-p) * (1-t)
        w = alpha * t + (1-alpha) * (1-t)
        return w * (1-pt).pow(gamma)