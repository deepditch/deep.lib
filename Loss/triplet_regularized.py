import torch
from torch import nn, optim
from torch.nn.modules.loss import *
import torch.nn.functional as F
from Loss.triplet import *

class TripletRegularizedLoss(nn.Module):
    def __init__(self, alpha, margin, loss_fn):     
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.loss_fn = loss_fn
        
    def forward(self, x, y):
        loss = self.loss_fn(x[-1][0], y)
        triplet = 0
        
        if (self.alpha > 0):
            for layer in x[:-1]:        
                triplet += batch_hard_triplet_loss(layer[0].view(layer[0].size(0), -1), y, self.margin)

            triplet *= self.alpha
            
        return loss + triplet

class TripletRegularizedCrossEntropyLoss(TripletRegularizedLoss):
    def __init__(self, alpha, margin):     
        super(TripletRegularizedCrossEntropyLoss, self).__init__(alpha, margin, F.cross_entropy)

class TripletRegularizedMultiMarginLoss(TripletRegularizedLoss):
    def __init__(self, alpha, margin):     
        super(TripletRegularizedMultiMarginLoss, self).__init__(alpha, margin, F.multi_margin_loss)