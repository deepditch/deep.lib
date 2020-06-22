import torch
from torch import nn, optim
from torch.nn.modules.loss import *
import torch.nn.functional as F

from deeplib.Loss.triplet import *


class TripletRegularizedLoss(nn.Module):
    def __init__(self, alpha, margin, loss_fn, triplet_loss_fn=batch_all_triplet_loss):     
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.loss_fn = loss_fn
        self.triplet_loss_fn = triplet_loss_fn
        
    def forward(self, x, y):
        loss = self.loss_fn(x[-1], y)
        triplet = 0
        
        if (self.alpha > 0):
            for layer in x[:-1]:    
                triplet += self.triplet_loss_fn(layer.view(layer.size(0), -1), y, self.margin)

            triplet *= self.alpha
            
        return loss + triplet


class TripletRegularizedCrossEntropyLoss(TripletRegularizedLoss):
    def __init__(self, alpha, margin, select=None, triplet_loss_fn=batch_all_triplet_loss):     
        super(TripletRegularizedCrossEntropyLoss, self).__init__(alpha, margin, F.cross_entropy, triplet_loss_fn)


class TripletRegularizedMultiMarginLoss(TripletRegularizedLoss):
    def __init__(self, alpha, margin, select=None, triplet_loss_fn=batch_all_triplet_loss):     
        super(TripletRegularizedMultiMarginLoss, self).__init__(alpha, margin, F.multi_margin_loss, triplet_loss_fn)


class MarginRegularizedCrossEntropyLoss(DistanceWeightedMarginLoss):
  def __init__(self, lmbda, num_classes, margin=.2, beta=1.2, nonzero_loss_cutoff=1.4, cutoff=0.5):
    super(MarginRegularizedCrossEntropyLoss, self).__init__(num_classes, margin, beta, nonzero_loss_cutoff, cutoff)
    self.lmbda = lmbda

  def forward(self, x, y):
    embed = x[0]
    out = x[1]

    margin_loss = super().forward(embed, y)
    out_loss = F.cross_entropy(out, y)
    
    return out_loss + self.lmbda * margin_loss


class MarginRegularizedMultiMarginLoss(DistanceWeightedMarginLoss):
  def __init__(self, lmbda, num_classes, margin=.2, beta=1.2, nonzero_loss_cutoff=1.4, cutoff=0.5):
    super(MarginRegularizedMultiMarginLoss, self).__init__(num_classes, margin, beta, nonzero_loss_cutoff, cutoff)
    self.lmbda = lmbda

  def forward(self, x, y):
    embed = x[0]
    out = x[1]

    margin_loss = super().forward(embed, y)
    out_loss = F.multi_margin_loss(out, y)
    
    return out_loss + self.lmbda * margin_loss
