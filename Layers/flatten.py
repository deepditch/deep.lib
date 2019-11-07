
import torch
from torch import nn, optim

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)