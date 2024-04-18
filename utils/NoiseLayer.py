import torch.nn as nn
import torch
import config as c
import numpy as np
from JPEGsimulator import JPEGSimulator
from noise import *

'''
part1: JPEGsim
part2: Gauss
'''


class NoiseLayer(nn.Module):
    def __init__(self, device='cuda'):
        super(NoiseLayer, self).__init__()
        self.jpeg = JPEGSimulator().to(device)
        self.jpeg.requires_grad_(False)

        self.GaussianNoise = GaussianNoise().to(device)
        self.GaussianBlur = GaussianBlur2D().to(device)
        self.GaussianNoise.requires_grad_(False)
        self.GaussianNoise.requires_grad_(False)

        self.noise = [self.GaussianBlur, self.GaussianNoise, self.jpeg]


    def forward(self, x):
        if c.noise:
            r = torch.randint(0, 3, [1])
            out = self.noise[r[0]](x)
        else:
            out = x

        return out
