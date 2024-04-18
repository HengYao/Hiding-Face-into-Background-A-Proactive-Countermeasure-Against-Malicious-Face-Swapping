# import skvideo.io
import torch
import matplotlib.pyplot as plt
import glob
import os
import PIL.Image as img
import cv2
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from JPEG_utils import diff_round, quality_to_factor, Quantization
from compression import compress_jpeg
from decompression import decompress_jpeg
import config as c


# 1.高斯噪声 (0,0.02)
class GaussianNoise(nn.Module):
    def __init__(self,loc=0,scale=0.03):
        super(GaussianNoise, self).__init__()
        self.loc=loc
        self.scale=scale
    def forward(self,input):
        shape = input.shape
        noise = np.random.normal(loc=self.loc, scale=self.scale, size=shape)
        noise = torch.from_numpy(noise).cuda()
        output = input + noise
        return output.float()


# 2.2D高斯滤波 (0,2),3x3
class GaussianBlur2D(nn.Module):
    def __init__(self, kernel_size=3, sigma=0.4, channels=3, height=c.resize_h, width=c.resize_w):  # 3*c.clip_len
        super(GaussianBlur2D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        self.height = height
        self.width = width
        self.weight = self.gaussian_kernel()
        self.mask = self.weight_mask()

    def gaussian_kernel(self):
        kernel = np.zeros(shape=(self.kernel_size, self.kernel_size))
        radius = self.kernel_size // 2
        for y in range(-radius, radius + 1):
            for x in range(-radius, radius + 1):
                v = 1.0 / (2 * np.pi * self.sigma ** 2) * np.exp(-1.0 / (2 * self.sigma ** 2) * (x ** 2 + y ** 2))
                kernel[y + radius, x + radius] = v
        kernel2d = kernel
        kernel = kernel / np.sum(kernel)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        weight = nn.Parameter(data=kernel, requires_grad=False)

        return weight

    def weight_mask(self):
        ones = torch.ones([1, self.channels, self.height, self.width])
        mask = F.conv2d(ones, self.weight, bias=None, stride=1, padding=self.kernel_size // 2, dilation=1,
                        groups=self.channels)
        return mask

    def forward(self, input):
        ### 转b,cn,h,w
        # input = torch.concat(input.unbind(2), 1)
        output = F.conv2d(input, self.weight, bias=None, stride=1, padding=self.kernel_size // 2, dilation=1,
                          groups=self.channels)
        output = output / self.mask.cuda()
        # b, cn, h, w = output.shape

        return output.float()
