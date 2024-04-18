from utils.JPEG import DiffJPEG
from utils.jpegMASK import JpegMask
from utils.compressNet import CompressNet
import torch.nn as nn
import torch
import config as c
import numpy as np
from rgb2yuv import *


def load(network, pathname, netname):
    state_dicts = torch.load(pathname)
    network_state_dict = {k: v for k, v in state_dicts[netname].items() if 'tmp_var' not in k}
    network.load_state_dict(network_state_dict)


class JPEGSimulator(nn.Module):
    def __init__(self,device='cuda'):
        super(JPEGSimulator, self).__init__()
        # jpeg_isp80 = DiffJPEG(quality=80).to(device)
        # jpeg_isp85 = DiffJPEG(quality=85).to(device)
        self.jpeg_isp = DiffJPEG(quality=90).to(device)

        # jpeg_hidden80 = JpegMask(Q=80).to(device)
        # jpeg_hidden85 = JpegMask(Q=85).to(device)
        self.jpeg_hidden = JpegMask(Q=90).to(device)

        self.simnet = CompressNet().to(device)
        load(self.simnet, c.MODEL_PATH + c.simnet_pth, 'net')
        # self.gauss_std = c.gauss_std

        self.rgbtoyuv = rgb_to_ycbcr_jpeg()
        self.rgbtoyuv = self.rgbtoyuv.to(device)
        self.yuvtorgb = ycbcr_to_rgb_jpeg()
        self.yuvtorgb = self.yuvtorgb.to(device)
        # self.jpeg_isp = [jpeg_isp80, jpeg_isp85, jpeg_isp90]
        # self.jpeg_hidden = [jpeg_hidden80, jpeg_hidden85, jpeg_hidden90]


    def forward(self, x,device='cuda'):
        # r = torch.randint(0, 3, [1])

        y_isp = self.jpeg_isp(x)
        y_hidden = self.jpeg_hidden(x)
        y_simnet = self.yuvtorgb(self.simnet(self.rgbtoyuv(x)))

        # 随机生成权重
        weight = np.random.rand(3)
        sum = weight.sum()
        weight = weight / sum

        out = weight[0] * y_isp + weight[1] * y_hidden + weight[2] * y_simnet
        # out=y_hidden


        return out
