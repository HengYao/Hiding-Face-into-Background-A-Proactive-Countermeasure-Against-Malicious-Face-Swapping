import torch
import torch.nn as nn
import torch.nn.functional as F
import config as c

class ConvBlock(nn.Module):
    def __init__(self,channel_in,channel_out,kernel_size, padding,stride=1):
        super(ConvBlock, self).__init__()
        self.conv=nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN=nn.BatchNorm2d(channel_out)
        self.ReLU= nn.LeakyReLU(0.1)

    def forward(self,x):
        return self.ReLU(self.BN(self.conv(x)))


class CompressNet(nn.Module):
    def __init__(self):
        super(CompressNet, self).__init__()
        self.conv0 = ConvBlock(3, 64, 1, 0)

        self.conv1 = ConvBlock(64, 64, 3, 1)
        self.conv2 = ConvBlock(64, 64, 3, 1)
        self.conv3 = ConvBlock(64, 64, 3, 1)

        self.conv4 = ConvBlock(64, 128, 3, 1)
        self.conv5 = ConvBlock(128, 128, 3, 1)
        self.conv6 = ConvBlock(128, 128, 3, 1)

        self.conv7 = ConvBlock(128, 64, 3, 1)
        self.conv8 = ConvBlock(64, 64,3, 1)
        self.conv9 = ConvBlock(64, 64, 3, 1)

        self.conv10 = ConvBlock(64, 3, 1, 0)

    def forward(self, input):

        # input=torch.cat((x,gauss),1)

        conv0 = self.conv0(input)

        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3_down = F.max_pool2d(conv3, 2, 2)

        conv4 = self.conv4(conv3_down)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)

        conv6_up = F.interpolate(conv6, scale_factor=2)

        conv7 = self.conv7(conv6_up)
        conv7_res = conv3 + conv7
        conv8 = self.conv8(conv7_res)
        conv9 = self.conv9(conv8)
        conv10 = self.conv10(conv9)
        y = input + conv10

        return y

# mode=CompressNet()
# x=torch.randn(4,3,288,288)
# y=mode(x,torch.normal(0,1,size=[4,3,288,288]))


def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()


