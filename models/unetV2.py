import torch
import torch.nn as nn
import torch.nn.functional as F
import config as c


class ConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN = nn.BatchNorm2d(channel_out)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.BN(self.conv(x)))

class UpConvBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(UpConvBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, 3, 2, 1, 1)
        self.BN = nn.BatchNorm2d(channel_out)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        return self.ReLU(self.BN(self.conv(x)))

class ConvBlock3(nn.Module):
    def __init__(self, channel_in, channel_mid, channel_out):
        super(ConvBlock3, self).__init__()
        self.conv1 = ConvBlock(channel_in, channel_mid)
        self.conv2 = ConvBlock(channel_mid, channel_mid)
        self.conv3 = ConvBlock(channel_mid, channel_out)

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x)))


class UnetV2(nn.Module):
    def __init__(self, channel_in):
        super(UnetV2, self).__init__()
        self.conv0 = ConvBlock3(channel_in, 32, 32)
        self.conv1 = ConvBlock3(32, 64, 64)
        self.conv2 = ConvBlock3(64, 128, 128)
        self.conv3 = ConvBlock3(128, 256, 256)

        self.conv4 = ConvBlock3(256, 512, 512)

        self.conv5 = ConvBlock3(512, 256, 256)
        self.conv6 = ConvBlock3(256, 128, 128)
        self.conv7 = ConvBlock3(128, 64, 64)
        self.conv8 = ConvBlock3(64, 32, 3)

        self.upconv0 = UpConvBlock(512,256)
        self.upconv1 = UpConvBlock(256,128)
        self.upconv2 = UpConvBlock(128,64)
        self.upconv3 = UpConvBlock(64,32)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv0_down = F.max_pool2d(conv0, 2, 2)

        conv1 = self.conv1(conv0_down)
        conv1_down = F.max_pool2d(conv1, 2, 2)

        conv2 = self.conv2(conv1_down)
        conv2_down = F.max_pool2d(conv2, 2, 2)

        conv3 = self.conv3(conv2_down)
        conv3_down = F.max_pool2d(conv3, 2, 2)

        conv4 = self.conv4(conv3_down)

        conv4_up = self.upconv0(conv4)
        conv5=self.conv5(torch.cat([conv3,conv4_up],1))

        conv5_up = self.upconv1(conv5)
        conv6=self.conv6(torch.cat([conv2,conv5_up],1))

        conv6_up = self.upconv2(conv6)
        conv7=self.conv7(torch.cat([conv1,conv6_up],1))

        conv7_up = self.upconv3(conv7)
        conv8=self.conv8(torch.cat([conv0,conv7_up],1))

        return conv8

# mode=UnetV2(3)
# x=torch.randn(4,3,256,256)
# y=mode(x)


# def init_model(mod):
#     for key, param in mod.named_parameters():
#         split = key.split('.')
#         if param.requires_grad:
#             param.data = c.init_scale * torch.randn(param.data.shape).cuda()
