import torch.optim
import torch.nn as nn
import config as c
from unetV2 import UnetV2


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.preprocess = UnetV2(3)
        self.encoder = UnetV2(6)
        # self.face_pro=ResFace(Bottleneck, [4, 4, 4, 4, 4, 4])
        # self.back_pro = ResBack(Bottleneck, [4, 4, 4, 4, 4, 4])
        # self.reconstruction=Reconstruction()
        self.decoder = UnetV2(3)

    def forward(self, mask, bg, face=None, rev=False):

        if not rev:  # 1bg 2 face
            face_pro=self.preprocess(face)
            # back_pro=self.back_pro(bg)
            out=self.encoder(torch.cat([bg,face_pro],1))
            out_bg = out * (1 - mask)
            zero_region = out * mask

            return out_bg, zero_region, None

        else:
            out = self.decoder(bg)

            out_face = out * mask

            return out, out_face



def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = c.init_scale * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)
