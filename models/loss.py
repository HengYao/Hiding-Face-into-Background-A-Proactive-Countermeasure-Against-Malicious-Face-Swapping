import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
import config as c


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_HH = x1 - x2 - x3 + x4

    return x_HH


class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class FH_Loss(nn.Module):
    def __init__(self):
        super(FH_Loss, self).__init__()

        self.bg_weight = c.bg_weight
        self.face_weight = c.face_weight
        self.facedwt_weight = c.facedwt_weight
        self.zero_faceregion_weight=c.zero_faceregion_weight
        self.whole_reveal_weight=c.whole_reveal_weight
        # self.zero_bgregion_weight=c.zero_bgregion_weight
        # self.inn_weight = c.inn_weight

        # self.id_weight=c.id_weight

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def background_loss(self, x, y):
        loss = self.l2(x, y)
        return loss

    def face_loss(self, x, y):
        loss = self.l2(x, y)
        return loss

    def face_DWTloss(self, x, y):
        x_HH = dwt_init(x)
        y_HH = dwt_init(y)
        loss = self.l2(x_HH, y_HH)
        return loss

    def zeroloss(self, x):
        loss = self.l2(x, torch.zeros_like(x))
        return loss

    def id_loss(self, z_id_X, z_id_Y):
        inner_product = (torch.bmm(z_id_X.unsqueeze(1), z_id_Y.unsqueeze(2)).squeeze())
        return self.l1(torch.ones_like(inner_product), inner_product)

    def inn_loss(self, x, y):
        loss = self.l2(torch.mul(x,y), torch.zeros_like(x))
        return loss

    def forward(self, background_GT, face_GT, background_Hiding, face_Reveal,zero_faceregion,whole_reveal,face_inn,wholeimage_GT):
        bg_loss = self.background_loss(background_GT, background_Hiding)

        face_loss = self.face_loss(face_GT, face_Reveal)

        facedwt_loss = self.face_DWTloss(face_GT, face_Reveal)

        zero_faceregion_loss=self.zeroloss(zero_faceregion)
        whole_reveal_loss=self.face_loss(wholeimage_GT,whole_reveal)
        # zero_bgregion_loss=self.zeroloss(zero_bgregion)

        # zero_inn_loss=self.inn_loss(face_GT,face_inn)

        return self.bg_weight * bg_loss +  self.face_weight * face_loss  + self.facedwt_weight * facedwt_loss+self.zero_faceregion_weight*zero_faceregion_loss+self.whole_reveal_weight*whole_reveal_loss#+self.inn_weight*zero_inn_loss
