#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from models.model import *
from torch.utils.tensorboard.writer import SummaryWriter
import dataset
import warnings
from models.loss import *
from utils.NoiseLayer import NoiseLayer

warnings.filterwarnings("ignore")


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin, pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin / 1.0 - pred / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def load(network, pathname, netname):
    state_dicts = torch.load(pathname)
    network_state_dict = {k: v for k, v in state_dicts[netname].items() if 'tmp_var' not in k}
    network.load_state_dict(network_state_dict)


#####################
# Model initialize: #
#####################
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Model()
    net = net.to(device)
    init_model(net)

    noiselayer=NoiseLayer()
    noiselayer=noiselayer.to(device)
    noiselayer.requires_grad_(False)


    para = get_parameter_number(net)
    print(para)

    params_trainable_net = (list(filter(lambda p: p.requires_grad, net.parameters())))

    opt_g = torch.optim.Adam(list(net.parameters()), lr=c.lr_g)
    weight_scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, c.weight_step, gamma=c.gamma)

    Loss_G = FH_Loss().to(device)

    if c.train_next:
        load(net, c.MODEL_PATH + c.suffix, 'net')

    try:
        writer = SummaryWriter(comment='FH')
        iteration = 0
        loss_history = []
        for i_epoch in range(c.epochs):

            #################
            #     train:    #
            #################

            for i_batch, data in enumerate(dataset.trainloader):
                # print(i_batch)
                wholeimage_GT = data[0].to(device)
                mask_GT = data[1].to(device)
                face_GT = wholeimage_GT * mask_GT
                background_GT = wholeimage_GT * (1 - mask_GT)

                opt_g.zero_grad()
                background_hiding,zero_faceregion,face_inn = net(mask_GT,background_GT,face_GT, rev=False)
                wholeimage_hiding = background_hiding + face_GT

                wholeimage_hiding_noised=noiselayer(wholeimage_hiding)
                background_hiding_noised=wholeimage_hiding_noised*(1 - mask_GT)

                whole_reveal,face_reveal = net(mask_GT,background_hiding_noised, rev=True)

                loss_gen = Loss_G(background_GT, face_GT, background_hiding, face_reveal,zero_faceregion,whole_reveal,face_inn,wholeimage_GT)
                loss_total = loss_gen
                loss_total.backward()
                opt_g.step()


                loss_history.append([loss_total.item(), 0.])
                iteration = iteration + 1
                weight_scheduler_g.step()
                # weight_scheduler_d.step()
                if (iteration % c.SAVE_itertaion_freq == 0) & (iteration != 0):
                    epoch_losses = np.mean(np.array(loss_history), axis=0)
                    epoch_losses[1] = np.log10(opt_g.param_groups[0]['lr'])

                    print('epoch:', iteration // 1000, 'K     TOTAL_loss:', epoch_losses[0])
                    loss_history = []
                    writer.add_scalars("Train", {"Total_Loss": epoch_losses[0]},
                                       iteration)
                    torch.save({'net': net.state_dict()},
                               c.MODEL_PATH + 'model_checkpoint_%.6i' % iteration + '.pt')


                if (iteration % c.val_iteration_freq == 0) & (iteration != 0):
                    pass
                    with torch.no_grad():
                        psnr_b = []
                        psnr_f = []
                        psnr_r=[]
                        net.eval()
                        for x in dataset.valloader:
                            wholeimage_GT = x[0].to(device)
                            mask_GT = x[1].to(device)
                            face_GT = wholeimage_GT * mask_GT
                            background_GT = wholeimage_GT * (1 - mask_GT)

                            background_hiding, _, _ = net(mask_GT, background_GT, face_GT,
                                                                               rev=False)
                            wholeimage_hiding = background_hiding + face_GT

                            whole_reveal, face_reveal = net(mask_GT, background_hiding, rev=True)

                            face_GT = face_GT.cpu().numpy().squeeze() * 255
                            np.clip(face_GT, 0, 255)
                            background_GT = background_GT.cpu().numpy().squeeze() * 255
                            np.clip(background_GT, 0, 255)
                            face_reveal = face_reveal.cpu().numpy().squeeze() * 255
                            np.clip(face_reveal, 0, 255)
                            background_hiding = background_hiding.cpu().numpy().squeeze() * 255
                            np.clip(background_hiding, 0, 255)
                            wholeimage_GT = wholeimage_GT.cpu().numpy().squeeze() * 255
                            np.clip(wholeimage_GT, 0, 255)
                            whole_reveal = whole_reveal.cpu().numpy().squeeze() * 255
                            np.clip(whole_reveal, 0, 255)
                            psnr_temp = computePSNR(background_GT, background_hiding)
                            psnr_b.append(psnr_temp)
                            psnr_temp_f = computePSNR(face_GT, face_reveal)
                            psnr_f.append(psnr_temp_f)

                            psnr_temp_r = computePSNR(wholeimage_GT, whole_reveal)
                            psnr_r.append(psnr_temp_r)

                        print("PSNR  cover:", np.mean(psnr_b), '     secret:', np.mean(psnr_f), '     secret_whole:', np.mean(psnr_r))
                        writer.add_scalars("PSNR_B", {"average psnr": np.mean(psnr_b)}, iteration)
                        writer.add_scalars("PSNR_F", {"average psnr": np.mean(psnr_f)}, iteration)
                        writer.add_scalars("PSNR_R", {"average psnr": np.mean(psnr_r)}, iteration)

            if (iteration > c.iterations):
                break


        torch.save({'net': net.state_dict()},
                   c.MODEL_PATH + 'model' + '.pt')
        writer.close()

    except:
        if c.checkpoint_on_error:
            torch.save({'net': net.state_dict()},
                       c.MODEL_PATH + 'model_ABORT' + '.pt')
        raise

    finally:
        pass
