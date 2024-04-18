import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import config as c
from natsort import natsorted
import cv2
import numpy as np
import torch


class FHNet_Dataset(Dataset):
    def __init__(self, mode="train"):
        self.mode = mode
        if mode == 'train':
            # train
            self.files_wholeimg = natsorted(sorted(glob.glob(c.TRAIN_IMG + "/*")))
            self.files_mask = natsorted(sorted(glob.glob(c.TRAIN_MASK + "/*")))
        else:
            # test
            self.files_wholeimg = natsorted(sorted(glob.glob(c.VAL_IMG + "/*")))
            self.files_mask = natsorted(sorted(glob.glob(c.VAL_MASK + "/*")))

    def __getitem__(self, index):
        try:

            wholeimage = cv2.imread(self.files_wholeimg[index])[...,::-1]
            mask = cv2.imread(self.files_mask[index])[...,::-1]

            wholeimage=cv2.resize(wholeimage,(c.resize_w,c.resize_h))/255
            mask = cv2.resize(mask, (c.resize_w, c.resize_h))/255

            mask=mask.round()

            wholeimage = torch.from_numpy(np.transpose(wholeimage, [2,0,1]))
            mask = torch.from_numpy(np.transpose(mask, [2,0,1]))

            return wholeimage.float(),mask.float()

        except:
            return self.__getitem__(index + 1)

    def __len__(self):
            return len(self.files_wholeimg)



# Training data loader
trainloader = DataLoader(
    FHNet_Dataset(mode="train"),
    batch_size=c.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
    drop_last=True
)
# Test data loader
valloader = DataLoader(
    FHNet_Dataset(mode="val"),
    batch_size=c.batchsize_val,
    shuffle=False,
    pin_memory=True,
    num_workers=1,
    drop_last=True
)