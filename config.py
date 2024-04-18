# PATH
TRAIN_IMG = r'D:\sxf_w2\datasets\celeba-256\train'
TRAIN_MASK = r'D:\sxf_w2\datasets\celeba-256\train_mask'

VAL_IMG = r'D:\sxf_w2\datasets\celeba-256\validation'
VAL_MASK = r'D:\sxf_w2\datasets\celeba-256\validation_mask'

MODEL_PATH = r'pth/'
suffix = r''

# arcface_pth = r'pth\Arcface.pth'

# setting
train_next = False

epochs = 120
iterations=230000 # 240k | 180k

val_freq = 2
SAVE_freq = 2

val_iteration_freq=15000
SAVE_itertaion_freq=3000


simnet_pth=r'jpegsim80k.pt'

resize_h = 256
resize_w = 256
batch_size = 8#
batchsize_val =8
checkpoint_on_error = True
# network
lr_g = 1e-4
lr_d = 1e-4
weight_step=70000
init_scale=0.1
gamma=0.6

bg_weight = 14
face_weight = 25
facedwt_weight=0
zero_faceregion_weight=0
whole_reveal_weight=7

noise=True

crop_h=64
crop_w=64
