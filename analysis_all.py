import numpy as np
import math
import cv2
import glob
import config as c


def calculate_apd(img1, img2):  # MAE

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    apd = np.mean(np.abs(img1 - img2))
    if apd == 0:
        return float('inf')

    return np.mean(apd)


def calculate_rmse(img1, img2):
    """
    Root Mean Squared Error
    Calculated individually for all bands, then averaged
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    rmse = np.sqrt(mse)

    return np.mean(rmse)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_part(img1, img2,mask):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)
    # mse = np.mean((img1 - img2) ** 2)
    mse=np.sum(((img1 - img2) ** 2)*mask)/(np.sum(mask))
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    GT_C = r'D:\sxf_w2\datasets\CelebA\crop_img\test'
    GT_S = r'D:\sxf_w2\datasets\CelebA\crop_img\test'

    path = 'test'
    cover_path = path + '/stego_ori'
    secret_path = path + '/reveal_whole_ori'
    cover_list = sorted(glob.glob(cover_path + "/*"))
    secret_list = sorted(glob.glob(secret_path + "/*"))

    GT_C_list = sorted(glob.glob(GT_C + "/*"))
    GT_S_list = sorted(glob.glob(GT_S + "/*"))

    MASK_path=r'D:\sxf_w2\datasets\CelebA\whole_img\test_mask'
    MASK_list = sorted(glob.glob(MASK_path + "/*"))

    print(len(cover_list), '对')

    apd_C = []
    rmse_C = []
    psnr_C = []
    ssim_C = []

    psnr_part_B = []

    apd_S = []
    rmse_S = []
    psnr_S = []
    ssim_S = []

    psnr_part_F = []

    for i in range(len(cover_list)):
        # cover = []
        # secret = []
        # GT_cover = 0
        # GT_secret = 0

        print(i)

        cover=cv2.imread(cover_list[i])
        secret=cv2.imread(secret_list[i])

        GT_cover=cv2.imread(GT_C_list[i])
        GT_secret=cv2.imread(GT_S_list[i])

        GT_cover = cv2.resize(GT_cover, (c.resize_w, c.resize_h))
        GT_secret = cv2.resize(GT_secret, (c.resize_w, c.resize_h))

        mask=cv2.imread(MASK_list[i])
        mask = cv2.resize(mask, (c.resize_w, c.resize_h))
        mask = np.array(mask).astype(np.float64)
        mask = (mask/255.0).round()

        cover = np.array(cover).astype(np.float64)
        secret = np.array(secret).astype(np.float64)
        GT_cover = np.array(GT_cover).astype(np.float64)
        GT_secret = np.array(GT_secret).astype(np.float64)


        '''
        apd
        '''
        apd_C.append(calculate_apd(GT_cover, cover))
        apd_S.append(calculate_apd(GT_secret, secret))

        '''
        psnr
        '''
        psnr_C.append(calculate_psnr(GT_cover, cover))
        psnr_S.append(calculate_psnr(GT_secret, secret))

        '''
        ssim
        '''
        ssim_C.append(calculate_ssim(GT_cover,cover))
        ssim_S.append(calculate_ssim(GT_secret,secret))

        '''
        psnr_B psnr_F
        '''
        psnr_part_B.append(calculate_psnr_part(GT_cover,cover,(1-mask)))
        psnr_part_F.append(calculate_psnr_part(GT_secret,secret,mask))


    apd_C = np.mean(apd_C)
    # rmse_C = rmse_C/len(cover_list)
    psnr_C = np.mean(psnr_C)
    ssim_C = np.mean(ssim_C)
    #
    apd_S = np.mean(apd_S)
    # rmse_S = rmse_S/len(cover_list)
    psnr_S = np.mean(psnr_S)
    ssim_S = np.mean(ssim_S)

    psnr_part_B = np.mean(psnr_part_B)
    psnr_part_F = np.mean(psnr_part_F)

print(path)
print('----------  APD  ---------')
print('Cover:', apd_C, '\t\t\tSecret:', apd_S)

print('----------  PSNR  ---------')
print('Cover:', psnr_C, '\t\t\tSecret:', psnr_S)

print('----------  SSIM  ---------')
print('Cover:',ssim_C,'\t\t\tSecret:',ssim_S)

print('----------  PSNR_part  ---------')
print('BG:',psnr_part_B,'\t\t\tF:',psnr_part_F)