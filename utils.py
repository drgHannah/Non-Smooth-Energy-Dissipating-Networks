''' Useful Functions '''
import torch
import torch.nn as nn
import numpy as np
import os 
import cv2
import matplotlib.image as mpimg
import math
from skimage.color import rgb2ycbcr
from skimage import metrics
import glob
from shutil import make_archive, copy2, rmtree
import random
import torchvision.transforms.functional as TF


def get_network_name(opt):
    ''' Get networks name based on its parameters.'''

    tupels =  (opt.dataset, opt.twonorm, opt.zeta1, opt.zeta2, opt.alpha, opt.mu)
    dels = ''.join(["_%s" for _ in range(len(tupels))])
    return 'net' + dels%tupels + '.pt'

def get_available_nets(no = None):
    ''' Find all trained available networks.'''

    path = './results/'
    names = ["dataset", "twonorm", "zeta1", "zeta2"]
    iter = 0
    for f in os.listdir(path):
        iter+=1
        if no == None or iter == no:
            print("NETWORK",iter,'\n')
            splitted = f.split('_')
            for i in range(len(splitted)-1):
                print(names[i],splitted[i+1])
            print()
    return [i for i in splitted]

def save_code_to_zip(save_path):
    ''' Save python code in zip. '''

    tmp_path = os.path.join(save_path,'tmp')
    all_files = glob.glob("./*.py")
    os.makedirs(tmp_path,exist_ok=True)
    for f in all_files:
        copy2(f,tmp_path)
    make_archive(os.path.join(save_path,'saved_code'),'zip',tmp_path)
    rmtree(tmp_path)

def weights_init_kaiming(m):
    ''' Apply init network weights. '''

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def create_image(image_data, patch_size, rand_no, dev):
    ''' Create training data by extracting a patch of a certain size from an image.
    Args:
        image_data: input image
        patch_size: size of the cropping area
        rand_no: seed
        dev: device
    Returns:
        Clear and noisy image patch.
    '''

    # get data and create patch
    rng_local = random.Random(rand_no)
    im = rng_local.choice(image_data).float()
    y1 = rng_local.randint(0, im.shape[0] - patch_size - 1)
    x1 = rng_local.randint(0, im.shape[1] - patch_size - 1)            
    patch = im[y1:(y1+patch_size),x1:(x1+patch_size)] # / 255.
    Y = patch.view(1, patch_size, patch_size)
    # data augmentation (random rotation / flipping)
    hflip = random.randint(0, 1)
    vflip = random.randint(0, 1)
    rotate = random.randint(0, 3)
    if hflip == 1:
        Y = TF.to_tensor(TF.hflip(TF.to_pil_image(Y)))
    if vflip == 1:
        Y = TF.to_tensor(TF.vflip(TF.to_pil_image(Y)))
    Y = TF.to_tensor(TF.rotate(TF.to_pil_image(Y), angle = rotate * 90, fill=(0,)))
    Y = Y.to(dev)
    noisy_Y = salt_and_pepper(Y, prob=0.1).to(dev) 
    return Y, noisy_Y

def salt_and_pepper(img, prob=0.5):
    """ Applies salt and pepper noise. """

    c,w,h=img.shape
    np.random.seed(0)
    rnd = np.random.rand(c*w*h)
    noisy = img.flatten().clone()
    noisy[rnd < prob/2] = 0.
    noisy[rnd > 1 - prob/2] = 1.
    return noisy.reshape(c,w,h)

def load_image(file_i, dev = "cpu", noiseamount=0.1):
    ''' Loads validation image and applies noise.
    
    Args: 
        file_i:
            index of the image
        noiseamount:
            Propability per pixel to be corrupted for each being 0 or 1.
    Returns:
        Clear and noisy image.
    '''

    filenames = sorted(glob.glob(os.path.join('datasets', 'images', 'image_SRF_2','*_HR.png')))
    hr_file = filenames[file_i * 1 + 0]
    im_hr = mpimg.imread(hr_file)
    if len(im_hr.shape) == 2:
            im_hr = cv2.cvtColor(im_hr, cv2.COLOR_GRAY2RGB)
    im_hr_YCrCb = rgb2ycbcr(im_hr).astype(np.float32) / 255.
    imY = im_hr_YCrCb[:, :, 0].squeeze()
    Y = torch.tensor(imY)[None].to(dev)
    b = salt_and_pepper(Y, prob=noiseamount)
    return Y, b

def metrics_psnr_ssim(clean, corr):
    ''' Calculates PSNR and SSIM.

    Args:
        clean: 
            image without noise.
        corr: 
            corrupted image.
    Returns:
        PSNR and SSIM value of the corrupted image.
    '''

    img1 = clean.cpu().numpy()[0,0]
    img2 = corr.cpu().numpy()[0,0]
    psnr = metrics.peak_signal_noise_ratio(img1, img2, data_range = 1)
    ssim  = metrics.structural_similarity(img1, img2, data_range = 1)
    return psnr,ssim

def extend(input):
    ''' Creates a 2D barcode image from a 1D barcode. '''
    return input.repeat(20,1)