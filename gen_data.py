''' Barcode Dataset Generation'''
import os 
import numpy as np
import treepoem
import random
import string
import torch
import glob
import zipfile
import json
import wget
import tarfile
from PIL import ImageFilter
from tqdm import tqdm
import matplotlib.image as mpimg
from skimage.color import rgb2ycbcr

with open('path.json') as f:
    path = json.load(f)
    base = path['base_path']
    image_path = os.path.join(base,path['image_path'])
    barcode_path = os.path.join(base,path['barcode_path'])

def load_barcode(data, blurfactor=2):
    ''' Load barcode.

    Args:
        data: 
            word (String) to encode
        blurfactor:
            standard deviation of the Gaussian kernel for blurring
    Returns:
        The barcode and its blurred version.
    '''
    image = treepoem.generate_barcode(
        barcode_type="code128",  
        data=data,
    )
    barcode = np.array(image.convert("1"))*1.0
    blurred = None
    if blurfactor is not None:
        blurred = np.array(image.filter(ImageFilter.GaussianBlur(blurfactor)).convert("L"))*1.0*(1.0/255.0)
    return barcode, blurred

def generate_barcodes(number, len=10, blurfactor = 1.0):
    ''' Generates and saves barcode dataset.
    
    Args:
        number: 
            number of words to encode
        len:
            length of each word to encode
        blurfactor:
            standard deviation of the Gaussian kernel for blurring
    '''

    # measure length of each dataitem
    word = ''.join("a" for i in range(len)) 
    barcode, blurred=load_barcode(word,blurfactor=blurfactor)
    m_len = barcode.shape[1]

    os.makedirs(barcode_path,exist_ok=True)
    data = []
    letters = string.ascii_letters + string.digits
    for i in tqdm(range(number)):
        word = ''.join(random.choice(letters) for i in range(len)) 
        barcode, blurred=load_barcode(word,blurfactor=blurfactor)
        h,w = barcode.shape
        if w != m_len:
            continue
        barcode=barcode[h//2]
        blurred=blurred[h//2]
        data.append((torch.tensor(barcode),torch.tensor(blurred)))
        if (i+1)%1024 == 0:
            torch.save(data, os.path.join(barcode_path,f'barcodes-{i}-{blurfactor}.pt'))
            data = []

def load_image_data():
    ''' Load data for the denoising task.'''

    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
    wget.download(url, out=base)
    my_tar = tarfile.open(os.path.join(base,'BSDS300-images.tgz'))
    my_tar.extractall(image_path) 
    my_tar.close()

def load_validation_data():
    ''' Load validation data.'''

    url = "https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip"
    wget.download(url, out=base)
    with zipfile.ZipFile(os.path.join(base,'BSD100_SR.zip'),"r") as zip_ref:
        zip_ref.extractall(image_path)

def create_noisy_data():
    ''' Create data for the denoising task.'''

    rootdirs = [os.path.join(image_path,"BSDS300/images/test/"),os.path.join(image_path,"BSDS300/images/train/")]
    imgs = [] 
    for root_dir in rootdirs:
        filenames = sorted(os.listdir(root_dir))
        filenames = sorted(glob.glob(root_dir+"/*.jpg"))
        for fn in filenames:
            print('processing %s' % fn)
            im = mpimg.imread(fn)
            im = rgb2ycbcr(im).astype(np.float32) / 255.  
            imY = im[:, :, 0].squeeze()
            torch_im = torch.tensor(imY)
            imgs.append(torch_im)
    torch.save(imgs, os.path.join(base,'images_denoise.pt'))
    return imgs

if __name__ == '__main__':

    # generate barcode data
    data=generate_barcodes(30000,5)

    # generate denoising data
    load_image_data()
    create_noisy_data()
    load_validation_data()