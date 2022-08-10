''' Datasets '''
import glob
import torch
import torch.utils.data as data
import itertools
from utils import create_image
import json
import os

with open('path.json') as f:
    path = json.load(f)
    base = path['base_path']
    barcode_path = os.path.join(base,path['barcode_path'])

class DenoisingDataset2D(data.Dataset):
    ''' Denoising Dataset '''
 
    def __init__(self, max_it = 10, grad_net = None, patch_size = 64, descent_operation = None):
        ''' Initializes Dataset.

        Args: 
            max_it: 
                maximum descent steps
            grad_net: 
                if not None - network that predicts the descent step
            patch_size: 
                size of training-patches
            descent_operation: 
                Decent_Op() object - calculates the descent step
        '''

        self.descent_op = descent_operation
        self.num_patches = 5 * 1024 * 1024 
        self.patch_size = patch_size
        self.max_it = max_it
        self.grad_net = grad_net
        self.image_data = torch.load(os.path.join(base, 'images_denoise.pt'))

    def __len__(self):
        ''' returns size of "virtual" datset'''

        return self.num_patches * self.max_it

    def __getitem__(self, idx):
        ''' Returns trainings sample, for a maximal specified number of descent steps.

        Args:
            idx: sample idx
        Returns: 
            resulting sample, predicted gradient at this position, clear image, noisy image, number of descent steps
        '''
        
        if self.grad_net is None:  # if there is no deep net, just work on cpu
            dev = torch.device('cpu')            
        else:
            dev = torch.device('cuda')

        num_it = (idx % self.max_it)

        Y, b = create_image(self.image_data, self.patch_size, idx//self.max_it, dev)

    
        X = b.mean(dim=(1,2)).view(1,1,1).expand(1, self.patch_size, self.patch_size).clone()
        grad = torch.zeros_like(X)

        with torch.no_grad():
            X, grad = self.descent_op.dec_op(X.detach(), b, num_it, self.grad_net)

        return X.detach(), grad, Y.detach(), b.detach(), num_it


class BarcodeDataset(data.Dataset):
    ''' Barcode Deblurring Dataset '''
    def __init__(self, max_it = 10, grad_net = None, descent_operation = None, blurfactor = 1.0):
        ''' Initializes Dataset.
        
        Args: 
            max_it: 
                maximum descent steps
            grad_net: 
                if not None - network that predicts the descent step
            descent_operation: 
                Decent_Op() object - calculates the descent step
            blurfactor: 
                standard deviation of the Gaussian kernel for blurring
        '''
    
        self.descent_op = descent_operation
        self.num_patches = 5 * 1024 * 1024 
        self.max_it = max_it
        self.grad_net = grad_net

        # load data
        no = len(glob.glob("./datasets/barcode/*")) * 1024 - 1
        self.image_data=[]
        for i in range(2,no//1024 + 1):
            val = i*1024 - 1
            
            self.image_data.append(torch.load(os.path.join(barcode_path,f'barcodes-{val}-{blurfactor}.pt')))
        self.image_data = list(itertools.chain(*self.image_data))
        self.leng = len(self.image_data)

        # remove data of wrong size
        removed = 0
        maxval = self.leng
        while i < maxval:
            data_elem = self.image_data[i]
            if data_elem[0].shape[0] != self.image_data[0][0].shape[0]:
                self.image_data.pop(i)
                removed+=1
                maxval-=1
            else:
                i+=1

        print(f"Removed {removed}/{self.leng} --> new length {len(self.image_data)}")
        self.leng = len(self.image_data)

    def __len__(self):
        return self.num_patches * self.max_it

    def __getitem__(self, idx):
        ''' Returns trainings sample, for a maximal specified number of descent steps.

        Args:
            idx: sample idx
        Returns: 
            resulting sample, predicted gradient at this position, clear data, degraded data, number of descent steps
        '''

        # Device
        if self.grad_net is None:  # if there is no deep net, just work on cpu
            dev = torch.device('cpu')            
        else:
            dev = torch.device('cuda')
        # Number Iterations
        num_it = (idx % self.max_it)
        # Get data
        Y,b = self.image_data[idx % self.leng]
        Y = Y.float()[None].to(dev)
        b = b.float()[None].to(dev)
        # Create Data
        X = b.clone()
        grad = torch.zeros_like(X)
        with torch.no_grad():
           X, grad = self.descent_op.dec_op(X, b, num_it, self.grad_net)
        return X.detach(), grad, Y.detach(), b.detach(), num_it



