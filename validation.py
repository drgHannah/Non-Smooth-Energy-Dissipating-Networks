import os
import torch
import json
import numpy as np
from tqdm import tqdm
from utils import load_image, metrics_psnr_ssim,extend
import matplotlib.pyplot as plt

with open('path.json') as f:
    path = json.load(f)
    base = path['base_path']
    barcode_path = os.path.join(base,path['barcode_path'])

def validate(model, gradient, energy, tau, max_it=25, patchsize = None):
    ''' Validate model

    Args:
        model: 
            trained model
        gradient:
            gradient function of energy 
        energy:
            energy function
        tau:
            learning rate
        max_it:
            number of descent steps
        patchsize:
            if not None: run validation on patched images
    Returns:
        mean (maximal) PSNR value for 100 validation images, mean (maximal) SSIM value, 
        mean iteration of maximal PSNR, example image at step with maximal PSNR, 
        PSNR curve for this exmaple image, and energy curve for this example image
    '''

    with torch.no_grad():
        model.eval()
        startpixel = 50
        all_psnrs = []
        all_ssim = []
        all_maxit = []
        device='cuda'
        text=""
        if patchsize is not None:
            text = "Patched: "
        example_energy = []
        for i in tqdm(range(100)):
            # load images
            no_noise, f = load_image(i)
            no_noise = no_noise[None]
            f = f[None].to(device)
            if patchsize is not None:
                no_noise = no_noise[:,:,startpixel:startpixel+patchsize,startpixel:startpixel+patchsize]
                f = f[:,:,startpixel:startpixel+patchsize,startpixel:startpixel+patchsize]
            psnr_img = []
            ssim_img = []
            example_imgs = []
            X = f[0].mean(dim=(1,2)).view(1,1).expand(f.shape[-2], f.shape[-1]).clone()[None,None].to(device)
            for j in range(max_it):
                # calculate learning rate
                tau_v = 1/(j+1)
                tau_r = np.maximum(tau_v,tau)
                # gradient
                grad = gradient(X, f).to(device)
                # descent direction
                if model is None: 
                    ddir = -grad
                else:       
                    with torch.no_grad():          
                        network_neg_grad = model(X, grad, f)
                        ddir = network_neg_grad[0]
                # PSNR and saves
                psnr, ssim = metrics_psnr_ssim(no_noise, X)
                psnr_img.append(psnr)
                ssim_img.append(ssim)
                if i == 0:
                    example_imgs.append(X[0,0].detach().cpu())
                    example_energy.append(energy(X, f).cpu())
                # step
                X = X + tau_r * ddir
            # save values at maximal PSNR
            max_it_psnr = np.argmax(psnr_img)
            all_maxit.append(max_it_psnr)
            all_psnrs.append(psnr_img[max_it_psnr])
            all_ssim.append(ssim_img[max_it_psnr])
            if i == 0:
                max_img = example_imgs[max_it_psnr]
                max_img_psnr_curve = ssim_img
        # mean all maximal PSNR values
        mean_psnr = np.mean(all_psnrs)
        mean_ssim = np.mean(all_ssim)
        maxit_val = np.mean(all_maxit)
        print(text+'Mean PSNR: %.4f, Mean SSIM: %.4f, Iteration %.4f \n'%(mean_psnr,mean_ssim,maxit_val))
    return mean_psnr,mean_ssim, maxit_val,max_img,max_img_psnr_curve,example_energy

def validate_barcode(model, gradient, energy, tau, max_it=25,blurfactor = 1.0):
    ''' Validate model for barcode deblurring task

    Args:
        model: 
            trained model
        gradient:
            gradient function of energy 
        energy:
            energy function
        tau:
            learning rate
        max_it:
            number of descent steps
        blurfactor:
            blur factor of blurred validation images
    Returns:
        mean SSIM value for 100 validation images at maximal PSNR, 
        mean iteration of maximal PSNR, example image at last descent step, 
        PSNR curve for this example image, and energy curve for this example image
    '''

    with torch.no_grad():
        model.eval()
        device='cuda'
        all_ssim = []
        all_maxit = []
        example_imgs = []
        example_energy = []
        # loads data from first saved barcode set (not used during training)
        image_data = torch.load(os.path.join(barcode_path,f'barcodes-{1023}-{blurfactor}.pt')) 
        for i in tqdm(range(20)):
            # load image
            Y, f = image_data[i]
            Y = Y[None,None].to(device)
            f = f[None,None].to(device)
            simil_img = []
            X = f.clone().to(device)
            for j in range(max_it):
                # calculate learning rate
                tau_v = 1/(j+1) * 2.6
                tau_r = np.maximum(tau_v,tau)
                # gradient
                grad = gradient(X).to(device)
                # descent direction
                if model is None: 
                    ddir = -grad
                else:       
                    with torch.no_grad():          
                        network_neg_grad = model(X.float(), grad.float(), f.float())
                        ddir = network_neg_grad[0]
                # saves 
                simil = torch.norm(Y-X)
                simil_img.append(simil.item())
                if i == 0:
                    example_imgs.append(X[0,0].detach().cpu())
                    example_energy.append(energy(X).cpu())
                # step
                X = X + tau_r * ddir
            # save values at maximal PSNR
            max_it_psnr = np.argmin(simil_img)
            all_maxit.append(max_it_psnr)
            all_ssim.append(simil_img[max_it_psnr])
            if i == 0:
                max_img = example_imgs[-1]
                max_img_psnr_curve = simil_img
        # mean all maximal PSNR values
        mean_ssim = np.mean(all_ssim)
        maxit_val = np.mean(all_maxit)
        print('Validation: Mean Simil: %.4f, Iteration %.4f \n'%(mean_ssim,maxit_val))
    return mean_ssim, maxit_val, max_img, max_img_psnr_curve, example_energy

def validate_and_save_plot(it, opt, writer, net, do):
    ''' Validate and Plot Training Results.
    
    Args: 
        it: 
            current training iteration
        opt: 
            settings
        writer: 
            summarywriter to save results
        net:
            trained network
        do: 
            Decent_Op() object
    '''

    if opt.dataset == 'barcode':
        mean_ssim, max_it,max_img, mi_psnr, mi_energy = validate_barcode(net, do.gradient, do.energy, 
                                                        do.get_stepsize(net.zeta1, net.zeta2, do.L, True), 30, blurfactor=opt.blurfactor)
        writer.add_scalar('Validation/valid_ssim', mean_ssim, it)
        writer.add_scalar('Validation/valid_iter', max_it, it)
        fig, (ax1, ax2,) = plt.subplots(1, 2,figsize=(10,5))
        fig.suptitle(str(it))
        ax1.imshow(extend(max_img),vmin=0,vmax=1,cmap="gray")
        ax2.plot(mi_energy)
        ax1.set_title('last image')
        ax2.set_title('energy')
        writer.add_figure('Validation/example-image', fig, it)
        plt.close()
    if opt.dataset == 'denoise2d':
        mean_psnr,mean_ssim,max_it,max_img, mi_psnr, mi_energy = validate(net, do.gradient, do.energy, do.get_stepsize(net.zeta1, net.zeta2, do.L, True), opt.iters)
        writer.add_scalar('Validation/valid_psnr', mean_psnr, it)
        writer.add_scalar('Validation/valid_ssim', mean_ssim, it)
        writer.add_scalar('Validation/valid_iter', max_it, it)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(30,10))
        fig.suptitle(str(it))
        ax1.imshow(max_img,vmin=0,vmax=1,cmap="gray")
        ax2.plot(mi_psnr)
        ax3.plot(mi_energy)
        ax1.set_title('image at max psnr')
        ax2.set_title('psnr')
        ax3.set_title('energy')
        writer.add_figure('Validation/example image', fig, it)
        plt.close()