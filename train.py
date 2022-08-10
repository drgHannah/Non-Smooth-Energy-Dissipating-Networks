''' Training of the Constrained Network

Modified and originally taken from

Moeller, Michael, Thomas Mollenhoff, and Daniel Cremers. 
"Controlling neural networks via energy dissipation." 
Proceedings of the IEEE/CVF International 
Conference on Computer Vision. 2019.
'''

import sys
import os
import copy
import tqdm
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import descent_op
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from models import ConstrainedNet, ConstrainedNet1D
from datasets import DenoisingDataset2D, BarcodeDataset
from utils import weights_init_kaiming,get_network_name,save_code_to_zip,extend
from validation import validate_and_save_plot

# default parameters
bs_default = 128#128
lr_default = 1e-4
zeta1_default = 25.0
zeta2_default = 10000.
iters_default = 10
path_default = '.'
num_features_default = 64
num_layers_default = 17
kaiming_default = 1
patch_size_default = 40
pretrained_init_default = 'no'
max_iters_default = 30000
dev = torch.device('cuda')

mean_ssim,max_it=0,0
dev = torch.device('cuda')

parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=bs_default, help="batch size")
parser.add_argument("--lr", type=float, default=lr_default, help="learning rate")
parser.add_argument("--zeta1", type=float, default=zeta1_default, help="constraint set parameter")
parser.add_argument("--zeta2", type=float, default=zeta2_default, help="constraint set parameter")
parser.add_argument("--iters", type=int, default=iters_default, help="number of iters to gen data")
parser.add_argument("--max_iters", type=int, default=max_iters_default, help="number of training steps")
parser.add_argument("--path", default=path_default, help="output path")
parser.add_argument("--num_features", type=int, default=num_features_default, help="number of features")
parser.add_argument("--num_layers", type=int, default=num_layers_default, help="number of layers")
parser.add_argument("--kaiming", type=int, default=kaiming_default, help="use kaiming init")
parser.add_argument("--patch_size", type=int, default=patch_size_default, help="patch size for SRes experiment")
parser.add_argument("--pretrained_init", default=pretrained_init_default, help="initialize with pretrained net")
parser.add_argument("--dataset", default="sres", help="either <barcode> or <denoise2d>")
parser.add_argument("--twonorm", type=int, default=0, help=" twonorm instead of one norm for denoising")
parser.add_argument("--blurfactor", type=int, default=1.0, help=" blurfactor for barcode dataset")
parser.add_argument("--mu", type=float, default=1, help="parameter for Moreau-Yosida regularization")
parser.add_argument("--alpha", type=int, default=1, help="weighting of the energy term")

opt = parser.parse_args()
params = vars(opt)

# create filename of experiment
experiment_id = time.strftime("%b%d_%H-%M-%S")
network_name = get_network_name(opt)
print('Filename:',experiment_id)
opt.path = os.path.join(opt.path, experiment_id)
os.makedirs(opt.path, exist_ok=True)
writer = SummaryWriter(os.path.join(opt.path,'runs'))
save_code_to_zip(opt.path)

# print parameters
print('Training parameters:')
original_stdout = sys.stdout 
with open(os.path.join(opt.path,'experiment.txt'), 'w') as f:
    for k, v in params.items():
        sys.stdout = f 
        print(k, '=', v)
        sys.stdout = original_stdout
        print(k, '=', v)

# load network and dataloader
if opt.dataset == 'denoise2d':
    # create dataset and dataloader
    do = descent_op.Decent_Op(alpha = opt.alpha, twonorm=opt.twonorm, mu = opt.mu)
    dl = DataLoader(DenoisingDataset2D(max_it = opt.iters, patch_size = opt.patch_size, grad_net = None, descent_operation = do),
                    batch_size = opt.bs,
                    shuffle = True,
                    num_workers = 1)
    # load network
    net = ConstrainedNet(in_chn = 3, zeta1 = opt.zeta1, zeta2 = opt.zeta2, \
                        features = opt.num_features, num_layers = opt.num_layers, constrained=True).to(dev)
elif opt.dataset == 'barcode':
    # create dataset and dataloader
    do = descent_op.Decent_Op_Barcode(mu = opt.mu)
    dl = DataLoader(BarcodeDataset(max_it = opt.iters, grad_net = None, descent_operation = do,blurfactor=opt.blurfactor),
                    batch_size = opt.bs,
                    shuffle = True,
                    num_workers = 1)
    # load network
    net = ConstrainedNet1D(in_chn = 3, zeta1 = opt.zeta1, zeta2 = opt.zeta2, \
                            features = opt.num_features, num_layers = opt.num_layers, constrained=True).to(dev)

# load pretrained weights
if opt.pretrained_init != 'no':
    state_dict = torch.load(opt.pretrained_init)
    new_state_dict = {}
    for key in state_dict:
        if key[13] == '0':
            weights = state_dict[key]
            weights = weights.expand(64, 3, 3, 3)  # weights are copied for each channel                                 
            new_state_dict[ key[7: ] ] = weights
        else:
            new_state_dict[ key[7: ] ] = state_dict[key]
    net.load_state_dict(new_state_dict)
if opt.kaiming == 1:
    net.apply(weights_init_kaiming)

# select optimizer
opti = optim.Adam(net.parameters(), lr = opt.lr)

# run optimzation
for it in tqdm.tqdm(range(opt.max_iters)):

    # get data
    try:
        (X, G, Y, f, numit) = next(data_iter)
    except:
        print("...new dataloader...")
        data_iter = iter(dl) 
        (X, G, Y, f, numit) = next(data_iter)
    
    # validate
    if (it)%150==0:
        validate_and_save_plot(it, opt, writer, net, do)

    X = X.to(dev)  # inputs
    G = G.to(dev)  # gradients
    Y = Y.to(dev)  # targets
    f = f.to(dev)  # degraded data

    opti.zero_grad()
    net.train()

    # net should predict update direction V, so that if we move along V we go from X to ground-truth Y
    pred = net(X, G, f)
    V = Y - X
    loss = ((pred - V) ** 2.0).sum() / (2.0 * Y.shape[0])
    loss.backward()
    opti.step()

    # plot
    writer.add_scalar('Loss/train', loss, it)
    with torch.no_grad():
        if it%150==0:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20,5))
            if opt.dataset == 'barcode':
                ax1.imshow(extend(pred[0,0]).cpu().detach(),vmin=0,vmax=1,cmap="gray")
                ax1.set_title('prediction')
                ax2.set_title('prediction+f / prediction+X')
                ax2.imshow(extend(pred[0,0]-V[0,0]+Y[0,0]).cpu().detach(),vmin=0,vmax=1,cmap="gray")
                ax3.set_title('f')
                ax3.imshow(extend(f[0,0]).cpu().detach(),vmin=0,vmax=1,cmap="gray")
                ax4.imshow(extend(Y[0,0]).cpu().detach(),vmin=0,vmax=1,cmap="gray")
                ax4.set_title('Y')
            if opt.dataset == 'denoise2d':
                ax1.imshow(pred[0,0].cpu().detach(),vmin=0,vmax=1,cmap="gray")
                ax1.set_title('prediction')
                ax2.set_title('prediction+f / prediction+X')
                ax2.imshow((pred[0,0]-V[0,0]+Y[0,0]).cpu().detach(),vmin=0,vmax=1,cmap="gray")
                ax3.set_title('f')
                ax3.imshow((f[0,0]).cpu().detach(),vmin=0,vmax=1,cmap="gray")
                ax4.imshow((Y[0,0]).cpu().detach(),vmin=0,vmax=1,cmap="gray")
                ax4.set_title('Y')
            writer.add_figure('Training/prediction', fig, it)
            plt.close()

    if it%10==0:
        print('iteration %07d, mini-batch loss: %f' % (it, loss.item()))
    
    # update the dataset based on current model every 100 steps
    if it % 100 == 1 and it > 1:
        net_cur = copy.deepcopy(net)
        if opt.dataset == 'denoise2d':
            dl = DataLoader(DenoisingDataset2D(max_it = opt.iters, grad_net = net_cur, patch_size = opt.patch_size, descent_operation=do), \
                                batch_size = opt.bs, \
                                shuffle = True)
        elif opt.dataset == 'barcode':
            dl = DataLoader(BarcodeDataset(max_it = opt.iters, grad_net = net_cur, descent_operation = do, blurfactor=opt.blurfactor),
                            batch_size = opt.bs,
                            shuffle = True)
        data_iter = None

    # save network (if not in debug mode)
    gettrace = getattr(sys, 'gettrace', None)
    if not gettrace() and (it+1) % 100 == 1:
            torch.save(net, os.path.join(opt.path, get_network_name(opt)))

# save final network (if not in debug mode)
gettrace = getattr(sys, 'gettrace', None)
if not gettrace():
    torch.save(net, os.path.join(opt.path, get_network_name(opt)))

