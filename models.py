''' Constrained Networks 

Modified and originally taken from

Moeller, Michael, Thomas Mollenhoff, and Daniel Cremers. 
"Controlling neural networks via energy dissipation." 
Proceedings of the IEEE/CVF International 
Conference on Computer Vision. 2019.
'''

import torch
import torch.nn as nn

class ConstrainedNet(nn.Module):
    def __init__(self, zeta1, zeta2, constrained=True, in_chn=1, out_chn=1, num_layers=17, features=64):
        """ Simple conv-relu-bn architecture, where the output is constrained to 
        maximally deviate from a given vector g in the forward pass by acos(zeta1/zeta2) degrees.

        Args:
            zeta1, zeta2: 
                projection parameters
            constrained: 
                boolean if network has projection layer
            in_chn: 
                number of channels of input
            out_chn: 
                number of output-channels
            num_layers: 
                number of layers
            features: 
                network width
        """

        super(ConstrainedNet, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=in_chn, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=out_chn, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self.channels = in_chn
        self.zeta1 = zeta1
        self.zeta2 = zeta2
        self.constrained = constrained
        
    def forward(self, x, g, f):
        """ Forward path of the constrained network.

        Args:
            x: 
                current iterate
            g:
                true gradient at x
            f:
                additional data (e.g. noisy image)
        """

        if self.constrained == False:
            return self.dncnn(f) 
        else: 
            bs = x.shape[0]
            if self.channels == 1 or self.channels == 9:
                xi = self.dncnn(x)
            else:
                xi = self.dncnn(torch.cat([x, g, f], dim=1))
            
            # differentiable parametrization of constraint set C_2
            norm_g = g.view(bs, -1).norm(dim=1)
            eta = torch.bmm(xi.view(bs, 1, -1), g.view(bs, -1, 1)).squeeze() / (torch.clamp(norm_g ** 2.0, min = 1e-6))
            xi_perp = xi - eta.view(bs, 1, 1, 1) * g
            eta_hat = torch.clamp(eta, min=self.zeta1, max=self.zeta2)
            norm_xi_perp = xi_perp.view(bs, -1).norm(dim = 1)
            radius = torch.clamp(torch.sqrt(torch.clamp(self.zeta2 ** 2.0 - eta_hat ** 2.0, min=1e-6)) * norm_g, min = 1e-6) 
            proj = xi_perp / torch.max(radius, norm_xi_perp).view(bs, 1, 1, 1)
            out = eta_hat.view(bs, 1, 1, 1) * g + radius.view(bs, 1, 1, 1) * proj
            return -out

class ConstrainedNet1D(nn.Module):
    def __init__(self, zeta1, zeta2, constrained=True, in_chn=1, out_chn=1, num_layers=17, features=64):
        """ Simple conv-relu-bn architecture, where the output is constrained to 
        maximally deviate from a given vector g in the forward pass by acos(zeta1/zeta2) degrees.

        Args:
            zeta1, zeta2: 
                projection parameters
            constrained: 
                boolean if network has projection layer
            in_chn: 
                number of channels of input
            out_chn: 
                number of output-channels
            num_layers: 
                number of layers
            features: 
                network width
        """

        super(ConstrainedNet1D, self).__init__()   
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv1d(in_channels=in_chn, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers-2):
            layers.append(nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm1d(features))
            layers.append(nn.ReLU(inplace=True))
            
        layers.append(nn.Conv1d(in_channels=features, out_channels=out_chn, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

        self.channels = in_chn
        self.zeta1 = zeta1
        self.zeta2 = zeta2
        self.constrained = constrained
        
    def forward(self, x, g, f):
        """ Forward path of the constrained network.

        Args:
            x: 
                current iterate
            g:
                true gradient at x
            f:
                additional data (e.g. blurry image)
        """

        if self.constrained == False:
            return self.dncnn(f) 
        else: 
            bs = x.shape[0]
            if self.channels == 1 or self.channels == 9:
                xi = self.dncnn(x)
            else:
                xi = self.dncnn(torch.cat([x, g, f], dim=1))
            xi=xi[:,None,:,:] 
            g=g[:,None,:,:] 

            # differentiable parametrization of constraint set C_2
            norm_g = g.view(bs, -1).norm(dim=1)
            eta = torch.bmm(xi.view(bs, 1, -1), g.view(bs, -1, 1)).squeeze() / (torch.clamp(norm_g ** 2.0, min = 1e-6))
            xi_perp = xi - eta.view(bs, 1, 1, 1) * g
            eta_hat = torch.clamp(eta, min=self.zeta1, max=self.zeta2)
            norm_xi_perp = xi_perp.view(bs, -1).norm(dim = 1)
            radius = torch.clamp(torch.sqrt(torch.clamp(self.zeta2 ** 2.0 - eta_hat ** 2.0, min=1e-6)) * norm_g, min = 1e-6) 
            proj = xi_perp / torch.max(radius, norm_xi_perp).view(bs, 1, 1, 1)
            out = eta_hat.view(bs, 1, 1, 1) * g + radius.view(bs, 1, 1, 1) * proj
            return -out