''' Descent Step Calculation '''
import torch
import numpy as np
from energies import Energies

class Decent_Op():
    ''' Runs the descent steps. '''
    
    def __init__(self, alpha = 1, twonorm = False, mu = 1) -> None:
        ''' Initialized the descent steps class.

        Args:
            alpha: 
                weighting of the energy term, e.g. alpha * ||x-y|| (default: 1.0)
            twonorm: 
                energy term has two norm loss (default: False)
            mu:
                parameter for Moreau-Yosida regularization
        '''
        self.twonorm = twonorm
        self.mu = mu
        if twonorm == True:
            self.gradient = Energies.gradient_l2_alpha(alpha)
            self.energy = Energies.energy_l2_alpha(alpha)
            self.L = alpha
        else:
            self.gradient = Energies.gradient_l1_alpha(alpha,mu)
            self.energy = Energies.moreau_l1_alpha(alpha,mu)
            self.L = 1/mu

    def get_stepsize(self, zeta1, zeta2, L, with_net = False):
        ''' Calculates steps size.
        Args:
            zeta1, zeta2: 
                projection parameters
            L: 
                lipschitz constant of the gradient of the Moreau envelope
            with_net:
                boolean value - descent step is predicted by a network (True) or not (False)
        Returns:
            Step size.
        '''
        if with_net:
            return zeta1 / ((zeta2**2)*L)
        else:
            return 0.9 * 2/L


    def dec_op(self, X, b, num_it, grad_net):
        ''' Runs descent step.
        Args:
            X: 
                current input
            b: 
                degraded data (e.g. ||x-b||)
            num_it:
                number of descent steps
            grad_net:
                not None if netowrk is supposed to predict descent direction
        Returns:
            Prediction and corresponding gradient.
        '''
        if num_it < 1:
            grad = self.gradient(X, b).to(X.device)

        if grad_net is not None:
            tau_d = self.get_stepsize(grad_net.zeta1, grad_net.zeta2, self.L, with_net=True)
        else:
            tau_d = self.get_stepsize(1, 1, self.L, with_net=False)

        f = b.detach()
        for i in range(num_it):
            # Recalculate step size for faster convergence
            tau_v = 1/(i+1)
            tau = np.maximum(tau_v,tau_d)
            # descent step (except in last iteration)
            grad = self.gradient(X, b)
            if i < num_it - 1:
                # descent direction
                if grad_net is None: 
                    ddir = -grad
                else:       
                    with torch.no_grad():          
                        network_neg_grad = grad_net(X[None], grad[None], f[None])
                        ddir = network_neg_grad[0]
                # update step
                X = X + tau * ddir
        return X.detach(), grad.detach()


class Decent_Op_Barcode():
    ''' Runs the descent steps for the deblurring barcode task. '''
    
    def __init__(self, mu = 1) -> None:
        ''' Initialized the descent steps class.

        Args:
            mu:
                parameter for Moreau-Yosida regularization
        '''
        self.mu = mu

        # Set Energy and Gradient
        self.gradient = Energies.gradient_moreau_W(self.mu)
        self.energy = Energies.moreau_W(self.mu)
        self.L = 1/self.mu

    def get_stepsize(self, zeta1, zeta2, L, with_net=False):
        ''' Calculates steps size.

        Args:
            zeta1, zeta2: 
                projection parameters
            L: 
                lipschitz constant of the gradient of the Moreau envelope
            with_net:
                boolean value - descent step is predicted by a network (True) or not (False)
        Returns:
            Step size.
        '''
        if with_net:
            return 0.9 * zeta1 / ((zeta2**2)*L)
        else:
            return 0.9 * 2/L

    def dec_op(self, X, b, num_it, grad_net):
        ''' Runs descent step.

        Args:
            X: 
                current input
            b: 
                degraded data (e.g. ||x-b||)
            num_it:
                number of descent steps
            grad_net:
                not None if netowrk is supposed to predict descent direction
        Returns:
            Prediction and corresponding gradient.
        '''

        if num_it < 1:
            grad = self.gradient(X)
        for i in range(num_it):
            # Recalculate step size for faster convergence
            tau_v = 1/(i+1) * 2.6
            tau = tau_v
            # descent step (except in last iteration)
            grad = self.gradient(X)
            if i < num_it - 1:
                # descent direction
                if grad_net is None: 
                    ddir = -grad
                else:       
                    with torch.no_grad():          
                        network_neg_grad = grad_net(X[None], grad[None], b[None])
                        ddir = network_neg_grad[0,0]
                X = X + tau * ddir
        return X.detach(), grad.detach()

