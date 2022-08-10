''' Energies '''
import torch

class Energies():
    ''' Includes energy functions and gradients.'''

    def __init__(self) -> None:
        pass

    #####################################################################################
    # Moreau Envelope and its Gradient of (u<0.5) * ||u||_1 + (u>0.5)* ||u-1||_1
    #####################################################################################
    @staticmethod
    def energy_W(u): 
        return (torch.abs((u-0.5)**2-0.25)).sum()

    @staticmethod
    def prox_W(v, mu):
        res = torch.zeros_like(v)
        res[v > 0.5] = 1

        if mu != 0.5:
            if -2*mu+1 > 0:
                case1 = ((mu - v) / (2 * mu - 1))
                bed1 = torch.logical_and(case1 > 0, case1 < 1)
                res[bed1] = case1[bed1]
            if 2*mu+1 > 0:
                case2 = ((v + mu) / (2 * mu + 1))
                bed2 = torch.logical_or(case2 <= 0, case2 >= 1)
                res[bed2] = case2[bed2]

        return res

    @staticmethod
    def gradient_moreau_W(mu):
        def gradient(u):
            return 1/mu * (u - Energies.prox_W(u, mu))
        return gradient

    @staticmethod
    def moreau_W(mu):
        def moreau(u):
            return Energies.energy_W(Energies.prox_W(u,mu)) + (1/(2*mu))*((u-Energies.prox_W(u,mu))**2).sum()
        return moreau

    @staticmethod
    def moreau_W_nosum(mu):
        def moreau(u):
            return (torch.abs((Energies.prox_W(u,mu)-0.5)**2-0.25)) + (1/(2*mu))*((u-Energies.prox_W(u,mu))**2)
        return moreau

    #####################################################################################
    # Moreau Envelope and its Gradient of alpha * ||u-f||_1
    #####################################################################################

    @staticmethod
    def gradient_l1_alpha(alpha, mu):
        def prox(u, mu, f):
            u = u - f
            d = u.abs() > mu
            p = u - u.sign() * mu
            return p * d + f
        def gradient(u, f):
            return 1/mu * (u - prox(u, mu*alpha, f))
        return gradient

    @staticmethod
    def moreau_l1_alpha(alpha, mu):
        def moreau(u, f):
            e_mu = (u - f).abs()
            d = (e_mu > (mu*alpha))
            return alpha*((e_mu - (mu*alpha) / 2) * d + (e_mu ** 2) * (~d) / (2 * (mu*alpha))).sum()
        return moreau

    @staticmethod
    def energy_l1_alpha(alpha, mu=None):
        def energy(u, f):
            return alpha*torch.sum(torch.abs(u-f))
        return energy

    @staticmethod
    def energy_l1_alpha(alpha, mu=None):
        def energy(u, f):
            return alpha*torch.sum(torch.abs(u-f))
        return energy

    @staticmethod
    def moreau_l1_alpha_withoutsum(alpha, mu):
        def moreau(u, f):
            e_mu = (u - f).abs()
            d = (e_mu > (mu*alpha))
            return alpha*((e_mu - (mu*alpha) / 2) * d + (e_mu ** 2) * (~d) / (2 * (mu*alpha)))
        return moreau

    #####################################################################################
    # Energy and Gradient of alpha * ||u-f||_2
    #####################################################################################

    @staticmethod
    def energy_l2_alpha(alpha):
        def energy(u, f):
            return 0.5*alpha*torch.sum((u-f)**2)
        return energy

    @staticmethod
    def energy_l2_alpha_withoutsum(alpha):
        def energy(u, f):
            return 0.5*alpha*((u-f)**2)
        return energy

    @staticmethod
    def gradient_l2_alpha(alpha):
        def gradient(u, f):
            return alpha*(u - f)
        return gradient