from GibbsSampler import NormalSampler, GRWMH, GibbsSampler
import numpy as np
import config


class CenteredNormalSampler(NormalSampler):

    def sample_normal(self, var_cls):
        sigma = 1/(1/var_cls + self.beam**2/self.noise)
        mu = sigma*self.beam*self.pix_map/self.noise
        return np.random.normal(mu, np.sqrt(sigma))


class CenteredGRWMH(GRWMH):

    def compute_log_likelihood(self, var_cls, alm_map):
        return -(1/2)*np.sum(((alm_map)**2)/var_cls) - (1/2)*np.sum(np.log(var_cls))


class CenteredGibbs(GibbsSampler):

    def __init__(self, nside, lmax, noise, beam, proposal_variance, pix_map, n_iter=100, pix_weight=4*np.pi/config.Npix, dimension=5, n_iter_grwmh = 1):
        super().__init__(nside, lmax, noise, beam, proposal_variance, pix_map, n_iter, pix_weight)
        self.grwmh_sampler = CenteredGRWMH(nside, lmax, proposal_variance, n_iter=n_iter_grwmh, dimension=dimension)
        self.normal_sampler = CenteredNormalSampler(nside, lmax, noise, beam, pix_weight, pix_map)
        self.name = "Centered Gibbs"