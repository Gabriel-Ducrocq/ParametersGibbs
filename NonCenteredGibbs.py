from GibbsSampler import NormalSampler, GRWMH, GibbsSampler
import numpy as np
import config


class NonCenteredNormalSampler(NormalSampler):
    def sample_normal(self, var_cls):
        sigma = 1/(np.sqrt(var_cls)*self.beam*(1/self.noise)*self.beam*np.sqrt(var_cls) + 1)
        mu = sigma*np.sqrt(var_cls)*self.beam*(1/self.noise)*self.pix_map
        return np.random.normal(mu, np.sqrt(sigma))



class NonCenteredGRWMH(GRWMH):
    def __init__(self, nside, lmax, proposal_variance, pix_map, beam, noise, n_iter = 1, dimension=6):
        super().__init__(nside, lmax, proposal_variance, n_iter = n_iter, dimension=dimension)
        self.pix_map = pix_map
        self.beam = beam
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])

    def compute_log_likelihood(self, var_cls, alm_map):
        return -(1/2)*np.sum((self.pix_map - self.beam*np.sqrt(var_cls)*alm_map)**2/self.noise)


class NonCenteredGibbsSampler(GibbsSampler):
    def __init__(self, nside, lmax, noise, beam, proposal_variance, pix_map, n_iter=100, pix_weight=4*np.pi/config.Npix, dimension=5):
        super().__init__(nside, lmax, noise, beam, proposal_variance, pix_map, n_iter, pix_weight)
        self.grwmh_sampler = NonCenteredGRWMH(nside, lmax, proposal_variance, pix_map, beam, noise,  n_iter=1, dimension=dimension)
        self.normal_sampler = NonCenteredNormalSampler(nside, lmax, noise, beam, pix_weight, pix_map)
        self.name = "Non centered Gibbs"



