import numpy as np
import utils
import config
from GibbsSampler import GibbsSampler, NormalSampler, GRWMH



class RescaleNormalSampler(NormalSampler):
    def sample_normal(self, var_cls):
        sigma = 1/(1/var_cls + self.beam**2/self.noise)
        weiner_map = sigma*self.beam*self.pix_map/self.noise
        fluctuations_map = np.random.normal(size = len(sigma))*np.sqrt(sigma)
        new_map = weiner_map + fluctuations_map
        return weiner_map, fluctuations_map, new_map


class RescaleMHSampler(GRWMH):
    def __init__(self, nside, lmax, proposal_variance, pix_map, beam, noise, n_iter = 1, dimension=5):
        super().__init__(nside, lmax, proposal_variance, n_iter = n_iter, dimension=dimension)
        self.pix_map = pix_map
        self.beam = beam
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])


    def compute_weiner_filter_map(self, var_cls):
        sigma = 1 / (1 / var_cls + self.beam ** 2 / self.noise)
        weiner_map = sigma * self.beam * self.pix_map / self.noise
        return weiner_map

    def compute_log_likelihood(self, var_cls, weiner_map, fluctuations_map):
        return -(1/2)*np.sum((self.pix_map -  self.beam*weiner_map)**2/self.noise) - (1/2)*np.sum((weiner_map**2/var_cls))\
                - (1/2)*np.sum(fluctuations_map*self.beam*(1/self.noise)*self.beam*fluctuations_map)

    def compute_log_MH_ratio(self, old_params, old_var_cls, old_weiner_map, old_fluctuations_map ,new_params, new_var_cls,
                             new_weiner_map, new_fluctuations_map):
        num = self.compute_log_likelihood(new_var_cls, new_weiner_map, new_fluctuations_map) + self.compute_log_prior(new_params)
        denom = self.compute_log_likelihood(old_var_cls, old_weiner_map, old_fluctuations_map) + self.compute_log_prior(old_params)
        return num - denom

    def run_step(self, old_params, old_var_cls, old_weiner_filter_map, old_fluctuations_map):
        accept = 0
        new_params, new_cls, new_var_cls = self.propose_new_params(old_params)
        new_weiner_filter_map = self.compute_weiner_filter_map(new_var_cls)
        new_fluctuations_map = np.sqrt(new_var_cls/old_var_cls)* old_fluctuations_map
        log_r = self.compute_log_MH_ratio(old_params, old_var_cls, old_weiner_filter_map, old_fluctuations_map ,new_params, new_var_cls,
                             new_weiner_filter_map, new_fluctuations_map)

        if np.log(np.random.uniform()) < log_r:
            old_params = new_params
            old_var_cls = new_var_cls
            old_weiner_filter_map = new_weiner_filter_map
            old_fluctuations_map = new_fluctuations_map
            accept = 1

        return old_params, old_var_cls, old_weiner_filter_map, old_fluctuations_map, accept

    def run(self, old_params, old_var_cls, old_weiner_filter_map, old_fluctuations_map):
        acceptions = []
        for i in range(self.n_iter):
            old_params, old_var_cls, old_weiner_filter_map, old_fluctuations_map, accept = self.run_step(old_params,
                                                            old_var_cls, old_weiner_filter_map, old_fluctuations_map)
            acceptions.append(accept)

        return old_params, old_var_cls, old_weiner_filter_map, old_fluctuations_map, acceptions


class Rescale(GibbsSampler):
    def __init__(self, nside, lmax, noise, beam, proposal_variance, pix_map, n_iter=100, pix_weight=4*np.pi/config.Npix, dimension=5):
        super().__init__(nside, lmax, noise, beam, proposal_variance, pix_map, n_iter, pix_weight)
        self.grwmh_sampler = RescaleMHSampler(nside, lmax, proposal_variance, pix_map, beam, noise,  n_iter=1, dimension=dimension)
        self.normal_sampler = RescaleNormalSampler(nside, lmax, noise, beam, pix_weight, pix_map)
        self.name = "Rescale MH sampler"

    def run(self, theta_init):
        history_theta = []
        h_acceptions = []
        theta = theta_init
        cls = utils.generate_cls(theta_init)
        var_cls = np.concatenate([np.array([cl if i == 0 else cl / 2 for i in range(2 * l + 1)])
                                  for l, cl in enumerate(cls, start=2)])

        for i in range(self.n_iter):
            if i % 100 == 0:
                print("Rescale")
                print(i)

            weiner_map, fluctuations_map, new_map = self.normal_sampler.sample_normal(var_cls)
            theta, var_cls, weiner_map, fluctuations_map, acceptions = self.grwmh_sampler.run(theta, var_cls, weiner_map,
                                                                                              fluctuations_map)

            history_theta.append(theta)
            h_acceptions.append(acceptions)

        print("Acceptance rate:")
        print(np.mean(h_acceptions))

        return np.array(history_theta), np.array(h_acceptions)

