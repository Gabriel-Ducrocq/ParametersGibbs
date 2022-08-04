from GibbsSampler import NormalSampler, GRWMH, GibbsSampler
import numpy as np
import config
import utils



class PartiallyCenteredNormalSampler(NormalSampler):
    def sample_normal(self, variance_low_l, variance_high_l):
        sigma = 1/(np.sqrt(variance_high_l)*self.beam*(1/self.noise)*self.beam*np.sqrt(variance_high_l) + 1/variance_low_l)
        mu = sigma*np.sqrt(variance_high_l)*self.beam*(1/self.noise)*self.pix_map
        return mu + np.random.normal(loc=0, scale=1,size =len(mu))*np.sqrt(sigma)

class PartiallyCenteredGRWMH(GRWMH):
    def __init__(self, nside, lmax, proposal_variance, pix_map, beam, noise, n_iter = 1, dimension=6, l_cut = config.l_cut):
        super().__init__(nside, lmax, proposal_variance, n_iter = n_iter, dimension=dimension)
        self.pix_map = pix_map
        self.beam = beam
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        self.l_cut = l_cut

    def compute_log_likelihood(self, variance_high_l, variance_low_l, alm_map):
        return -(1/2)*np.sum(((self.pix_map - self.beam*np.sqrt(variance_high_l)*alm_map)**2)/self.noise)\
               - (1/2)*np.sum(alm_map**2/variance_low_l) - (1/2)*np.sum(np.log(variance_low_l))

    def compute_log_MH_ratio(self, old_params, old_var_low_l, old_var_high_l, new_params, new_var_low_l, new_var_high_l
                             , alm_map):
        num = self.compute_log_likelihood(new_var_high_l, new_var_low_l, alm_map) + self.compute_log_prior(new_params)
        denom = self.compute_log_likelihood(old_var_high_l, old_var_low_l, alm_map) + self.compute_log_prior(old_params)
        return num - denom

    def run_step(self, old_params, old_var_low_l, old_var_high_l, alm_map):
        accept = 0
        new_params, new_cls, new_var_cls = self.propose_new_params(old_params)
        new_var_low_l = np.ones(len(new_var_cls))
        new_var_high_l = np.ones(len(new_var_cls))
        new_var_low_l[:self.l_cut-4] *= new_var_cls[:self.l_cut-4]
        new_var_high_l[self.l_cut-4:] *= new_var_cls[self.l_cut-4:]
        log_r = self.compute_log_MH_ratio(old_params, old_var_low_l, old_var_high_l, new_params, new_var_low_l,
                                          new_var_high_l, alm_map)

        if np.log(np.random.uniform()) < log_r:
            old_params = new_params
            old_var_low_l = new_var_low_l
            old_var_high_l = new_var_high_l
            accept = 1

        return old_params, old_var_low_l, old_var_high_l, accept

    def run(self, init_params, old_var_low_l, old_var_high_l,  alm_map):
        acceptions = []
        old_params = init_params
        for i in range(self.n_iter):
            old_params, old_var_low_l, old_var_high_l, accept = self.run_step(old_params, old_var_low_l, old_var_high_l,
                                                                              alm_map)
            acceptions.append(accept)

        return old_params, old_var_low_l, old_var_high_l, acceptions



class PNCPGibbsSampler(GibbsSampler):
    def __init__(self, nside, lmax, noise, beam, proposal_variance, pix_map, n_iter=100, pix_weight=4*np.pi/config.Npix,
                 dimension=5, l_cut = config.l_cut):
        super().__init__(nside, lmax, noise, beam, proposal_variance, pix_map, n_iter, pix_weight)
        self.grwmh_sampler = PartiallyCenteredGRWMH(nside, lmax, proposal_variance, pix_map, beam, noise, n_iter=1,
                                                    dimension=dimension, l_cut = l_cut)
        self.normal_sampler = PartiallyCenteredNormalSampler(nside, lmax, noise, beam, pix_weight, pix_map)
        self.l_cut = l_cut
        self.name = "Partially Non Centered Gibbs"

    def run(self, theta_init):
        history_theta = []
        h_acceptions = []
        theta = theta_init
        cls = utils.generate_cls(theta_init)
        var_cls = np.concatenate([np.array([cl if i == 0 else cl / 2 for i in range(2 * l + 1)])
                                  for l, cl in enumerate(cls, start=2)])

        var_low_l = np.ones(len(var_cls))
        var_high_l = np.ones(len(var_cls))
        var_low_l[:self.l_cut-4] *= var_cls[:self.l_cut-4]
        var_high_l[self.l_cut-4:] *= var_cls[self.l_cut-4:]
        for i in range(self.n_iter):
            if i % 100 == 0:
                print(self.name)
                print(i)
            
            history_theta.append(theta)
            alm_map = self.normal_sampler.sample_normal(var_low_l, var_high_l)
            theta, var_low_l, var_high_l, acceptions = self.grwmh_sampler.run(theta, var_low_l, var_high_l, alm_map)
            h_acceptions.append(acceptions)


        h_acceptions = np.array(h_acceptions)
        print("Acception rate")
        print(np.mean(h_acceptions))

        return np.array(history_theta), h_acceptions
