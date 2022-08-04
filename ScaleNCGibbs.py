import numpy as np
import config
from CrankNicolson import CrankNicolson
from GibbsSampler import NormalSampler, GRWMH, GibbsSampler
import utils



class ScaleNCNormalSampler(NormalSampler):

    def sample_normal(self, var_cls, scale_param):
        sigma = 1/(np.sqrt(np.exp(scale_param)*10**(-10))*self.beam*(1/self.noise)*self.beam*np.sqrt(np.exp(scale_param)*10**(-10))+ 1/var_cls)
        mu = sigma*np.sqrt(np.exp(scale_param)*10**(-10))*self.beam*(1/self.noise)*self.pix_map
        return mu + np.random.normal(size=len(mu))*np.sqrt(sigma)


class ScaleNCGRWMH(GRWMH):
    def __init__(self, nside, lmax, proposal_variance, pix_map, beam, noise, n_iter = 1, dimension=6):
        super().__init__(nside, lmax, proposal_variance, n_iter = n_iter, dimension=dimension)
        self.pix_map = pix_map
        self.beam = beam
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])


    def generate_unscaled_cls(self, params):
        scale_param = params[-1]
        unscaled_cls = utils.generate_cls(params)/(np.exp(scale_param)*10**(-10))
        return unscaled_cls

    def propose_new_params(self, old_params):
        if len(self.proposal_variance.shape) == 1:
            new_params = np.random.normal(loc=old_params, scale=self.proposal_variance)
        else:
            new_params = old_params + np.dot(self.proposal_variance,np.random.normal(loc=0, scale=1, size=self.dimension))

        new_cls = self.generate_unscaled_cls(new_params)
        new_full_var_cls = utils.generate_var_cls(new_cls)
        return new_params, new_cls, new_full_var_cls

    def compute_log_likelihood(self, var_cls, scale_param, alm_map):
        return -(1/2)*np.sum((self.pix_map - self.beam*np.sqrt(np.exp(scale_param)*10**(-10))*alm_map)**2/self.noise) - (1/2)*np.sum(alm_map**2/var_cls) \
                - (1/2)*np.sum(np.log(var_cls))

    def compute_log_MH_ratio(self, old_params, old_var_cls, new_params, new_var_cls, alm_map):
        old_scale = old_params[-1]
        new_scale = new_params[-1]
        num = self.compute_log_likelihood(new_var_cls, new_scale, alm_map) + self.compute_log_prior(new_params)
        denom = self.compute_log_likelihood(old_var_cls, old_scale,  alm_map) + self.compute_log_prior(old_params)
        return num - denom


class ScaleNCGibbs(GibbsSampler):

    def __init__(self, nside, lmax, noise, beam, proposal_variance, pix_map, n_iter=100, pix_weight=4*np.pi/config.Npix,
                 dimension=5, n_iter_grwmh = 1, crank_nicolson = False, cn_beta = 0.5, cn_iter = 1):
        super().__init__(nside, lmax, noise, beam, proposal_variance, pix_map, n_iter, pix_weight)
        self.grwmh_sampler = ScaleNCGRWMH(nside, lmax, proposal_variance, pix_map, beam, noise, n_iter=n_iter_grwmh, dimension=dimension)
        self.normal_sampler = ScaleNCNormalSampler(nside, lmax, noise, beam, pix_weight, pix_map)
        self.name = "Centered Gibbs"
        self.use_crank_nicolson = crank_nicolson
        self.cranknicolson_sampler = CrankNicolson(nside, lmax, noise, beam, cn_beta, pix_map, n_iter=cn_iter)


    def run(self, theta_init):
        history_theta = []
        h_acceptions = []
        h_acceptions_cranknicolson = []
        theta = theta_init
        cls = self.grwmh_sampler.generate_unscaled_cls(theta_init)
        var_cls = np.concatenate([np.array([cl if i == 0 else cl / 2 for i in range(2 * l + 1)])
                                  for l, cl in enumerate(cls, start=2)])

        if self.use_crank_nicolson:
            alm_map = self.normal_sampler.sample_normal(var_cls, theta[-1])

        for i in range(self.n_iter):
            if i % 100 == 0:
                print(self.name)
                print(i)

            history_theta.append(theta)
            if self.use_crank_nicolson:
                alm_map, all_acceptions = self.cranknicolson_sampler.run(var_cls, alm_map)
                h_acceptions_cranknicolson.append(all_acceptions)
            else:
                alm_map = self.normal_sampler.sample_normal(var_cls, theta[-1])

            theta, var_cls, acceptions = self.grwmh_sampler.run(theta, var_cls, alm_map)
            h_acceptions.append(acceptions)


        h_acceptions = np.array(h_acceptions)
        print("Acception rate MH:")
        print(np.mean(h_acceptions))
        if self.use_crank_nicolson:
            print("Acception rate Crank Nicolson:")
            print(np.mean(h_acceptions_cranknicolson))

        return np.array(history_theta), h_acceptions
