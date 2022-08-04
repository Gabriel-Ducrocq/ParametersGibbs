import numpy as np
import utils
import config


class GRWMH():

    def __init__(self, nside, lmax, proposal_variance, n_iter = 1, dimension=5):
        self.nside = nside
        self.lmax = lmax
        self.n_iter = n_iter
        self.dimension = dimension
        print(proposal_variance)
        if len(proposal_variance.shape) == 1:
            self.proposal_variance = np.sqrt(proposal_variance)
        else:
            self.proposal_variance = np.linalg.cholesky(proposal_variance)

        self.metropolis_within_gibbs = True

    def propose_new_params(self, old_params):
        if len(self.proposal_variance.shape) == 1:
            new_params = np.random.normal(loc=old_params, scale=self.proposal_variance)
        else:
            new_params = old_params + np.dot(self.proposal_variance,np.random.normal(loc=0, scale=1, size=self.dimension))

        new_cls = utils.generate_cls(new_params)
        new_full_var_cls = utils.generate_var_cls(new_cls)
        return new_params, new_cls, new_full_var_cls

    def compute_log_prior(self, params):
        ###Flat prior assumed !
        return 0
        lower_bound = config.COSMO_PARAMS_MEAN - 10*config.COSMO_PARAMS_SIGMA
        upper_bound = config.COSMO_PARAMS_MEAN + 10*config.COSMO_PARAMS_SIGMA

        if np.all(params >=lower_bound) and np.all(params <= upper_bound):
            return np.sum(np.log(1 / (upper_bound - lower_bound)))

        else:
            return -np.inf

    def compute_log_likelihood(self, var_cls, alm_map):
        return None

    def compute_log_MH_ratio(self, old_params, old_var_cls, new_params, new_var_cls, alm_map):
        num = self.compute_log_likelihood(new_var_cls, alm_map) + self.compute_log_prior(new_params)
        denom = self.compute_log_likelihood(old_var_cls, alm_map) + self.compute_log_prior(old_params)
        return num - denom

    def run_step(self, old_params, old_var_cls, alm_map):
        accept = 0
        new_params, new_cls, new_var_cls = self.propose_new_params(old_params)
        log_r = self.compute_log_MH_ratio(old_params, old_var_cls, new_params, new_var_cls, alm_map)

        if np.log(np.random.uniform()) < log_r:
            old_params = new_params
            old_var_cls = new_var_cls
            accept = 1

        return old_params, old_var_cls, accept

    def run(self, init_params, init_var_cls,  alm_map):
        acceptions = []
        old_params = init_params
        old_var_cls = init_var_cls
        for i in range(self.n_iter):

            old_params, old_var_cls, accept = self.run_step(old_params, old_var_cls, alm_map)
            acceptions.append(accept)

        return old_params, old_var_cls, acceptions


class NormalSampler():

    def __init__(self, nside, lmax, noise, beam, pix_weight, pix_map):
        self.nside = nside
        self.lmax = lmax
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        self.beam = beam
        self.pix_weight = pix_weight
        self.pix_map = pix_map

    def sample_normal(self, var_cls):
        return None


class GibbsSampler():

    def __init__(self, nside, lmax, noise, beam, proposal_variance, pix_map, n_iter, pix_weight):
        self.nside = nside
        self.lmax = lmax
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        self.grwmh_sampler = None
        self.beam = beam
        self.n_iter = n_iter
        self.normal_sampler = None
        self.proposal_variance = proposal_variance
        self.pix_map = pix_map
        self.pix_weight = pix_weight
        self.name = None
        self.use_crank_nicolson = False
        self.cranknicolson_sampler = None


    def run(self, theta_init):
        history_theta = []
        h_acceptions = []
        h_acceptions_cranknicolson = []
        theta = theta_init
        cls = utils.generate_cls(theta_init)
        var_cls = np.concatenate([np.array([cl if i == 0 else cl / 2 for i in range(2 * l + 1)])
                                  for l, cl in enumerate(cls, start=2)])

        if self.use_crank_nicolson:
            alm_map = self.normal_sampler.sample_normal(var_cls)

        for i in range(self.n_iter):
            if i % 10 == 0:
                print(self.name)
                print(i)

            history_theta.append(theta)
            if self.use_crank_nicolson:
                alm_map, all_acceptions = self.cranknicolson_sampler.run(var_cls, alm_map)
                h_acceptions_cranknicolson.append(all_acceptions)
            else:
                alm_map = self.normal_sampler.sample_normal(var_cls)

            theta, var_cls, acceptions = self.grwmh_sampler.run(theta, var_cls, alm_map)
            h_acceptions.append(acceptions)


        h_acceptions = np.array(h_acceptions)
        print("Acception rate MH:")
        print(np.mean(h_acceptions))
        if self.use_crank_nicolson:
            print("Acception rate Crank Nicolson:")
            print(np.mean(h_acceptions_cranknicolson))

        return np.array(history_theta), h_acceptions



