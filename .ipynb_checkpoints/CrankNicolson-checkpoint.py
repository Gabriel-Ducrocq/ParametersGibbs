import numpy as np




class CrankNicolson():

    def __init__(self, nside, lmax, noise, beam, beta, obs_map, n_iter=1, within_gibbs=True):
        self.nside = nside
        self.lmax = lmax
        self.n_iter = n_iter
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        self.beam = beam
        self.beta = beta
        self.metropolis_within_gibbs = within_gibbs
        self.obs_map = obs_map

    def propose_alm_map(self, old_map, var_cls):
        return np.sqrt(1 - self.beta**2)*old_map + self.beta*np.random.normal(size=len(var_cls))*np.sqrt(var_cls)

    def compute_log_likelihood(self, alm_map):
        return -(1/2)*np.sum(((self.obs_map - self.beam*alm_map)**2)/self.noise)

    def compute_log_ratio(self, old_alm_map, new_alm_map):
        return self.compute_log_likelihood(new_alm_map) - self.compute_log_likelihood(old_alm_map)

    def run_step(self, var_cls, old_map):
        acception = 0
        new_map = self.propose_alm_map(old_map, var_cls)
        log_r = self.compute_log_ratio(old_map, new_map)

        if np.log(np.random.uniform()) < log_r:
            old_map = new_map
            acception = 1

        return old_map, acception

    def run(self, var_cls, init_map):
        old_map = init_map
        h_acceptions = []

        for i in range(self.n_iter):
            old_map, accept = self.run_step(var_cls, old_map)
            h_acceptions.append(accept)

        return old_map, h_acceptions

