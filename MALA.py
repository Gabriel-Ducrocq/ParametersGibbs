from GibbsSampler import GRWMH
import numpy as np

class MALA():
    def __init__(self, nside, lmax, noise, beam, beta, obs_map, n_iter=1, within_gibbs=True):
        self.nside = nside
        self.lmax = lmax
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        self.beam = beam
        self.pix_map = obs_map
        self.n_iter = n_iter
        self.beta = beta
        self.within_gibbs = within_gibbs

    def compute_log_gradient(self, alm_map, var_cls):
        sigma = 1/(1/var_cls + self.beam**2/self.noise)
        mu = sigma*self.beam*self.pix_map/self.noise
        return -(1/sigma)*alm_map + (1/sigma)*mu

    def propose_new_map(self, old_map, var_cls):
        sigma = 1/(1/var_cls + self.beam**2/self.noise)
        gradient = self.compute_log_gradient(old_map, var_cls)
        new_map = old_map + self.beta*sigma*gradient + np.sqrt(2*self.beta*sigma)*np.random.normal(size=len(old_map))
        return new_map

    def compute_log_proposal(self, alm_old, alm_new, var_cls):
        sigma = 1 / (1 / var_cls + self.beam ** 2 / self.noise)
        mu = alm_old + self.beta*sigma*self.compute_log_gradient(alm_old, var_cls)
        sigma = 2*self.beta*sigma

        return -(1/2)*np.sum((alm_new - mu)**2/sigma)

    def compute_log_density(self, alm_map, var_cls):
        sigma = 1/(1/var_cls + self.beam**2/self.noise)
        mu = sigma*self.beam*self.pix_map/self.noise
        return -(1/2)*np.sum((alm_map - mu)**2/sigma)

    def compute_log_ratio(self, alm_old, alm_new, var_cls):
        num = self.compute_log_density(alm_new, var_cls) + self.compute_log_proposal(alm_new, alm_old, var_cls)
        denom = self.compute_log_density(alm_old, var_cls) + self.compute_log_proposal(alm_old, alm_new, var_cls)
        return num - denom

    def run_step(self, alm_old, var_cls):
        accept = 0
        alm_new = self.propose_new_map(alm_old, var_cls)
        log_r = self.compute_log_ratio(alm_old, alm_new, var_cls)
        if np.log(np.random.uniform()) < log_r:
            alm_old = alm_new
            accept = 1

        return alm_old, accept

    def run(self, alm, var_cls):
        h_accept = []
        h_map = []
        for i in range(self.n_iter):
            alm, accept = self.run_step(alm, var_cls)
            h_accept.append(accept)
            if not self.within_gibbs:
                if i % 100000 == 0:
                    print("MALA")
                    print(i)

                h_map.append(alm[10])

        if not self.within_gibbs:
            h_accept = np.array(h_accept)
            print("Acceptance rate MALA:")
            print(np.mean(h_accept))
            return alm,np.array(h_accept), np.array(h_map)

        return alm, h_accept





