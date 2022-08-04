from GibbsSampler import GRWMH
import numpy as np


class DirectSampler(GRWMH):

    def __init__(self, nside, lmax, proposal_variance, noise, beam, n_iter = 1, dimension=5):
        super().__init__(nside, lmax, proposal_variance, n_iter, dimension=dimension)
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        self.beam = beam
        self.metropolis_within_gibbs = False

    def compute_log_likelihood(self, var_cls, alm_map):
        return -(1/2)*np.sum((alm_map**2)/(self.beam**2*var_cls+self.noise)) - (1/2)*np.sum(np.log(self.beam**2*var_cls+self.noise))

    def run(self, init_params, init_var_cls,  alm_map):
        acceptions = []
        history_params = []
        old_params = init_params
        history_params.append(old_params)
        old_var_cls = init_var_cls
        for i in range(self.n_iter):
            if not self.metropolis_within_gibbs:
                if i% 10 == 0:
                    print("Direct sampler")
                    print(i)

            old_params, old_var_cls, accept = self.run_step(old_params, old_var_cls, alm_map)
            acceptions.append(accept)
            history_params.append(old_params)

        print("Acceptance rate:")
        print(np.mean(acceptions))
        return np.array(history_params), old_var_cls, acceptions


