import healpy as hp
import numpy as np
import config
import utils
import copy
import healpy as hp
import matplotlib.pyplot as plt


class GRWMH():
    def __init__(self, l_max, map, proposal_variance, prio_mean, prior_stdd, dim = 6, n_iterations=100):
        self.lmax = l_max
        self.map = map
        self.dim = dim
        self.prior_mean = prio_mean
        self.prior_stdd = prior_stdd
        self.n_iterations = n_iterations
        if len(proposal_variance.shape) == 1:
            self.proposal_variance = np.diag(proposal_variance)
            self.proposal_stdd = np.diag(np.sqrt(proposal_variance))
        else:
            self.proposal_variance = proposal_variance
            self.proposal_stdd = np.linalg.cholesky(proposal_variance)

    def propose_new_parameters(self, old_params):
        new_params = old_params + np.dot(self.proposal_stdd, np.random.normal(0, 1, size=self.dim))
        new_cls_TT, new_cls_EE, new_cls_BB, new_cls_TE = utils.generate_cls(new_params)
        new_variance = np.zeros((len(new_cls_TT), 3, 3))
        new_variance[:, 0, 0] = new_cls_TT
        new_variance[:, 1, 1] = new_cls_EE
        new_variance[:, 2, 2] = new_cls_BB
        new_variance[:, 1, 0] = new_cls_TE
        new_variance[:, 0, 1] = new_cls_TE

        new_precision, _ = utils.compute_inverse_and_cholesky_constraint_realization(new_variance)
        return {"params":new_params, "variance":new_variance, "precision":new_precision}

    def compute_log_prior(self, params):
        return -(1/2)*np.sum((params-self.prior_mean)**2/self.prior_stdd**2)

    def compute_log_likelihood(self, precision, variance):
        one_product = utils.matrix_product(precision, self.map)
        numerator = -(1/2)*np.sum(np.conj(self.map)*one_product)
        log_det = np.sum(np.log([np.linalg.det(cls) for cls in variance[2:]]))
        return numerator - (1/2)*log_det

    def compute_log_MH_ratio(self, old_data, new_data):
        log_lik_part = self.compute_log_likelihood(new_data["precision"], new_data["variance"]) \
        - self.compute_log_likelihood(old_data["precision"], old_data["variance"])

        log_prior_part = self.compute_log_prior(old_data["params"]) - self.compute_log_prior(old_data["params"])
        return log_lik_part + log_prior_part

    def run(self, init_params):
        acceptance = 0
        h_params = []
        new_cls_TT, new_cls_EE, new_cls_BB, new_cls_TE = utils.generate_cls(init_params)
        new_variance = np.zeros((len(new_cls_TT), 3, 3))
        new_variance[:, 0, 0] = new_cls_TT
        new_variance[:, 1, 1] = new_cls_EE
        new_variance[:, 2, 2] = new_cls_BB
        new_variance[:, 1, 0] = new_cls_TE
        new_variance[:, 0, 1] = new_cls_TE

        new_precision, _ = utils.compute_inverse_and_cholesky_constraint_realization(new_variance)
        data = {"params": init_params, "variance": new_variance, "precision": new_precision}

        for i in range(self.n_iterations):
            print(i)
            new_data = self.propose_new_parameters(data["params"])
            log_ratio = self.compute_log_MH_ratio(data, new_data)
            if np.log(np.random.uniform()) < log_ratio:
                data = copy.deepcopy(new_data)
                acceptance += 1

            h_params.append(data["params"])

        print("Acceptance ratio:", acceptance/self.n_iterations)
        return np.array(h_params)


if __name__ == '__main__':
    """
    theta_true = config.COSMO_PARAMS_MEAN # + config.COSMO_PARAMS_SIGMA
    cls_TT_true, cls_EE_true, cls_BB_true, cls_TE_true = utils.generate_cls(theta_true)
    alm_map_T, alm_map_E, alm_map_B = hp.synalm([cls_TT_true, cls_EE_true, cls_BB_true, cls_TE_true], new=True)

    data = np.zeros((len(alm_map_T), 3), dtype=np.complex128)
    data[:, 0] = alm_map_T
    data[:, 1] = alm_map_E
    data[:, 2] = alm_map_B

    np.save("data.npy", data)
    """
    data = np.load("data.npy")
    data = data.items()

    grwmh = GRWMH(config.L_MAX_SCALARS, data, config.proposal_variance, config.COSMO_PARAMS_MEAN, config.COSMO_PARAMS_SIGMA)
    result = grwmh.run(config.COSMO_PARAMS_MEAN)

    np.save("outcome.npy", result)

