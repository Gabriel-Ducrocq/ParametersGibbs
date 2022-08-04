from GibbsSampler import NormalSampler, GRWMH, GibbsSampler
import numpy as np
import config
import utils
from CenteredGibbs import CenteredGRWMH, CenteredNormalSampler
from NonCenteredGibbs import NonCenteredGRWMH


class Interweaving():
    def __init__(self, nside, lmax, noise, beam, proposal_variance_centered, proposal_variance_non_centered,
                 pix_map, n_iter, pix_weight = 4*np.pi/config.Npix, dimension=5, n_iter_grwmh_nc = 1, n_iter_grwmh_centered=1):
        self.nside = nside
        self.lmax = lmax
        self.noise = np.array([noise if i == 0 else noise / 2 for l in range(2, lmax + 1) for i in range(2 * l + 1)])
        self.grwmh_centered_sampler = CenteredGRWMH(nside, lmax, proposal_variance_centered, n_iter=n_iter_grwmh_centered, dimension=dimension)
        self.grwmh_non_centered_sampler = NonCenteredGRWMH(nside, lmax, proposal_variance_non_centered, pix_map, beam, noise, n_iter = n_iter_grwmh_nc, dimension=dimension)
        self.n_iter = n_iter
        self.normal_sampler = CenteredNormalSampler(nside, lmax, noise, beam, pix_weight, pix_map)


    def run(self, theta_init):
        h_theta = []
        acceptions_intermediate = []
        acceptions = []
        theta_old = theta_init

        h_theta.append(theta_old)
        cls_old = utils.generate_cls(theta_old)
        var_cls_old = utils.generate_var_cls(cls_old)

        for i in range(self.n_iter):
            if i % 10 == 0:
                print("Interweaving")
                print(i)

            s_centered = self.normal_sampler.sample_normal(var_cls_old)
            intermediate_theta, intermediate_var_cls, intermediate_acception = self.grwmh_centered_sampler.run(
                theta_old, var_cls_old, s_centered)

            s_non_centered = s_centered/np.sqrt(intermediate_var_cls)

            theta_old, var_cls_old, accepts = self.grwmh_non_centered_sampler.run(intermediate_theta,
                                                                                     intermediate_var_cls, s_non_centered)

            h_theta.append(theta_old)
            acceptions_intermediate.append(intermediate_acception)
            acceptions.append(accepts)


        print("Intermediate acceptance rate:")
        print(np.mean(acceptions_intermediate))
        print("Acceptance rate non centered:")
        print(np.mean(acceptions))

        return np.array(h_theta), acceptions_intermediate, acceptions