import config
import utils
import time
import healpy as hp
import numpy as np
from GibbsSampler import GibbsSampler
from CrankNicolson import CrankNicolson
from copy import deepcopy
import matplotlib.pyplot as plt

if __name__ == '__main__':
    theta_true = config.COSMO_PARAMS_MEAN # + config.COSMO_PARAMS_SIGMA
    cls_TT_true, cls_EE_true, cls_BB_true, cls_TE_true = utils.generate_cls(theta_true)
    import matplotlib.pyplot as plt

    snr = cls_TT_true[2:] * (config.bl_gauss ** 2) / (config.noise_covar_I* 4 * np.pi / config.Npix)
    plt.plot(snr)
    plt.show()

    snr = cls_EE_true[2:] * (config.bl_gauss ** 2) / (config.noise_covar_Q* 4 * np.pi / config.Npix)
    plt.plot(snr)
    plt.show()
    #plt.plot(cls_TT_true[2:]*config.bl_gauss**2/(4*np.pi/config.Npix)**2)
    #plt.show()
    #plt.plot(cls_EE_true[2:]*config.bl_gauss**2/(4*np.pi/config.Npix)**2)
    #plt.show()
    #plt.plot(cls_BB_true[2:]*config.bl_gauss**2/(4*np.pi/config.Npix)**2)
    #plt.show()
    #plt.show()
    #plt.plot(cls_TE_true)
    #plt.show()
    pixel_map = {"I":None, "Q":None, "U":None}
    pixel_map["I"], pixel_map["Q"], pixel_map["U"] =hp.synfast([cls_TT_true, cls_EE_true, cls_BB_true, cls_TE_true],
                                                               pol=True, new=True, lmax=config.L_MAX_SCALARS,
               nside=config.NSIDE)

    data = {"theta_true":theta_true, "noise":config.noise, "nside":config.NSIDE, "beam_fwhm":config.fwhm_radians,
            "lmax":config.L_MAX_SCALARS, "I":pixel_map["I"], "Q":pixel_map["Q"], "U":pixel_map["U"]}

    np.save("data_NSIDE_4.npy", data)

    centered_gibbs = GibbsSampler(config.NSIDE, config.L_MAX_SCALARS, config.noise_covar_I, config.noise_covar_Q,
                                  config.fwhm_radians, config.proposal_variance
                                   ,pixel_map, n_iter=100, n_iter_grwmh=1, gibbs_cr=False, mask_path=None)

    #crankN = CrankNicolson(config.NSIDE, config.L_MAX_SCALARS, config.noise_covar_I,
    #                       config.noise_covar_Q, config.fwhm_radians, pixel_map, mask_path = None, pcg_accuracy=1e-6,
    #             gibbs_cr = False, overrelax=False, beta = 0.2)


    res = centered_gibbs.run(config.COSMO_PARAMS_MEAN)

    """
    theta_init = config.COSMO_PARAMS_MEAN
    alms_T, alms_E, alms_B = hp.synalm([cls_TT_true, cls_EE_true, cls_BB_true, cls_TE_true], lmax=config.L_MAX_SCALARS, new=True)
    alms = np.zeros((len(alms_T), 3), dtype=complex)
    alms[:, 0]= alms_T
    alms[:, 1]= alms_E
    alms[:, 2]= alms_B

    all_cls = np.zeros((len(cls_TT_true), 3, 3))
    all_cls[:, 0, 0] = cls_TT_true
    all_cls[:, 1, 1] = cls_EE_true
    all_cls[:, 2, 2] = cls_BB_true
    all_cls[:, 1, 0] = all_cls[:, 0, 1] = cls_TE_true

    inv_cls, _ = utils.compute_inverse_and_cholesky_constraint_realization(all_cls)
    start = time.time()
    _, _, _, _, h_theta = centered_gibbs.grwmh_sampler.run(theta_init, all_cls, inv_cls, alms)
    end = time.time()
    print("Total time:", end - start)
    #h_theta, _ = centered_gibbs.run(theta_true)


    h_theta = np.array(h_theta)

    for i in range(len(h_theta)):
        plt.hist(h_theta[:, i], density=True)
        plt.axvline(theta_true[i])
        plt.show()
        plt.close()
    """

