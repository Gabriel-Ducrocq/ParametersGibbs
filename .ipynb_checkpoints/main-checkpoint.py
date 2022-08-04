from CenteredGibbs import CenteredGibbs
import config
import utils
import time
import numpy as np
import matplotlib.pyplot as plt
from MHSampler import DirectSampler
from PNCP import PNCPGibbsSampler
from Rescaling import Rescale
from NonCenteredGibbs import NonCenteredGibbsSampler
from Interweaving import Interweaving
from statsmodels.graphics.tsaplots import plot_acf


if __name__ == '__main__':
    theta_true = config.COSMO_PARAMS_MEAN + config.COSMO_PARAMS_SIGMA
    cls_true = utils.generate_cls(theta_true)
    var_cl_full = utils.generate_var_cls(cls_true)

    #alm_true = np.random.normal(scale=np.sqrt(var_cl_full))
    #pix_map =config.beam*alm_true + np.random.normal(scale=np.sqrt(config.noise_covar))

    #data = {"theta_true":theta_true, "cls_true":cls_true, "var_°cl_full":var_cl_full, "alm_true":alm_true, "pix_map":pix_map,
    #        "noise":config.noise, "nside":config.NSIDE, "beam_fwhm":config.fwhm_radians, "lmax":config.L_MAX_SCALARS}

    #np.save("data_NSIDE_512", data)

    data = np.load("data_NSIDE_512.npy", allow_pickle=True)
    data = data.item()
    pix_map = data["pix_map"]

    centered_gibbs = CenteredGibbs(config.NSIDE, config.L_MAX_SCALARS, config.noise, config.beam, config.proposal_variance_gibbs
                                   ,pix_map, n_iter=100, n_iter_grwmh=1, crank_nicolson=True, cn_beta=0.0005)

    #direct_sampler = DirectSampler(config.NSIDE, config.L_MAX_SCALARS, config.proposal_variance_direct, config.noise,
    #                               config.beam, n_iter=10000)

    #non_centered_gibbs = NonCenteredGibbsSampler(config.NSIDE, config.L_MAX_SCALARS, config.noise, config.beam,
    #                                             config.proposal_variance_gibbs_nc,pix_map, n_iter=10000)

    #interweaving = Interweaving(config.NSIDE, config.L_MAX_SCALARS, config.noise, config.beam,
    #                            config.proposal_variance_gibbs_asis, config.proposal_variance_gibbs_nc_asis,
    #             pix_map, n_iter=100, pix_weight = 4*np.pi/config.Npix)

    #pncp_sampler = PNCPGibbsSampler(config.NSIDE, config.L_MAX_SCALARS, config.noise, config.beam, config.proposal_variance_pncp_gibbs
    #                 , pix_map, n_iter=100000)

    #rescale_sampler = Rescale(config.NSIDE, config.L_MAX_SCALARS, config.noise, config.beam,
    #                          config.proposal_variance_rescale, pix_map, n_iter=10000)

    #plt.plot(cls_true*config.bl_gauss**2)
    #plt.axhline(y=config.noise)
    #plt.show()

    start = time.time()
    print("launching centered gibbs sampler")
    h_theta, acceptions = centered_gibbs.run(theta_true)
    end = time.time()
    print(end-start)
    data_centered = {"h_theta":h_theta, "acceptions":acceptions}
    np.save("data_centered_gibbs_nersc.npy", data_centered)


    #start = time.time()
    #print("launching direct sampler")
    #h_theta, _, acceptions = direct_sampler.run(theta_true, var_cl_full, pix_map)
    #end = time.time()
    #print(end-start)
    #data_direct = {"h_theta":h_theta, "acceptions":acceptions}
    #np.save("data_direct_tuned.npy", data_direct)

    #start = time.time()
    #print("launching non centered sampler")
    #h_theta, acceptions = non_centered_gibbs.run(theta_true)
    #end = time.time()
    #print(end-start)
    #data_non_centered = {"h_theta":h_theta, "acceptions":acceptions}
    #np.save("data_non_centered.npy", data_direct)

    #start = time.time()
    #print("launching interweaving sampler")
    #h_theta, accept_inter, accept = interweaving.run(theta_true)
    #end = time.time()
    #print(end-start)
    #data_asis = {"h_theta":h_theta, "acceptions_intermediate":accept_inter, "acceptions":accept}
    #np.save("data_asis_tuned.npy", data_asis)

    #start = time.time()
    #print("launching pncp sampler")
    #h_theta, accept = pncp_sampler.run(theta_true)
    #end = time.time()
    #print(end-start)
    #data_pncp = {"h_theta":h_theta, "acceptions":accept}
    #np.save("data_pncp_tuned.npy", data_pncp)


    #start = time.time()
    #print("launching rescale sampler")
    #h_theta, accept = rescale_sampler.run(theta_true)
    #end = time.time()
    #print(end-start)
    #data_rescale = {"h_theta":h_theta, "acceptions":accept}
    #np.save("data_rescale_tuned.npy", data_rescale)



    """
    data_centered_gibbs = np.load("data_centered_gibbs.npy", allow_pickle = True)
    data_centered_gibbs = data_centered_gibbs.item()
    h_theta_centered = data_centered_gibbs["h_theta"]

    data_direct = np.load("data_direct.npy", allow_pickle = True)
    data_direct = data_direct.item()
    h_theta_direct = data_direct["h_theta"]

    data_non_centered = np.load("data_non_centered_gibbs.npy", allow_pickle = True)
    data_non_centered = data_non_centered.item()
    h_theta_non_centered = data_non_centered["h_theta"]

    data_asis = np.load("data_asis.npy", allow_pickle = True)
    data_asis = data_asis.item()
    h_theta_asis = data_asis["h_theta"]



    lower_bound = config.COSMO_PARAMS_MEAN - 10 * config.COSMO_PARAMS_SIGMA
    upper_bound = config.COSMO_PARAMS_MEAN + 10 * config.COSMO_PARAMS_SIGMA
    for i in range(5):
        plt.plot(h_theta_direct[:, i], label="Direct")
        plt.plot(h_theta_centered[:, i], label="Centered Gibbs")
        plt.plot(h_theta_non_centered[:, i], label="NonCentered")
        plt.plot(h_theta_asis[:, i], label="ASIS")
        plt.axhline(y=theta_true[i])
        plt.legend(loc="upper right")
        plt.title(config.COSMO_PARAMS_NAMES[i])
        plt.show()
        plt.close()
        plt.hist(h_theta_direct[200:, i], label="Direct", alpha=0.5, density=True)
        plt.hist(h_theta_centered[200:, i], label="Centered Gibbs", alpha=0.5, density=True)
        plt.hist(h_theta_non_centered[200:, i], label="NonCentered Gibbs", alpha=0.5, density=True)
        plt.axvline(x=theta_true[i])
        plt.legend(loc="upper right")
        plt.show()
        plt.close()
        plot_acf(h_theta_direct[:, i], lags=100, title="Direct")
        plot_acf(h_theta_centered[:, i], lags=100, title="Centered Gibbs")
        plot_acf(h_theta_non_centered[:, i], lags=100, title="Non Centered Gibbs")
        plot_acf(h_theta_asis[:, i], lags=100, title="ASIS")
    """

