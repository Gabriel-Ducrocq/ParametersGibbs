import healpy as hp
import numpy as np
import config
import utils
import matplotlib.pyplot as plt


a = np.random.normal(size = 100000)*np.sqrt(1/2)

plt.hist(a, density=True)
plt.show()

cls_TT, cls_EE, cls_BB, cls_TE = utils.generate_cls(config.COSMO_PARAMS_MEAN, pol=True)

all_cls = np.zeros((len(cls_TT), 3, 3))
all_cls[:, 0, 0] = cls_TT
all_cls[:, 1, 1] = cls_EE
all_cls[:, 2, 2] = cls_BB
all_cls[:, 1, 0] = all_cls[:, 0, 1] = cls_TE








map_I, map_Q, map_U = hp.synfast([cls_TT, cls_EE, cls_BB, cls_TE], lmax=config.L_MAX_SCALARS, nside=config.NSIDE,
                                 fwhm=config.fwhm_radians, new=True)

alms_t, alms_e, alms_b = hp.map2alm([map_I, map_U, map_Q], lmax=config.L_MAX_SCALARS, pol=True)

hp.almxfl(alms_t, config.bl_gauss, inplace=True)
hp.almxfl(alms_e, config.bl_gauss, inplace=True)
hp.almxfl(alms_b, config.bl_gauss, inplace=True)

alms = np.zeros((len(alms_t), 3), dtype=complex)
alms[:, 0] = alms_t
alms[:, 1] = alms_e
alms[:, 2] = alms_b

alms = utils.matrix_product(all_cls, alms)

alms_t = hp.almxfl(alms[:, 0], config.bl_gauss, inplace=False)
alms_e = hp.almxfl(alms[:, 1], config.bl_gauss, inplace=False)
alms_b = hp.almxfl(alms[:, 2], config.bl_gauss, inplace=False)

I, Q, U = hp.alm2map([alms_t, alms_e, alms_b], pol=True, nside=config.NSIDE, lmax=config.L_MAX_SCALARS)





alms_t, alms_e, alms_b = hp.map2alm([I, U, Q], lmax=config.L_MAX_SCALARS, pol=True)

hp.almxfl(alms_t, 1/config.bl_gauss, inplace=True)
hp.almxfl(alms_e, 1/config.bl_gauss, inplace=True)
hp.almxfl(alms_b, 1/config.bl_gauss, inplace=True)

alms = np.zeros((len(alms_t), 3), dtype=complex)
alms[:, 0] = alms_t
alms[:, 1] = alms_e
alms[:, 2] = alms_b

inv_cov, _ = utils.compute_inverse_and_cholesky_constraint_realization(all_cls)
alms = utils.matrix_product(inv_cov, alms)

alms_t = hp.almxfl(alms[:, 0], 1/config.bl_gauss, inplace=False)
alms_e = hp.almxfl(alms[:, 1], 1/config.bl_gauss, inplace=False)
alms_b = hp.almxfl(alms[:, 2], 1/config.bl_gauss, inplace=False)


I, Q, U = hp.alm2map([alms_t, alms_e, alms_b], pol=True, nside=config.NSIDE, lmax=config.L_MAX_SCALARS)





print(np.abs((map_I - I)/I))