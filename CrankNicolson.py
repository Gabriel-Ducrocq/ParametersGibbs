import numpy as np
import qcinv
import healpy as hp
import utils


class CrankNicolson():

    def __init__(self, nside, lmax, noise_I, noise_Q, beam_fwhm_radians, pix_map, mask_path = None, pcg_accuracy=1e-6,
                 gibbs_cr = False, overrelax=False, beta = 0.99):
        ### We assume noise Q = noise U
        self.nside = nside
        self.Npix = 12*nside**2
        self.lmax = lmax
        self.noise_covar_I = noise_I
        self.noise_covar_Q = noise_Q
        self.inv_noise_covar_I = 1/noise_I
        self.inv_noise_covar_Q = 1/noise_Q
        self.beam_fwhm_radians = beam_fwhm_radians
        self.bl_gauss = hp.gauss_beam(self.beam_fwhm_radians, lmax=self.lmax)
        self.pix_weight = 4*np.pi/self.Npix
        self.pix_map = pix_map
        self.complex_dim = int((self.lmax + 1) * (self.lmax + 2) / 2)
        self.pcg_accuracy = 1e-10
        self.n_gibbs = 1
        self.alpha_overrelax = -0.97
        self.mask_path = mask_path
        self.gibbs_cr = gibbs_cr
        self.overrelax = overrelax

        if mask_path is None:
            map = {"I": None, "Q": None, "U": None}
            map["I"] = self.pix_map["I"] * self.inv_noise_covar_I
            map["Q"] = self.pix_map["Q"] * self.inv_noise_covar_Q
            map["U"] = self.pix_map["U"] * self.inv_noise_covar_Q
            alms_T, alms_E, alms_B = hp.map2alm([map["I"], map["Q"], map["U"]], lmax=self.lmax, pol=True)
            alms_T /= self.pix_weight
            alms_E /= self.pix_weight
            alms_B /= self.pix_weight

            hp.almxfl(alms_T, self.bl_gauss, inplace=True)
            hp.almxfl(alms_E, self.bl_gauss, inplace=True)
            hp.almxfl(alms_B, self.bl_gauss, inplace=True)

            self.r = np.zeros((len(alms_T), 3), dtype=complex)
            self.r[:, 0] = alms_T
            self.r[:, 1] = alms_E
            self.r[:, 2] = alms_B

            self.inv_noise_covar_I *= np.ones(self.Npix)
            self.inv_noise_covar_Q *= np.ones(self.Npix)
            self.inv_noise = [self.inv_noise_covar_I, self.inv_noise_covar_Q]
            self.noise = [1/self.inv_noise_covar_I,1/self.inv_noise_covar_Q]
        else:
            self.mask = hp.ud_grade(hp.read_map(mask_path), self.nside)
            self.inv_noise_covar_I *= self.mask
            self.inv_noise_covar_Q *= self.mask
            self.inv_noise = [self.inv_noise_covar_I, self.inv_noise_covar_Q]
            self.noise =[self.noise_covar_I*self.mask, self.noise_covar_Q*self.mask]

            map = {"I": None, "Q": None, "U": None}
            map["I"] = self.pix_map["I"] * self.inv_noise_covar_I
            map["Q"] = self.pix_map["Q"] * self.inv_noise_covar_Q
            map["U"] = self.pix_map["U"] * self.inv_noise_covar_Q
            alms_T, alms_E, alms_B = hp.sphtfunc.smoothalm(
                hp.map2alm([map["I"], map["Q"], map["U"]], lmax=self.lmax, pol=True),
                pol=True, fwhm=self.beam_fwhm_radians, inplace=False)

            self.r = np.zeros(3 * len(alms_T), dtype=complex)
            self.r[0::3] = alms_T
            self.r[1::3] = alms_E
            self.r[2::3] = alms_B

        class cl(object):
            pass

        self.s_cls = cl

        self.pcg_accuracy = 1.0e-6
        self.n_inv_filt = qcinv.opfilt_tp.alm_filter_ninv(self.inv_noise, self.bl_gauss, marge_maps_t=[], marge_maps_p=[])
        self.chain_descr = [[0, ["diag_cl"], self.lmax, self.nside, 4000, self.pcg_accuracy, qcinv.cd_solve.tr_cg,
                             qcinv.cd_solve.cache_mem()]]

        self.mu = np.max(self.inv_noise) + 1e-8
        self.beta = beta

    def propose(self, old_s):
        s_I = self.beta*np.random.normal(size=self.Npix)*np.sqrt(self.noise[0]) + np.sqrt(1 - self.beta**2)*old_s[:, 0]
        s_Q = self.beta*np.random.normal(size=self.Npix)*np.sqrt(self.noise[1]) + np.sqrt(1 - self.beta**2)*old_s[:, 1]
        s_U = self.beta*np.random.normal(size=self.Npix)*np.sqrt(self.noise[1]) + np.sqrt(1 - self.beta**2)*old_s[:, 2]

        new_S = np.zeros((self.Npix, 3))
        new_S[:, 0] = s_I
        new_S[:, 1] = s_Q
        new_S[:, 2] = s_U

        return new_S

    def compute_log_likelihood(self, new_s, all_cls):
        inv_cls, _ = utils.compute_inverse_and_cholesky_constraint_realization(all_cls)

        y_I = self.pix_map["I"] + new_s[:, 0]
        y_Q = self.pix_map["Q"] + new_s[:, 1]
        y_U = self.pix_map["U"] + new_s[:, 2]

        alms_t, alms_e, alms_b = hp.map2alm([y_I, y_Q, y_U], pol=True, lmax=self.lmax, iter=0)

        hp.almxfl(alms_t, 1/self.bl_gauss, inplace=True)
        hp.almxfl(alms_e, 1/self.bl_gauss, inplace=True)
        hp.almxfl(alms_b, 1/self.bl_gauss, inplace=True)

        alms = np.zeros((len(alms_t), 3), dtype=complex)
        alms[:, 0] = alms_t
        alms[:, 1] = alms_e
        alms[:, 2] = alms_b

        inv_cl_tt = np.zeros(len(all_cls[:, 0, 0]))
        inv_cl_tt[all_cls[:, 0, 0] != 0] = 1/all_cls[all_cls[:, 0, 0] != 0, 0, 0]
        firrst_product = hp.almxfl(alms_t, inv_cl_tt, inplace=False)
        second_product = np.sum(np.conjugate(alms_t)*firrst_product)
        #first_product = utils.matrix_product(inv_cls, alms)
        #second_product = np.sum(np.conjugate(alms)*first_product)
        result = second_product.real#*(self.pix_weight)**2

        return -result

    def compute_log_likelihood_ratio(self, old_s, new_s, all_cls):
        num = self.compute_log_likelihood(new_s, all_cls)
        denom = self.compute_log_likelihood(old_s, all_cls)

        return num - denom


    def run(self, old_s, all_cls):
        ###We first make the change of variable:
        old_s[:, 0] -= self.pix_map["I"]
        old_s[:, 1] -= self.pix_map["Q"]
        old_s[:, 2] -= self.pix_map["U"]

        ###Then we sample

        new_s = self.propose(old_s)
        log_r = self.compute_log_likelihood_ratio(old_s, new_s, all_cls)
        print("Log ratio:", log_r)
        accept = 0

        if np.log(np.random.uniform()) < log_r:
            accept = 1
            old_s = new_s


        ### Finally we change back
        old_s[:, 0] += self.pix_map["I"]
        old_s[:, 1] += self.pix_map["Q"]
        old_s[:, 2] += self.pix_map["U"]
        return old_s, accept





