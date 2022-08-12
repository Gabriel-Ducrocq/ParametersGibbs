import numpy as np
import utils
import config
import healpy as hp
import qcinv
import time
import matplotlib.pyplot as plt


class GRWMH():

    def __init__(self, nside, lmax, proposal_variance, n_iter = 1):
        self.nside = nside
        self.lmax = lmax
        self.n_iter = n_iter
        self.dimension = proposal_variance.shape[0]
        if len(proposal_variance.shape) == 1:
            self.proposal_stdd = np.sqrt(proposal_variance)
        else:
            self.proposal_stdd = np.linalg.cholesky(proposal_variance)

        self.metropolis_within_gibbs = True

    def propose_new_params(self, old_params):
        if len(self.proposal_stdd.shape) == 1:
            new_params = np.random.normal(loc=old_params, scale=self.proposal_stdd)
            new_params[-1] = old_params[-1]
        else:
            new_params = old_params + np.dot(self.proposal_stdd,np.random.normal(loc=0, scale=1, size=self.dimension))

        cls_tt, cls_ee, cls_bb, cls_te = utils.generate_cls(new_params)
        all_cls_new = np.zeros((len(cls_tt), 3, 3))
        all_cls_new[:, 0, 0] = cls_tt
        all_cls_new[:, 1, 1] = cls_ee
        all_cls_new[:, 2, 2] = cls_bb
        all_cls_new[:, 1, 0] = all_cls_new[:, 0, 1] = cls_te

        inv_cls_new, _ = utils.compute_inverse_and_cholesky_constraint_realization(all_cls_new)
        return new_params, all_cls_new, inv_cls_new

    def compute_log_prior(self, params):
        log_prior = -(1/2)*np.sum(((params[:-1] - config.COSMO_PARAMS_MEAN[:-1])/config.COSMO_PARAMS_SIGMA[:-1])**2)
        return log_prior

    def compute_log_likelihood(self, all_inv_cls, alm_map):
        first_product = utils.matrix_product(all_inv_cls, alm_map)
        result = np.sum(np.conjugate(alm_map)*first_product)

        ### It is NOT all_inv_cls here ! We should use the the covariance matrix, not the precision
        log_det =np.sum(np.log([np.linalg.det(cls) for cls in all_inv_cls[2:]]))

        print("Overall")
        print(-result + (1 / 2) * log_det)
        ###Here we take + (1/2)*log(det) because we computed the log det of the inverse covariance matrix.
        return -result + (1/2)*log_det

    def compute_log_MH_ratio(self, old_params, old_inv_cls, new_params, new_inv_cls, alm_map):
        num = self.compute_log_likelihood(new_inv_cls, alm_map) + self.compute_log_prior(new_params)
        denom = self.compute_log_likelihood(old_inv_cls, alm_map) + self.compute_log_prior(old_params)
        print("Num:", num)
        print("Denom:", denom)
        print("Diff:", num - denom)
        return num - denom

    def run_step(self, old_params, old_cls, old_inv_cls, alm_map):
        accept = 0
        new_params, new_cls, new_inv_cls = self.propose_new_params(old_params)
        log_r = self.compute_log_MH_ratio(old_params, old_inv_cls, new_params, new_inv_cls, alm_map)
        print("Acceptance proba")
        print(log_r)
        if np.log(np.random.uniform()) < log_r.real:
            old_params = new_params
            old_inv_cls = new_inv_cls
            old_cls = new_cls
            accept = 1

        return old_params, old_cls, old_inv_cls, accept

    def run(self, init_params, init_cls, init_inv_cls,  alm_map):
        acceptions = []
        #h_params = []
        old_params = init_params
        old_inv_cls = init_inv_cls
        old_cls = init_cls
        for i in range(self.n_iter):
            old_params, old_cls, old_inv_cls, accept = self.run_step(old_params, old_cls, old_inv_cls, alm_map)
            acceptions.append(accept)
            #h_params.append(old_params)

        print("Acceptance rate:")
        print(np.mean(acceptions))
        return old_params, old_cls, old_inv_cls, acceptions#, h_params






class NormalSampler():

    def __init__(self, nside, lmax, noise_I, noise_Q, beam_fwhm_radians, pix_map, mask_path = None, pcg_accuracy=1e-6,
                 gibbs_cr = False, overrelax=False):
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
            self.noise = [1 / self.inv_noise_covar_I, 1 / self.inv_noise_covar_Q]
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


    def sample_no_mask(self, all_cls):
        additive_pixel_term_I = self.bl_gauss**2*self.inv_noise_covar_I[0]/self.pix_weight
        additive_pixel_term_Q = self.bl_gauss**2*self.inv_noise_covar_Q[0]/self.pix_weight
        additive_pixel_term_U = self.bl_gauss**2*self.inv_noise_covar_Q[0]/self.pix_weight
        additive_pixel_term = np.zeros((len(additive_pixel_term_I), 3))
        additive_pixel_term[:, 0] = additive_pixel_term_I
        additive_pixel_term[:, 1] = additive_pixel_term_Q
        additive_pixel_term[:, 2] = additive_pixel_term_U
        cov_matrix, cholesky_cov_matrix = utils.compute_inverse_and_cholesky_constraint_realization(all_cls, additive_pixel_term)

        mu = utils.matrix_product(cov_matrix, self.r)

        alms = standard_normal = np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3))\
                          + np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3))*1j

        alms[:self.lmax+1, :] = np.random.normal(size=(self.lmax+1, 3))

        alms = utils.matrix_product(cholesky_cov_matrix, standard_normal) + mu

        alms[0, 1:] = alms[1, 1:] = 0
        alms[self.lmax+1, 1:] = 0

        return alms


    def sample_mask(self, all_cls):
        cls_TT = all_cls[:,0, 0]
        cls_EE = all_cls[:, 1, 1]
        cls_BB = all_cls[:, 2, 2]
        cls_TE = all_cls[:, 0, 1]

        self.s_cls.cltt = cls_TT
        self.s_cls.clee = cls_EE
        self.s_cls.clbb = cls_BB
        self.s_cls.clte = cls_TE
        self.s_cls.lmax = self.lmax

        chain = qcinv.multigrid.multigrid_chain(qcinv.opfilt_tp, self.chain_descr, self.s_cls, self.n_inv_filt,
                                                debug_log_prefix=None)
        soltn = qcinv.opfilt_tp.teblm(np.zeros((3, int(qcinv.util_alm.lmax2nlm(self.lmax))), dtype=np.complex))
        pix_map = [self.pix_map["I"], self.pix_map["Q"], self.pix_map["U"]]


        fluc_T = hp.map2alm(np.random.normal(size=self.Npix)*np.sqrt(self.inv_noise_covar_I),lmax=self.lmax, iter=0, pol=False)
        fluc_E, fluc_B = hp.map2alm_spin([np.random.normal(size=self.Npix)*np.sqrt(self.inv_noise_covar_Q),
                                          np.random.normal(size=self.Npix)*np.sqrt(self.inv_noise_covar_Q)], spin=2, lmax=self.lmax)

        fluc_T *= self.Npix / (4 * np.pi)
        fluc_E *= self.Npix / (4 * np.pi)
        fluc_B *= self.Npix / (4 * np.pi)

        hp.almxfl(fluc_T, self.bl_gauss, inplace=True)
        hp.almxfl(fluc_E, self.bl_gauss, inplace=True)
        hp.almxfl(fluc_B, self.bl_gauss, inplace=True)

        inverse, chol = utils.compute_inverse_and_cholesky_constraint_realization(all_cls)

        white_noise = np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3))\
                                        + np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3))*1j

        white_noise[:self.lmax+1, :] = np.random.normal(size=(self.lmax+1, 3))

        second_fluc_term = utils.matrix_product(chol, white_noise)


        b_fluctuations = {"tlm": fluc_T + second_fluc_term[:, 0], "elm":(fluc_E+second_fluc_term[:, 1]),
                          "blm":(fluc_B+second_fluc_term[:, 2])}


        _ = chain.sample(soltn, pix_map, b_fluctuations, pol=True, temp=True)

        alms = np.zeros((self.complex_dim, 3), dtype=complex)
        alms[:, 0] = soltn.tlm
        alms[:, 1] = soltn.elm
        alms[:, 2] = soltn.blm

        return alms


    def sample_gibbs_change_variables(self, all_cls, old_s):
        var_I = self.mu - self.inv_noise[0]
        var_Q = self.mu - self.inv_noise[1]
        var_U = self.mu - self.inv_noise[1]


        additive_pixel_term_I = self.bl_gauss**2*self.mu/self.pix_weight
        additive_pixel_term_Q = self.bl_gauss**2*self.mu/self.pix_weight
        additive_pixel_term_U = self.bl_gauss**2*self.mu/self.pix_weight
        additive_pixel_term = np.zeros((len(additive_pixel_term_I), 3))
        additive_pixel_term[:, 0] = additive_pixel_term_I
        additive_pixel_term[:, 1] = additive_pixel_term_Q
        additive_pixel_term[:, 2] = additive_pixel_term_U

        inv_s_cov, chol_s_cov = utils.compute_inverse_and_cholesky_constraint_realization(all_cls, additive_pixel_term)


        for m in range(self.n_gibbs):
            map_I = hp.alm2map(hp.almxfl(old_s[:, 0], self.bl_gauss, inplace=False),
                                            lmax=self.lmax, nside=self.nside)

            map_Q, map_U = hp.alm2map_spin([hp.almxfl(old_s[:, 1], self.bl_gauss, inplace=False),
                                              hp.almxfl(old_s[:, 2], self.bl_gauss, inplace=False)],
                                            lmax=self.lmax, nside=self.nside, spin=2)


            mean_I = var_I*map_I
            mean_Q = var_Q*map_Q
            mean_U = var_U*map_U

            v_I = np.random.normal(size=len(mean_I))*np.sqrt(var_I) + mean_I
            v_Q = np.random.normal(size=len(mean_Q))*np.sqrt(var_Q) + mean_Q
            v_U = np.random.normal(size=len(mean_U))*np.sqrt(var_U) + mean_U



            alms_T = hp.map2alm(v_I + self.inv_noise[0] * self.pix_map["I"], lmax=self.lmax,iter=3)

            alms_E, alms_B = hp.map2alm_spin([v_Q + self.inv_noise[1] * self.pix_map["Q"],
                                            v_U + self.inv_noise[1] * self.pix_map["U"]], lmax=self.lmax,spin=2)

            alms_T = hp.almxfl(alms_T / self.pix_weight, self.bl_gauss, inplace=False)
            alms_E = hp.almxfl(alms_E / self.pix_weight, self.bl_gauss, inplace=False)
            alms_B = hp.almxfl(alms_B / self.pix_weight, self.bl_gauss, inplace=False)

            alms = np.zeros((len(alms_T), 3), dtype=complex)
            alms[:, 0] = alms_T
            alms[:, 1] = alms_E
            alms[:, 2] = alms_B

            mean_s = utils.matrix_product(inv_s_cov, alms)

            white_noise = np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3)) \
                          + 1j*np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3))

            white_noise[:self.lmax+1, :] = np.random.normal(size=(self.lmax+1, 3))

            old_s = utils.matrix_product(chol_s_cov, white_noise) + mean_s

            ### Because A^T A is not exactly equal to identity with alm2map_spin and map2alm_spin:
            ### it is equal to identity with zeros on the diag components corresponding to monopole and dipole.
            ### Since for l=0 and l = 1 Cl= 0, it follows that the these components have covariance 0
            old_s[0, 1:] = old_s[1, 1:] = 0
            old_s[self.lmax + 1, 1:] = 0

        return old_s


    def overrelaxation(self, all_cls, old_s):
        var_I = self.mu - self.inv_noise[0]
        var_Q = self.mu - self.inv_noise[1]
        var_U = self.mu - self.inv_noise[1]


        additive_pixel_term_I = self.bl_gauss**2*self.mu/self.pix_weight
        additive_pixel_term_Q = self.bl_gauss**2*self.mu/self.pix_weight
        additive_pixel_term_U = self.bl_gauss**2*self.mu/self.pix_weight
        additive_pixel_term = np.zeros((len(additive_pixel_term_I), 3))
        additive_pixel_term[:, 0] = additive_pixel_term_I
        additive_pixel_term[:, 1] = additive_pixel_term_Q
        additive_pixel_term[:, 2] = additive_pixel_term_U

        inv_s_cov, chol_s_cov = utils.compute_inverse_and_cholesky_constraint_realization(all_cls, additive_pixel_term)


        for m in range(self.n_gibbs):
            map_I = hp.alm2map(hp.almxfl(old_s[:, 0], self.bl_gauss, inplace=False),
                                            lmax=self.lmax, nside=self.nside)

            map_Q, map_U = hp.alm2map_spin([hp.almxfl(old_s[:, 1], self.bl_gauss, inplace=False),
                                              hp.almxfl(old_s[:, 2], self.bl_gauss, inplace=False)],
                                            lmax=self.lmax, nside=self.nside, spin=2)


            mean_I = var_I*map_I
            mean_Q = var_Q*map_Q
            mean_U = var_U*map_U

            v_I = np.random.normal(size=len(mean_I))*np.sqrt(var_I) + mean_I
            v_Q = np.random.normal(size=len(mean_Q))*np.sqrt(var_Q) + mean_Q
            v_U = np.random.normal(size=len(mean_U))*np.sqrt(var_U) + mean_U



            alms_T = hp.map2alm(v_I + self.inv_noise[0] * self.pix_map["I"], lmax=self.lmax,iter=3)

            alms_E, alms_B = hp.map2alm_spin([v_Q + self.inv_noise[1] * self.pix_map["Q"],
                                            v_U + self.inv_noise[1] * self.pix_map["U"]], lmax=self.lmax,spin=2)

            alms_T = hp.almxfl(alms_T / self.pix_weight, self.bl_gauss, inplace=False)
            alms_E = hp.almxfl(alms_E / self.pix_weight, self.bl_gauss, inplace=False)
            alms_B = hp.almxfl(alms_B / self.pix_weight, self.bl_gauss, inplace=False)

            alms = np.zeros((len(alms_T), 3), dtype=complex)
            alms[:, 0] = alms_T
            alms[:, 1] = alms_E
            alms[:, 2] = alms_B

            mean_s = utils.matrix_product(inv_s_cov, alms)

            white_noise = np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3)) \
                          + 1j*np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3))

            white_noise[:self.lmax+1, :] = np.random.normal(size=(self.lmax+1, 3))

            old_s = utils.matrix_product(chol_s_cov, white_noise) + mean_s

            ### Because A^T A is not exactly equal to identity with alm2map_spin and map2alm_spin:
            ### it is equal to identity with zeros on the diag components corresponding to monopole and dipole.
            ### Since for l=0 and l = 1 Cl= 0, it follows that the these components have covariance 0
            old_s[0, 1:] = old_s[1, 1:] = 0
            old_s[self.lmax + 1, 1:] = 0


            ##Again
            map_I = hp.alm2map(hp.almxfl(old_s[:, 0], self.bl_gauss, inplace=False),
                                            lmax=self.lmax, nside=self.nside)

            map_Q, map_U = hp.alm2map_spin([hp.almxfl(old_s[:, 1], self.bl_gauss, inplace=False),
                                              hp.almxfl(old_s[:, 2], self.bl_gauss, inplace=False)],
                                            lmax=self.lmax, nside=self.nside, spin=2)


            mean_I = var_I*map_I
            mean_Q = var_Q*map_Q
            mean_U = var_U*map_U

            v_I = np.random.normal(size=len(mean_I))*np.sqrt(var_I)*np.sqrt(1 - self.alpha_overrelax**2) \
                  + self.alpha_overrelax*(v_I - mean_I) + mean_I
            v_Q = np.random.normal(size=len(mean_Q))*np.sqrt(var_Q)*np.sqrt(1 - self.alpha_overrelax**2) \
                  + self.alpha_overrelax*(v_Q - mean_Q) + mean_Q
            v_U = np.random.normal(size=len(mean_U))*np.sqrt(var_U)*np.sqrt(1 - self.alpha_overrelax**2) \
                  + self.alpha_overrelax*(v_U - mean_U) + mean_U



            alms_T = hp.map2alm(v_I + self.inv_noise[0] * self.pix_map["I"], lmax=self.lmax,iter=3)

            alms_E, alms_B = hp.map2alm_spin([v_Q + self.inv_noise[1] * self.pix_map["Q"],
                                            v_U + self.inv_noise[1] * self.pix_map["U"]], lmax=self.lmax,spin=2)

            alms_T = hp.almxfl(alms_T / self.pix_weight, self.bl_gauss, inplace=False)
            alms_E = hp.almxfl(alms_E / self.pix_weight, self.bl_gauss, inplace=False)
            alms_B = hp.almxfl(alms_B / self.pix_weight, self.bl_gauss, inplace=False)

            alms = np.zeros((len(alms_T), 3), dtype=complex)
            alms[:, 0] = alms_T
            alms[:, 1] = alms_E
            alms[:, 2] = alms_B

            mean_s = utils.matrix_product(inv_s_cov, alms)

            white_noise = np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3)) \
                          + 1j*np.sqrt(1/2)*np.random.normal(size=(self.complex_dim, 3))

            white_noise[:self.lmax+1, :] = np.random.normal(size=(self.lmax+1, 3))

            old_s = utils.matrix_product(chol_s_cov, white_noise)*np.sqrt(1-self.alpha_overrelax**2)\
                    + self.alpha_overrelax*(old_s - mean_s) + mean_s

            ### Because A^T A is not exactly equal to identity with alm2map_spin and map2alm_spin:
            ### it is equal to identity with zeros on the diag components corresponding to monopole and dipole.
            ### Since for l=0 and l = 1 Cl= 0, it follows that the these components have covariance 0
            old_s[0, 1:] = old_s[1, 1:] = 0
            old_s[self.lmax + 1, 1:] = 0


        return old_s

    def sample_pixel(self, all_cls, old_s_pix):
        self.beta = 0.99

        s_I = self.beta*np.random.normal(size=self.Npix)*np.sqrt(self.noise[0])  + np.sqrt(1-self.beta**2)*old_s_pix["I"]
        s_Q = self.beta*np.random.normal(size=self.Npix) * np.sqrt(self.noise[1])+ np.sqrt(1-self.beta**2)*old_s_pix["Q"]
        s_U = self.beta*np.random.normal(size=self.Npix) * np.sqrt(self.noise[1])+ np.sqrt(1-self.beta**2)*old_s_pix["U"]

        alms_t, alms_e, alms_b = hp.map2alm([s_I -self.pix_map["I"]
                                                , s_Q- self.pix_map["Q"]
                                                , s_U- self.pix_map["U"]], lmax=config.L_MAX_SCALARS, pol=True, iter = 3)

        hp.almxfl(alms_t, 1/self.bl_gauss, inplace=True)
        hp.almxfl(alms_e, 1/self.bl_gauss, inplace=True)
        hp.almxfl(alms_b, 1/self.bl_gauss, inplace=True)

        inv_cls, _ = utils.compute_inverse_and_cholesky_constraint_realization(all_cls)

        alms = np.zeros((len(alms_t), 3), dtype=complex)
        alms[:, 0]=alms_t
        alms[:, 1]=alms_e
        alms[:, 2]=alms_b
        first_prod = utils.matrix_product(inv_cls, alms)
        second_prod = np.sum(np.conjugate(alms)*first_prod).real
        result1 = second_prod*(4*np.pi/self.Npix)**2

        alms_t, alms_e, alms_b = hp.map2alm([old_s_pix["I"] - self.pix_map["I"]
                                                , old_s_pix["Q"] - self.pix_map["Q"],
                                             old_s_pix["U"] - self.pix_map["U"] ], lmax=config.L_MAX_SCALARS, pol=True, iter = 3)

        hp.almxfl(alms_t, 1/self.bl_gauss, inplace=True)
        hp.almxfl(alms_e, 1/self.bl_gauss, inplace=True)
        hp.almxfl(alms_b, 1/self.bl_gauss, inplace=True)

        inv_cls, _ = utils.compute_inverse_and_cholesky_constraint_realization(all_cls)

        alms = np.zeros((len(alms_t), 3), dtype=complex)
        alms[:, 0]=alms_t
        alms[:, 1]=alms_e
        alms[:, 2]=alms_b

        #plt.hist(np.abs(alms[:, 0]), bins = 50)
        #plt.show()
        first_prod = utils.matrix_product(inv_cls, alms)
        second_prod = np.sum(np.conjugate(alms)*first_prod).real
        result2 = second_prod*(4*np.pi/self.Npix)**2

        ratio = -(result1 - result2)

        print("LogRatio:", ratio)
        #print((interm1 - interm2)[:10])
        #plt.hist(interm1.flatten(), density=True, bins = 50)
        #plt.show()
        #print("Interm 1 sum:",np.sum(interm1))
        if np.log(np.random.uniform()) < ratio:
            return {"I":s_I, "Q":s_Q, "U":s_U}

        else:
            return old_s_pix


    def sample_alms(self, old_map, all_cls):
        self.beta = 0.99


        inv_cls, chol_cls = utils.compute_inverse_and_cholesky_constraint_realization(all_cls)

        alms = np.random.normal(size =(self.complex_dim, 3))*np.sqrt(1/2) + 1j*np.random.normal(size =(self.complex_dim, 3))*np.sqrt(1/2)
        alms[:self.lmax+1] = np.random.normal(size=(self.lmax+1, 3))

        alms = utils.matrix_product(chol_cls, alms)

        alms_t = hp.almxfl(alms[:, 0], self.bl_gauss, inplace=False)
        alms_e = hp.almxfl(alms[:, 1], self.bl_gauss, inplace=False)
        alms_b = hp.almxfl(alms[:, 2], self.bl_gauss, inplace=False)

        I, Q, U = hp.alm2map([alms_t, alms_e, alms_b], lmax=self.lmax, nside=self.nside, pol=True)

        new_I = self.beta*I + np.sqrt(1-self.beta**2)*old_map[:, 0]
        new_Q = self.beta*Q + np.sqrt(1-self.beta**2)*old_map[:, 1]
        new_U = self.beta*U + np.sqrt(1-self.beta**2)*old_map[:, 2]

        rI = -(1/2)*np.sum((new_I - self.pix_map["I"])**2*self.inv_noise[0])
        rQ = -(1/2)*np.sum((new_Q - self.pix_map["Q"])**2*self.inv_noise[0])
        rU = -(1/2)*np.sum((new_U - self.pix_map["U"])**2*self.inv_noise[0])


        rIold = -(1/2)*np.sum((old_map[:, 0] - self.pix_map["I"])**2*self.inv_noise[0])
        rQold = -(1/2)*np.sum((old_map[:, 1] - self.pix_map["Q"])**2*self.inv_noise[0])
        rUold = -(1/2)*np.sum((old_map[:, 2] - self.pix_map["U"])**2*self.inv_noise[0])

        accept = 0
        log_ratio = rI + rQ + rU - rIold - rQold - rUold
        print("Log ratio:", log_ratio)
        if np.log(np.random.uniform()) < log_ratio:
            old_map[:, 0] = rI
            old_map[:, 1] = rQ
            old_map[:, 2] = rU
            accept = 1

        return old_map, accept



    def sample_normal(self, all_cls, old_s=None):
        if self.mask_path is None:
            return self.sample_no_mask(all_cls)
        elif self.gibbs_cr:
            return self.sample_gibbs_change_variables(all_cls, old_s)
        elif self.overrelax:
            return self.overrelaxation(all_cls, old_s)
        else:
            return self.sample_mask(all_cls)


class GibbsSampler():

    def __init__(self, nside, lmax, noise_I, noise_Q, beam, proposal_variance, pix_map, n_iter, pix_weight= 4*np.pi/config.Npix, gibbs_cr=False,
                 overrelax=False, n_iter_grwmh = 1, mask_path = None):
        self.nside = nside
        self.lmax = lmax
        self.grwmh_sampler = None
        self.beam = beam
        self.n_iter = n_iter
        self.normal_sampler = None
        self.proposal_variance = proposal_variance
        self.pix_map = pix_map
        self.pix_weight = pix_weight
        self.gibbs_cr = gibbs_cr
        self.overrelax = overrelax
        self.grwmh_sampler = GRWMH(nside, lmax, proposal_variance, n_iter=n_iter_grwmh)
        self.normal_sampler = NormalSampler(nside, lmax, noise_I, noise_Q, beam, pix_map, gibbs_cr= gibbs_cr,
                                            overrelax=overrelax, mask_path=mask_path)

    def run(self, theta_init):
        history_theta = []
        h_acceptions = []

        theta = theta_init
        cls_tt, cls_ee, cls_bb, cls_te = utils.generate_cls(theta_init)
        all_cls = np.zeros((len(cls_tt), 3, 3))
        all_cls[:, 0, 0] = cls_tt
        all_cls[:, 1, 1] = cls_ee
        all_cls[:, 2, 2] = cls_bb
        all_cls[:, 0, 1] = all_cls[:, 1, 0] = cls_te

        print(all_cls)
        inv_cls, _ = utils.compute_inverse_and_cholesky_constraint_realization(all_cls)

        alm_map = None
        if self.gibbs_cr or self.overrelax:
            alm_map = self.normal_sampler.sample_mask(all_cls)

        for i in range(self.n_iter):
            if i % 100 == 0:
                print("Iteration:", i)

            history_theta.append(theta)
            start = time.time()
            alm_map = self.normal_sampler.sample_normal(all_cls, alm_map)
            end = time.time()
            print("Normal sampling time:", end - start)

            start = time.time()
            theta, all_cls, inv_cls, acceptions = self.grwmh_sampler.run(theta, all_cls, inv_cls, alm_map)
            end = time.time()
            print("GRWMH sampler time:", end - start)
            history_theta.append(theta)
            h_acceptions.append(acceptions)
            print("\n\n\n")


        print("Acceptance rate:", np.mean(h_acceptions))

        return np.array(history_theta), h_acceptions



