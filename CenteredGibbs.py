from GibbsSampler import NormalSampler
import numpy as np
import config
import utils
import healpy as hp
import qcinv


class CenteredNormalSampler(NormalSampler):


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

    def sample_normal(self, all_cls, old_s=None):
        if self.mask_path is None:
            return self.sample_no_mask(all_cls)
        elif self.gibbs_cr:
            return self.sample_gibbs_change_variables(all_cls, old_s)
        elif self.overrelax:
            return self.overrelaxation(all_cls, old_s)
        else:
            return self.sample_mask(all_cls)


