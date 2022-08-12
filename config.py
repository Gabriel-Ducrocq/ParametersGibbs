import numpy as np
import healpy as hp


COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "H0", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN = np.array([0.9665, 0.02242, 0.11933, 68.2, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 1.5, 0.014, 0.0071])
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])

#COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "H0", "ln10^{10}A_s"]
#COSMO_PARAMS_MEAN = np.array([0.9665, 0.02242, 0.11933, 68.2, 3.047])
#COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 1.5, 0.014])

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'
observations = None

NSIDE = 32
Npix = 12*NSIDE**2
L_MAX_SCALARS = 2*NSIDE
#L_MAX_SCALARS = 2500
dim_alm = (L_MAX_SCALARS+1)**2 - 4

#mask_path = "HFI_Mask_GalPlane-apo0_2048_R2_80%_bis.00.fits"


noise = 40**2*(4*np.pi)/Npix
#noise = 1.84e-3
noise_covar = np.array([noise if i == 0 else noise / 2 for l in range(2, L_MAX_SCALARS + 1) for i in range(2 * l + 1)])
noise_covar_I = 0.2**2
noise_covar_Q = (0.2/np.sqrt(2))**2
scale = np.array([2*np.pi/(l*(l+1)) for l in range(2, L_MAX_SCALARS+1)])

#fwhm_arcmin = 13
#fwhm_radians = (np.pi/(180*60))*fwhm_arcmin
fwhm_radians = (np.pi / 180) * 0.35
bl_gauss = hp.gauss_beam(fwhm_radians, lmax=L_MAX_SCALARS)[2:]
beam = np.array([bl for l, bl in enumerate(bl_gauss, start=2) for i in range(2 * l + 1)])

l_cut = 555
cpt = 10

proposal_variance = 0.02*COSMO_PARAMS_SIGMA
#proposal_variance_gibbs = 0.05*COSMO_PARAMS_SIGMA**2
#proposal_variance_pncp_gibbs = 0.015*COSMO_PARAMS_SIGMA**2
#proposal_variance_rescale = 0.2*COSMO_PARAMS_SIGMA**2
#proposal_variance_gibbs_nc = 0.007*COSMO_PARAMS_SIGMA**2
#proposal_variance_direct = 0.4*COSMO_PARAMS_SIGMA**2
#proposal_variance_gibbs_asis = 0.04*COSMO_PARAMS_SIGMA**2
#proposal_variance_gibbs_nc_asis = 0.006*COSMO_PARAMS_SIGMA**2


#data_direct = np.load("data_direct.npy", allow_pickle=True)
#data_direct = data_direct.item()
#h_theta_direct = data_direct["h_theta"]
#roposal_variance_direct = (2.4/np.sqrt(6))*np.cov(h_theta_direct.transpose())

#data_centered_gibbs = np.load("data_centered_gibbs.npy", allow_pickle = True)
#data_centered_gibbs = data_centered_gibbs.item()
#h_theta_centered = data_centered_gibbs["h_theta"]
#proposal_variance = 0.2*(2.4/np.sqrt(6))*np.cov(h_theta_centered.transpose())
#proposal_variance_gibbs = 0.095*proposal_variance_direct


#data_asis = np.load("data_asis.npy", allow_pickle=True)
#data_asis = data_asis.item()
#h_theta_asis = data_asis["h_theta"]
#proposal_variance_gibbs_asis = 0.3*(2.4/np.sqrt(6))*np.cov(h_theta_asis.transpose())
#proposal_variance_gibbs_nc_asis = 0.02*(2.4/np.sqrt(6))*np.cov(h_theta_asis.transpose())


#data_rescale = np.load("data_rescale.npy", allow_pickle=True)
#data_rescale = data_rescale.item()
#h_theta_rescale = data_rescale["h_theta"]
#proposal_variance_rescale = (2.4/np.sqrt(5))*np.cov(h_theta_rescale.transpose())


#data_pncp = np.load("data_pncp.npy", allow_pickle=True)
#data_pncp = data_pncp.item()
#h_theta_pncp = data_pncp["h_theta"]
#roposal_variance_pncp_gibbs = 0.05*(2.4/np.sqrt(5))*np.cov(h_theta_pncp.transpose())