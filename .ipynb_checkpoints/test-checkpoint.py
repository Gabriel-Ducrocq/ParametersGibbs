import utils
import config
import time
import healpy as hp
import camb
import numpy as np
import matplotlib.pyplot as plt


def generate_cls(theta):
    ns, omega_b, omega_cdm, H0, ln_A_s = theta
    As = np.exp(ln_A_s)*10**(-10)
    tau_reio = 0.0561
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=omega_b, omch2=omega_cdm, tau=tau_reio)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(config.L_MAX_SCALARS)

    print("Computation time:")
    start = time.time()
    results = camb.get_results(pars)
    pow_spec = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    end = time.time()
    print(end-start)
    pow_spec = pow_spec["total"][2:config.L_MAX_SCALARS+1, 0]
    pow_spec *= config.scale
    return pow_spec


times_class = []
times_camb = []
times_synthesis = []
times_analysis = []

for i in range(1000):
    if i % 10 == 0:
        print(i)
        
    start = time.clock()
    cls = utils.generate_cls(config.COSMO_PARAMS_MEAN)
    end = time.clock()
    print("Time CLASS generation:")
    duration = end - start
    times_class.append(duration)
    print(end - start)

    start = time.clock()
    cls_camb = generate_cls(config.COSMO_PARAMS_MEAN)
    end = time.clock()
    print("Time CAMB generation:")
    duration = end - start
    times_camb.append(duration)
    print(end-start)

    s = hp.synalm(cls, lmax=config.L_MAX_SCALARS)
    start = time.clock()
    d = hp.alm2map(s, nside=config.NSIDE, lmax=config.L_MAX_SCALARS)
    end = time.clock()
    print("Time synthesis")
    duration = end - start
    times_synthesis.append(duration)
    print(end-start)


    start = time.clock()
    alm = hp.map2alm(d, lmax=config.L_MAX_SCALARS)
    end = time.clock()
    print("Time analysis")
    duration = end - start
    times_analysis.append(duration)
    print(end-start)

    
    
plt.hist(times_class, density=True, alpha=0.5, label="CLASS")
plt.hist(times_camb, density=True, alpha=0.5, label="CAMB")
plt.hist(times_synthesis, density=True, alpha=0.5, label="synthesis")
plt.hist(times_analysis, density=True, alpha=0.5, label="analysis")
plt.legend(loc="upper right")
plt.savefig("Durations_comparisons.png")
plt.close()


