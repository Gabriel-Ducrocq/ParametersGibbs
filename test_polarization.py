import healpy as hp
import numpy as np
from classy import Class
import config



cosmo = Class()


def generate_cls(theta):
    params = {'output': config.OUTPUT_CLASS,
              'l_max_scalars': config.L_MAX_SCALARS,
              'lensing': config.LENSING}
    d = {name:val for name, val in zip(config.COSMO_PARAMS_NAMES, theta)}
    dd = {"tau_reio":0.0561}
    params.update(d)
    params.update(dd)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.raw_cl(config.L_MAX_SCALARS)
    #10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls["tt"] *= 2.7255e6**2
    cosmo.struct_cleanup()
    cosmo.empty()
    return cls["tt"]*2.7255e6**2, cls["ee"]*2.7255e6**2, cls["bb"]*2.7255e6**2, cls["te"]*2.7255e6**2


tt, ee, bb, te = generate_cls(config.COSMO_PARAMS_MEAN)

tt_map, ee_map, bb_map = hp.synalm((tt, ee, bb, te), lmax=config.L_MAX_SCALARS, new=True)