from classy import Class
import config
import numpy as np
from numba import prange, njit, complex128

cosmo = Class()


def generate_cls(theta, pol = True):
    params = {'output': config.OUTPUT_CLASS,
              "modes":"s,t",
              "r":0.001,
              'l_max_scalars': config.L_MAX_SCALARS,
              'lensing': config.LENSING}
    d = {name:val for name, val in zip(config.COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(config.L_MAX_SCALARS)
    # 10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls_tt = cls["tt"]*2.7255e6**2
    if not pol:
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt
    else:
        cls_ee = cls["ee"]*2.7255e6**2
        cls_bb = cls["bb"]*2.7255e6**2
        cls_te = cls["te"]*2.7255e6**2
        cosmo.struct_cleanup()
        cosmo.empty()
        return cls_tt, cls_ee, cls_bb, cls_te


def generate_var_cls(cls):
    result = generate_var_cl_cython_sph(cls)
    return np.asarray(result)



@njit(parallel=True)
def matrix_product(dls_, b):
    """

    :param dls_: dls in m major
    :param b: vector of complex alms in m major
    :return: the product of the (inverse) signal covariance matrix and the vector of alms.
    """
    complex_dim = int((config.L_MAX_SCALARS + 1) * (config.L_MAX_SCALARS + 2) / 2)
    alms_shape = np.zeros((complex_dim, 3, 3), dtype=complex128)
    result = np.zeros((complex_dim, 3), dtype=complex128)

    for l in prange(config.L_MAX_SCALARS + 1):
        for m in range(l + 1):
            idx = m * (2 * config.L_MAX_SCALARS + 1 - m) // 2 + l
            if l == 0:
                alms_shape[idx, :, :] = dls_[l, :, :]
            else:
                alms_shape[idx, :, :] = dls_[l, :, :]

    for i in prange(0, complex_dim):
        result[i, :] = np.dot(alms_shape[i, :, :], b[i, :])

    return result


@njit()
def invert_2x2_matrix(m):
    """

    :param m: 2x2 invertible real matrix to invert
    :return: inverse of m
    """
    det = m[0, 0]*m[1, 1] - m[0, 1]*m[1, 0]
    inv_m = np.zeros(m.shape)
    inv_m[0, 0] = m[1, 1]
    inv_m[1, 1] = m[0, 0]
    inv_m[1, 0] = -m[1, 0]
    inv_m[0, 1] = -m[0, 1]
    inv_m /= det
    return inv_m


@njit()
def invert_signal_3x3_matrix(m):
    """

    :param m: 3x3 signal block covariance matrix. First block is 2x2 invertible covariance of Cl_TT, Cl_TE, Cl_EE. Second is Cl_BB.
    :return: inverse of m
    """
    inv_m = np.zeros(m.shape)
    inv_first_block = invert_2x2_matrix(m[:2, :2])
    inv_m[:2, :2] = inv_first_block
    inv_m[2, 2] = 1/m[2, 2]
    return inv_m


@njit()
def cholesky_2x2_matrix(m):
    """

    :param m: 2x2 symmetric positive-definite matrix
    :return: Lower triangle matrix in the Cholesky decomposition of m
    """
    cholesky_m = np.zeros(m.shape)
    cholesky_m[0, 0] = np.sqrt(m[0, 0])
    cholesky_m[1, 0] = m[1, 0]/np.sqrt(m[0, 0])
    cholesky_m[1, 1] = np.sqrt(m[1, 1] - cholesky_m[1, 0]**2)
    return cholesky_m


@njit()
def compute_cholesky(m):
    """

    :param m: 3x3 signal_block covariance matrix. First block is 2x2 invertible covariance of Cl_TT, Cl_TE, Cl_EE. Second is Cl_BB.
    :return: Lower triangle matrix in the Cholesky decomposition of m
    """
    cholesky_m = np.zeros(m.shape)
    cholesky_m[:2, :2] = cholesky_2x2_matrix(m[:2, :2])
    cholesky_m[2, 2] = np.sqrt(m[2, 2])
    return cholesky_m


@njit(parallel=False)
def compute_inverse_and_cholesky_constraint_realization(all_cls, add_term=None):
    """

    :param all_cls: LMAX+1 blocks of the signal covariance matrix
    :param add_term: LMAX + 1 additive terms for e.g pixel part in constraint realization step with no mask
    :return: the inverse of the all_cls^(-1) + add_term
    """
    inv_cls = np.zeros((len(all_cls), 3, 3))
    if add_term is not None:
        chol_inv_cls = np.zeros((len(all_cls), 3, 3))
        for i in prange(2):
            inv_cls[i, :, :] = np.diag(1/add_term[i, :])
            chol_inv_cls[i, :, :] = np.diag(np.sqrt(1/add_term[i, :]))
        for i in prange(2, len(all_cls)):
            inv_cls[i, :, :] = invert_signal_3x3_matrix(invert_signal_3x3_matrix(all_cls[i, :, :]) + np.diag(add_term[i, :]))
            chol_inv_cls[i, :, :] = compute_cholesky(inv_cls[i, :, :])

        return inv_cls, chol_inv_cls

    else:
        chol_inv_cls = np.zeros((len(all_cls), 3, 3))
        for i in prange(2, len(all_cls)):
            inv_cls[i, :, :] = invert_signal_3x3_matrix(all_cls[i, :, :])
            chol_inv_cls[i, :, :] = compute_cholesky(inv_cls[i, :, :])

        return inv_cls, chol_inv_cls

@njit()
def compute_determinant_2x2_matrix(m):
    return m[0, 0]*m[1, 1] - m[1, 0]*m[0, 1]


@njit()
def compute_determinant_3x3_matrix(m):
    ##Compute the determinant of 3x3 matrix such that m[2, 0] = m[2, 1] = m[1, 2] = m[2, 0] = 0
    det_2x2 = compute_determinant_2x2_matrix(m[:2, :2])
    return det_2x2*m[2, 2]


@njit(parallel=True)
def compute_log_determinant(all_cls):
    det = np.zeros(len(all_cls)-2)
    for i in prange(2, len(all_cls)):
        det[i-2] = compute_determinant_3x3_matrix(all_cls[i, :, :])

    return np.sum(np.log(det))