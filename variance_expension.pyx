
import numpy as np


cpdef double[:] generate_var_cl_cython_sph(double[:] cls_):
    cdef int L_max, size, l, m, i
    L_max = len(cls_)+1
    size = (L_max + 1)**2 - 4
    i = 0
    cdef double[:] variance = np.zeros(size)
    for l in range(0, L_max-1):
        for m in range(2*(l+2)+1):
            if m == 0:
                variance[i] = cls_[l]
            else:
                variance[i] = cls_[l]/2

            i +=1

    return variance