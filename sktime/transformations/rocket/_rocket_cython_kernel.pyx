# cython: language_level=3
import numpy as np
cimport numpy as cnp

def multiply_accumulate(cnp.ndarray[cnp.float32_t, ndim=1] x, float scalar):
    """Simple calculation kernel for MiniRocketCython prototype."""
    cdef int n = x.shape[0]
    cdef int i
    cdef float acc = 0.0
    for i in range(n):
        acc += x[i] * scalar
    return acc
