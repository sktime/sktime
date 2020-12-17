# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: language_level=3


cimport cython

import numpy as np
cimport numpy as np
np.import_array()


cpdef np.ndarray acf(object x, const np.int32 [:] interval, int max_lag):
    cdef:
        int i, j, lag
        int n_instances = x.shape[0]
        int interval_length = interval[1] - interval[0]
        np.ndarray[np.float64, ndim=1] s1 = np.zeros((max_lag,))
        np.ndarray[np.float64, ndim=1] ss1 = np.zeros((max_lag,))
        np.ndarray[np.float64, ndim=2] acf_x = np.zeros((n_instances, max_lag))

    with nogil:
        for i in range(n_instances):

            # Pre-compute the prefix-sum array
            for j in range(interval_length):
                if j == 0:
                    s1[j] = x[i, j]
                    ss1[j] = np.square(x[i, j])
                else:
                    s1[j] = x[i, j] + s1[j-1]
                    ss1[j] = np.square(x[i, j]) + ss1[j-1]

            # Auto-correlation function transform
            for lag in range(1, max_lag+1):
                # Do it ourselves to avoid zero variance warnings


        