# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython MiniRocketMultivariate kernels.

Ahead-of-time compiled ports of sktime's ``_minirocket_multi_numba`` kernels
(``_fit_biases_multi`` and ``_transform_multi``). Same math, no numba JIT
warmup. Both consume float32/int32 arrays produced by the pure-numpy fit
scaffolding in ``MiniRocketMultivariateCython`` and are numerically equivalent
to the numba implementation (verified against it as groundtruth in tests).
"""

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport free, malloc

cnp.import_array()

# combinations(range(9), 3) -> 84 kernels, flattened to 252 int32.
cdef int[252] _IDX
cdef int _fill_idx():
    cdef int n = 0, i, j, k
    for i in range(9):
        for j in range(i + 1, 9):
            for k in range(j + 1, 9):
                _IDX[3 * n] = i
                _IDX[3 * n + 1] = j
                _IDX[3 * n + 2] = k
                n += 1
    return n
cdef int _NUM_KERNELS = _fill_idx()


def transform(
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] X,
    int[::1] num_channels_per_combination,
    int[::1] channel_indices,
    int[::1] dilations,
    int[::1] num_features_per_dilation,
    float[::1] biases,
):
    """Port of ``_transform_multi``. X is (n_instances, n_columns, n_timepoints)."""
    cdef int n_instances = X.shape[0]
    cdef int n_columns = X.shape[1]
    cdef int n_timepoints = X.shape[2]
    cdef float[:, :, ::1] Xv = X

    cdef int num_kernels = _NUM_KERNELS
    cdef int num_dilations = dilations.shape[0]

    cdef int total_fpd = 0
    cdef int d
    for d in range(num_dilations):
        total_fpd += num_features_per_dilation[d]
    cdef int num_features = num_kernels * total_fpd

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] features = np.zeros(
        (n_instances, num_features), dtype=np.float32
    )
    cdef float[:, ::1] feat = features

    cdef int csize = n_columns * n_timepoints
    # Per-instance work buffers (single-threaded; reused across dilations).
    cdef float* C_alpha = <float*>malloc(csize * sizeof(float))
    cdef float* C_gamma = <float*>malloc(9 * csize * sizeof(float))
    cdef float* C = <float*>malloc(n_timepoints * sizeof(float))
    if C_alpha == NULL or C_gamma == NULL or C == NULL:
        free(C_alpha); free(C_gamma); free(C)
        raise MemoryError()

    cdef int ex, di, ki, g, c, t, ch
    cdef int padding, padding0, padding1, nfd, dilation
    cdef int start, end, base, combination_index
    cdef int fis, fie, ncc, ncs, nce
    cdef int i0, i1, i2, fc, n_valid, lo, hi
    cdef float bias, x, count

    try:
      with nogil:
        for ex in range(n_instances):
            fis = 0
            ncs = 0
            combination_index = 0
            for di in range(num_dilations):
                padding0 = di % 2
                dilation = dilations[di]
                padding = ((9 - 1) * dilation) // 2
                nfd = num_features_per_dilation[di]

                # C_alpha = A = -X ; C_gamma = 0 ; C_gamma[4] = G = 3X
                for c in range(n_columns):
                    base = c * n_timepoints
                    for t in range(n_timepoints):
                        x = Xv[ex, c, t]
                        C_alpha[base + t] = -x
                        C_gamma[4 * csize + base + t] = x + x + x
                for g in range(9):
                    if g == 4:
                        continue
                    for c in range(n_columns):
                        base = g * csize + c * n_timepoints
                        for t in range(n_timepoints):
                            C_gamma[base + t] = 0.0

                # gamma_index 0..3
                end = n_timepoints - padding
                for g in range(4):
                    if end > 0:
                        for c in range(n_columns):
                            base = c * n_timepoints
                            for t in range(end):
                                C_alpha[base + n_timepoints - end + t] += -Xv[ex, c, t]
                                C_gamma[g * csize + base + n_timepoints - end + t] = (
                                    3.0 * Xv[ex, c, t]
                                )
                    end += dilation

                # gamma_index 5..8
                start = dilation
                for g in range(5, 9):
                    if start < n_timepoints:
                        for c in range(n_columns):
                            base = c * n_timepoints
                            for t in range(n_timepoints - start):
                                C_alpha[base + t] += -Xv[ex, c, start + t]
                                C_gamma[g * csize + base + t] = (
                                    3.0 * Xv[ex, c, start + t]
                                )
                    start += dilation

                for ki in range(num_kernels):
                    fie = fis + nfd
                    ncc = num_channels_per_combination[combination_index]
                    nce = ncs + ncc
                    padding1 = (padding0 + ki) % 2
                    i0 = _IDX[3 * ki]
                    i1 = _IDX[3 * ki + 1]
                    i2 = _IDX[3 * ki + 2]

                    # C[t] = sum over the combination's channels
                    for t in range(n_timepoints):
                        C[t] = 0.0
                    for ch in range(ncs, nce):
                        c = channel_indices[ch]
                        base = c * n_timepoints
                        for t in range(n_timepoints):
                            C[t] += (
                                C_alpha[base + t]
                                + C_gamma[i0 * csize + base + t]
                                + C_gamma[i1 * csize + base + t]
                                + C_gamma[i2 * csize + base + t]
                            )

                    if padding1 == 0:
                        lo = 0
                        hi = n_timepoints
                    else:
                        lo = padding
                        hi = n_timepoints - padding
                    n_valid = hi - lo

                    if n_valid > 0:
                        for fc in range(nfd):
                            bias = biases[fis + fc]
                            count = 0.0
                            for t in range(lo, hi):
                                if C[t] > bias:
                                    count += 1.0
                            feat[ex, fis + fc] = count / n_valid

                    fis = fie
                    combination_index += 1
                    ncs = nce
    finally:
        free(C_alpha)
        free(C_gamma)
        free(C)

    return features


def fit_biases(
    cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] X,
    int[::1] num_channels_per_combination,
    int[::1] channel_indices,
    int[::1] dilations,
    int[::1] num_features_per_dilation,
    int[::1] instance_indices,
):
    """Build per-combination convolution output C, summed over channels.

    Port of the convolution-building half of ``_fit_biases_multi``. Returns a
    ``(num_combinations, n_timepoints)`` float32 array; the caller applies
    ``np.quantile`` per combination to obtain biases. ``instance_indices`` holds
    the ``np.random.randint(n_instances)`` draw for each combination, computed by
    the caller to reproduce the numba random sequence exactly.
    """
    cdef int n_timepoints = X.shape[2]
    cdef float[:, :, ::1] Xv = X

    cdef int num_kernels = _NUM_KERNELS
    cdef int num_dilations = dilations.shape[0]
    cdef int num_combinations = num_kernels * num_dilations

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] out = np.zeros(
        (num_combinations, n_timepoints), dtype=np.float32
    )
    cdef float[:, ::1] outv = out

    # 9 gamma planes x n_timepoints for a single (instance, channel) row.
    cdef float* C_alpha = <float*>malloc(n_timepoints * sizeof(float))
    cdef float* C_gamma = <float*>malloc(9 * n_timepoints * sizeof(float))
    if C_alpha == NULL or C_gamma == NULL:
        free(C_alpha); free(C_gamma)
        raise MemoryError()

    cdef int di, ki, g, t, ch, ex, c
    cdef int padding, dilation, start, end
    cdef int comb, ncs, nce, ncc
    cdef int i0, i1, i2
    cdef float x

    try:
        comb = 0
        ncs = 0
        for di in range(num_dilations):
            dilation = dilations[di]
            padding = ((9 - 1) * dilation) // 2

            for ki in range(num_kernels):
                ncc = num_channels_per_combination[comb]
                nce = ncs + ncc
                ex = instance_indices[comb]
                i0 = _IDX[3 * ki]
                i1 = _IDX[3 * ki + 1]
                i2 = _IDX[3 * ki + 2]

                for t in range(n_timepoints):
                    outv[comb, t] = 0.0

                # Accumulate C = C_alpha + C_gamma[i0,i1,i2], summed over channels.
                for ch in range(ncs, nce):
                    c = channel_indices[ch]

                    # C_alpha = A = -X ; C_gamma = 0 ; C_gamma[4] = G = 3X
                    for t in range(n_timepoints):
                        x = Xv[ex, c, t]
                        C_alpha[t] = -x
                        C_gamma[4 * n_timepoints + t] = x + x + x
                    for g in range(9):
                        if g == 4:
                            continue
                        for t in range(n_timepoints):
                            C_gamma[g * n_timepoints + t] = 0.0

                    end = n_timepoints - padding
                    for g in range(4):
                        if end > 0:
                            for t in range(end):
                                C_alpha[n_timepoints - end + t] += -Xv[ex, c, t]
                                C_gamma[g * n_timepoints + n_timepoints - end + t] = (
                                    3.0 * Xv[ex, c, t]
                                )
                        end += dilation

                    start = dilation
                    for g in range(5, 9):
                        if start < n_timepoints:
                            for t in range(n_timepoints - start):
                                C_alpha[t] += -Xv[ex, c, start + t]
                                C_gamma[g * n_timepoints + t] = (
                                    3.0 * Xv[ex, c, start + t]
                                )
                        start += dilation

                    for t in range(n_timepoints):
                        outv[comb, t] += (
                            C_alpha[t]
                            + C_gamma[i0 * n_timepoints + t]
                            + C_gamma[i1 * n_timepoints + t]
                            + C_gamma[i2 * n_timepoints + t]
                        )

                comb += 1
                ncs = nce
    finally:
        free(C_alpha)
        free(C_gamma)

    return out
