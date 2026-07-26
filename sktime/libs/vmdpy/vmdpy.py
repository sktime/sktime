"""Python implementation of the Variational Mode Decomposition method."""

import math

import numpy as np

__author__ = ["vrcarva", "LorchZachery", "danferns"]


def VMD(f, alpha, tau, K, DC, init, tol):
    """Variational mode decomposition.

    Python implementation by Vinícius Rezende Carvalho - vrcarva
    code based on Dominique Zosso's MATLAB code, available at:
    https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition

    Original paper: Dragomiretskiy, K., & Zosso, D. (2014) [1]_.

    Parameters
    ----------
    f : array_like
        the time domain signal (1D) to be decomposed
    alpha : float
        the balancing parameter of the data-fidelity constraint
    tau : float
        time-step of the dual ascent ( pick 0 for noise-slack )
    K : int
        the number of modes to be recovered
    DC : bool
        true if the first mode is put and kept at DC (0-freq)
    init : int
        0 = all omegas start at 0
        1 = all omegas start uniformly distributed
        2 = all omegas initialized randomly
    tol : float
        tolerance of convergence criterion; typically around 1e-6

    Returns
    -------
    u : array_like
        the collection of decomposed modes
    u_hat : array_like
        spectra of the modes
    omega : array_like
        estimated mode center-frequencies

    References
    ----------
    .. [1] K. Dragomiretskiy and D. Zosso, - Variational Mode Decomposition:
        IEEE Transactions on Signal Processing, vol. 62, no. 3, pp. 531-544, Feb.1,
        2014, doi: 10.1109/TSP.2013.2288675.
    """  # noqa: E501
    f = np.array(f)
    # Period and sampling frequency of input signal
    fs = 1.0 / len(f)

    T = len(f)
    fMirr = np.array(np.flip(f[0 : math.ceil(T / 2)]))
    fMirr = np.append(fMirr, f)
    fMirr = np.append(fMirr, np.flip(f[math.ceil(T / 2) :]))

    # Time Domain 0 to T (of mirrored signal)
    T = len(fMirr)
    t = np.arange(1, T + 1) / T

    # Spectral Domain discretization
    freqs = t - 0.5 - (1 / T)

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    Niter = 500
    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K)

    # Construct and center f_hat
    f_hat = np.fft.fftshift(np.fft.fft(fMirr))
    f_hat_plus = np.copy(f_hat)  # copy f_hat
    f_hat_plus[: T // 2] = 0

    # Initialization of omega_k
    omega_plus = np.zeros([Niter, K])

    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * i
    elif init == 2:
        omega_plus[0, :] = np.sort(
            np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(1, K))
        )
    else:
        omega_plus[0, :] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros([Niter, len(freqs)], dtype=complex)

    # other inits
    uDiff = tol + np.spacing(1)  # update step
    n = 0  # loop counter
    sum_uk = 0  # accumulator
    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros([Niter, len(freqs), K], dtype=complex)

    # *** Main loop for iterative updates***

    while uDiff > tol and n < Niter - 1:  # not converged and below iterations limit
        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]

        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus_enumerator = f_hat_plus - sum_uk - lambda_hat[n, :] / 2
        u_hat_plus_denominator = 1.0 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2
        u_hat_plus[n + 1, :, k] = u_hat_plus_enumerator / u_hat_plus_denominator

        # update first omega if not held at 0
        if not DC:
            wts = abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2
            omega_plus[n + 1, k] = _safe_average(freqs[T // 2 : T], weights=wts)

        # update of any other mode
        for k in np.arange(1, K):
            # accumulator
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            # mode spectrum
            u_hat_plus_enumerator = f_hat_plus - sum_uk - lambda_hat[n, :] / 2
            u_hat_plus_denominator = 1.0 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2
            u_hat_plus[n + 1, :, k] = u_hat_plus_enumerator / u_hat_plus_denominator
            # center frequencies
            wts = abs(u_hat_plus[n + 1, T // 2 : T, k]) ** 2
            omega_plus[n + 1, k] = _safe_average(freqs[T // 2 : T], weights=wts)

        # Dual ascent
        lambda_update = tau * (np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus)
        lambda_hat[n + 1, :] = lambda_hat[n, :] + lambda_update

        # loop counter
        n = n + 1

        # converged yet?
        uDiff = np.spacing(1)
        for i in range(K):
            dot_left = u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]
            dot_right = np.conj(u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])
            uDiff = uDiff + (1 / T) * np.dot(dot_left, dot_right)

        uDiff = np.abs(uDiff)

    # Postprocessing and cleanup

    # discard empty space if converged early
    Niter = np.min([Niter, n])
    omega = omega_plus[:Niter, :]

    idxs = np.flip(np.arange(1, T // 2 + 1), axis=0)

    # Signal reconstruction
    u_hat = np.zeros([T, K], dtype=complex)
    u_hat[T // 2 : T, :] = u_hat_plus[Niter - 1, T // 2 : T, :]
    u_hat[idxs, :] = np.conj(u_hat_plus[Niter - 1, T // 2 : T, :])
    u_hat[0, :] = np.conj(u_hat[-1, :])

    u = np.zeros([K, len(t)])
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # remove mirror part
    u = u[:, T // 4 : 3 * T // 4]

    # recompute spectrum
    u_hat = np.zeros([u.shape[1], K], dtype=complex)
    for k in range(K):
        u_hat[:, k] = np.fft.fftshift(np.fft.fft(u[k, :]))

    return u, u_hat, omega


def _safe_average(x, weights):
    """Compute the weighted average of x, with safe handling of zero weights.

    If not all weights are zero, the function returns the weighted arithmetic mean of x,
    equivalent to ``np.average(x, weights=weights)``.
    If all weights are zero, the function returns the unweighted arithmetic mean of x.

    Parameters
    ----------
    x : 1D array_like
        the data to be averaged
    weights : 1D array_like
        the weights for the average

    Returns
    -------
    average : float
        the weighted average of x
    """
    wtsum = np.sum(weights)
    if wtsum == 0:
        return np.mean(x)
    else:
        return np.dot(x, weights) / wtsum
