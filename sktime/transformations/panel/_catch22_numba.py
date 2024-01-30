"""Isolated numba imports for catch22."""

__author__ = ["MatthewMiddlehurst"]

import math

import numpy as np

from sktime.utils.numba.njit import njit


@njit(fastmath=True, cache=True)
def _histogram_mode(X, num_bins, smin, smax):
    bin_width = (smax - smin) / num_bins

    if bin_width == 0:
        return np.nan

    histogram = np.zeros(num_bins)
    for val in X:
        idx = int((val - smin) / bin_width)
        idx = num_bins - 1 if idx >= num_bins else idx
        histogram[idx] += 1

    edges = np.zeros(num_bins + 1, dtype=np.float32)
    for i in range(len(edges)):
        edges[i] = i * bin_width + smin

    max_count = 0
    num_maxs = 1
    max_sum = 0
    for i in range(num_bins):
        v = (edges[i] + edges[i + 1]) / 2
        if histogram[i] > max_count:
            max_count = histogram[i]
            num_maxs = 1
            max_sum = v
        elif histogram[i] == max_count:
            num_maxs += 1
            max_sum += v

    return max_sum / num_maxs


@njit(fastmath=True, cache=True)
def _long_stretch(X_binary, val):
    last_val = 0
    max_stretch = 0
    for i in range(len(X_binary)):
        if X_binary[i] != val or i == len(X_binary) - 1:
            stretch = i - last_val
            if stretch > max_stretch:
                max_stretch = stretch
            last_val = i

    return max_stretch


@njit(fastmath=True, cache=True)
def _outlier_include(X):
    total = 0
    threshold = 0
    for v in X:
        if v >= 0:
            total += 1
            if v > threshold:
                threshold = v

    if threshold < 0.01:
        return 0

    num_thresholds = int(threshold / 0.01) + 1
    means = np.zeros(num_thresholds)
    dists = np.zeros(num_thresholds)
    medians = np.zeros(num_thresholds)
    for i in range(num_thresholds):
        d = i * 0.01

        count = 0
        r = np.zeros(len(X))
        for n in range(len(X)):
            if X[n] >= d:
                r[count] = n + 1
                count += 1

        if count == 0:
            continue

        diff = np.zeros(count - 1)
        for n in range(len(diff)):
            diff[n] = r[n + 1] - r[n]

        means[i] = np.mean(diff) if len(diff) > 0 else 9999999999
        dists[i] = len(diff) * 100 / total
        medians[i] = np.median(r[:count]) / (len(X) / 2) - 1

    mj = 0
    fbi = num_thresholds - 1
    for i in range(num_thresholds):
        if dists[i] > 2:
            mj = i
        if means[i] == 9999999999:
            fbi = num_thresholds - 1 - i

    trim_limit = max(mj, fbi)

    return np.median(medians[: trim_limit + 1])


def _autocorr(X, X_fft):
    ca = np.fft.ifft(_multiply_complex_arr(X_fft))
    return _get_acf(X, ca)


@njit(fastmath=True, cache=True)
def _multiply_complex_arr(X_fft):
    c = np.zeros(len(X_fft), dtype=np.complex128)
    for i, n in enumerate(X_fft):
        c[i] = n * (n.real + 1j * -n.imag)
    return c


@njit(fastmath=True, cache=True)
def _get_acf(X, ca):
    acf = np.zeros(len(X))
    if ca[0].real != 0:
        for i in range(len(X)):
            acf[i] = ca[i].real / ca[0].real
    return acf


@njit(fastmath=True, cache=True)
def _summaries_welch_rect(X, centroid, X_fft):
    new_length = int(len(X_fft) / 2) + 1
    p = np.zeros(new_length)
    pi2 = 2 * math.pi
    p[0] = (np.power(_complex_magnitude(X_fft[0]), 2) / len(X)) / pi2
    for i in range(1, new_length - 1):
        p[i] = ((np.power(_complex_magnitude(X_fft[i]), 2) / len(X)) * 2) / pi2
    p[new_length - 1] = (
        np.power(_complex_magnitude(X_fft[new_length - 1]), 2) / len(X)
    ) / pi2

    w = np.zeros(new_length)
    a = 1.0 / len(X_fft)
    for i in range(0, new_length):
        w[i] = i * a * math.pi * 2

    if centroid:
        cs = np.zeros(new_length)
        cs[0] = p[0]
        for i in range(1, new_length):
            cs[i] = cs[i - 1] + p[i]

        threshold = cs[new_length - 1] / 2
        for i in range(1, new_length):
            if cs[i] > threshold:
                return w[i]
        return np.nan
    else:
        tau = int(np.floor(new_length / 5))
        nsum = 0
        for i in range(tau):
            nsum += p[i]

        return nsum * (w[1] - w[0])


@njit(fastmath=True, cache=True)
def _complex_magnitude(c):
    return np.sqrt(c.real * c.real + c.imag * c.imag)


@njit(fastmath=True, cache=True)
def _local_simple_mean(X, train_length):
    res = np.zeros(len(X) - train_length)
    for i in range(len(res)):
        nsum = 0
        for n in range(train_length):
            nsum += X[i + n]
        res[i] = X[i + train_length] - nsum / train_length
    return res


@njit(fastmath=True, cache=True)
def _ac_first_zero(X_ac):
    for i in range(1, len(X_ac)):
        if X_ac[i] <= 0:
            return i

    return len(X_ac)


@njit(fastmath=True, cache=True)
def _fluct_prop(X, og_length, dfa):
    a = np.zeros(50, dtype=np.int_)
    a[0] = 5
    n_tau = 1
    smin = 1.6094379124341003  # Math.log(5);
    smax = np.log(og_length / 2)
    inc = (smax - smin) / 49
    for i in range(1, 50):
        val = int(np.round(np.exp(smin + inc * i) + 0.000000000001))
        if val != a[n_tau - 1]:
            a[n_tau] = val
            n_tau += 1

    if n_tau < 12:
        return np.nan

    f = np.zeros(n_tau)
    for i in range(n_tau):
        tau = a[i]
        buff_size = int(len(X) / tau)
        lag = 0
        if buff_size == 0:
            buff_size = 1
            lag = 1

        buffer = np.zeros((buff_size, tau))
        count = 0
        for n in range(buff_size):
            for j in range(tau - lag):
                buffer[n][j] = X[count]
                count += 1

        d = np.zeros(tau)
        for n in range(tau):
            d[n] = n + 1

        for n in range(buff_size):
            c1, c2 = _linear_regression(d, buffer[n], tau, 0)

            for j in range(tau):
                buffer[n][j] = buffer[n][j] - (c1 * (j + 1) + c2)

            if dfa:
                for j in range(tau):
                    f[i] += buffer[n][j] * buffer[n][j]
            else:
                f[i] += np.power(np.max(buffer[n]) - np.min(buffer[n]), 2)

        if dfa:
            f[i] = np.sqrt(f[i] / (buff_size * tau))
        else:
            f[i] = np.sqrt(f[i] / buff_size)

    log_a = np.zeros(n_tau)
    log_f = np.zeros(n_tau)
    for i in range(n_tau):
        log_a[i] = np.log(a[i])
        log_f[i] = np.log(f[i])

    sserr = np.zeros(n_tau - 11)
    for i in range(6, n_tau - 5):
        c1_1, c1_2 = _linear_regression(log_a, log_f, i, 0)
        c2_1, c2_2 = _linear_regression(log_a, log_f, n_tau - i + 1, i - 1)

        sum1 = 0
        for n in range(i):
            sum1 += np.power(log_a[n] * c1_1 + c1_2 - log_f[n], 2)
        sserr[i - 6] += np.sqrt(sum1)

        sum2 = 0
        for n in range(n_tau - i + 1):
            sum2 += np.power(log_a[n + i - 1] * c2_1 + c2_2 - log_f[n + i - 1], 2)
        sserr[i - 6] += np.sqrt(sum2)

    return (np.argmin(sserr) + 6) / n_tau


@njit(fastmath=True, cache=True)
def _linear_regression(X, y, n, lag):
    sumx = 0
    sumx2 = 0
    sumxy = 0
    sumy = 0
    for i in range(lag, n + lag):
        sumx += X[i]
        sumx2 += X[i] * X[i]
        sumxy += X[i] * y[i]
        sumy += y[i]

    denom = n * sumx2 - sumx * sumx
    if denom == 0:
        return 0, 0

    return (n * sumxy - sumx * sumy) / denom, (sumy * sumx2 - sumx * sumxy) / denom


@njit(fastmath=True, cache=True)
def _spline_fit(X):
    breaks = np.array([0, len(X) / 2 - 1, len(X) - 1])
    h0 = np.array([breaks[1] - breaks[0], breaks[2] - breaks[1]])
    h_copy = np.array([h0[0], h0[1], h0[0], h0[1]])
    hl = np.array([h_copy[3], h_copy[2], h_copy[1]])
    hr = np.array([h_copy[0], h_copy[1], h_copy[2]])

    hlCS = np.zeros(3)
    hlCS[0] = hl[0]
    for i in range(1, 3):
        hlCS[i] = hlCS[i - 1] + hl[i]

    bl = np.zeros(3)
    for i in range(3):
        bl[i] = breaks[0] - hlCS[i]

    hrCS = np.zeros(3)
    hrCS[0] = hr[0]
    for i in range(1, 3):
        hrCS[i] = hrCS[i - 1] + hr[i]

    br = np.zeros(3)
    for i in range(3):
        br[i] = breaks[2] - hrCS[i]

    breaksExt = np.zeros(9)
    for i in range(3):
        breaksExt[i] = bl[2 - i]
        breaksExt[i + 3] = breaks[i]
        breaksExt[i + 6] = br[i]

    hExt = np.zeros(8)
    for i in range(8):
        hExt[i] = breaksExt[i + 1] - breaksExt[i]

    coeffs = np.zeros((32, 4))
    for i in range(0, 32, 4):
        coeffs[i][0] = 1

    ii = np.zeros((4, 8), dtype=np.int32)
    for i in range(8):
        ii[0][i] = i
        ii[1][i] = min(1 + i, 7)
        ii[2][i] = min(2 + i, 7)
        ii[3][i] = min(3 + i, 7)

    H = np.zeros(32)
    for i in range(32):
        H[i] = hExt[ii[i % 4][int(i / 4)]]

    for k in range(1, 4):
        for j in range(k):
            for u in range(32):
                coeffs[u][j] *= H[u] / (k - j)

        Q = np.zeros((4, 8))
        for u in range(32):
            for m in range(4):
                Q[u % 4][int(u / 4)] += coeffs[u][m]

        for u in range(8):
            for m in range(1, 4):
                Q[m][u] += Q[m - 1][u]

        for u in range(32):
            if u % 4 > 0:
                coeffs[u][k] = Q[u % 4 - 1][int(u / 4)]

        fmax = np.zeros(32)
        for i in range(8):
            for j in range(4):
                fmax[i * 4 + j] = Q[3][i]

        for j in range(k + 1):
            for u in range(32):
                coeffs[u][j] /= fmax[u]

        for i in range(29):
            for j in range(k + 1):
                coeffs[i][j] -= coeffs[3 + i][j]

        for i in range(0, 32, 4):
            coeffs[i][k] = 0

    scale = np.ones(32)
    for k in range(3):
        for i in range(32):
            scale[i] /= H[i]

        for i in range(32):
            coeffs[i][3 - (k + 1)] *= scale[i]

    jj = np.zeros((4, 2), dtype=np.int32)
    for i in range(4):
        for j in range(2):
            if i == 0:
                jj[i][j] = 4 * (1 + j)
            else:
                jj[i][j] = 3

    for i in range(1, 4):
        for j in range(2):
            jj[i][j] += jj[i - 1][j]

    coeffs_out = np.zeros((8, 4))
    for i in range(8):
        coeffs_out[i] = coeffs[jj[i % 4][int(i / 4)] - 1]

    xsB = np.zeros(len(X) * 4)
    indexB = np.zeros(len(xsB), dtype=np.int32)
    breakInd = 1
    for i in range(len(X)):
        if i >= breaks[1] and breakInd < 2:
            breakInd += 1

        for j in range(4):
            xsB[i * 4 + j] = i - breaks[breakInd - 1]
            indexB[i * 4 + j] = j + (breakInd - 1) * 4

    vB = np.zeros(len(xsB))
    for i in range(len(xsB)):
        vB[i] = coeffs_out[indexB[i]][0]

    for i in range(1, 4):
        for j in range(len(xsB)):
            vB[j] = vB[j] * xsB[j] + coeffs_out[indexB[j]][i]

    A = np.zeros(len(X) * 5)
    breakInd = 0
    for i in range(len(xsB)):
        if i / 4 >= breaks[1]:
            breakInd = 1
        A[i % 4 + breakInd + int(i / 4) * 5] = vB[i]

    AT = np.zeros(len(A))
    ATA = np.zeros(25)
    ATb = np.zeros(5)
    for i in range(len(X)):
        for j in range(5):
            AT[j * len(X) + i] = A[i * 5 + j]

    for i in range(5):
        for j in range(5):
            for k in range(len(X)):
                ATA[i * 5 + j] += AT[i * len(X) + k] * A[k * 5 + j]

    for i in range(5):
        for k in range(len(X)):
            ATb[i] += AT[i * len(X) + k] * X[k]

    AElim = np.zeros((5, 5))
    for i in range(5):
        n = i * 5
        AElim[i] = ATA[n : n + 5]

    for i in range(5):
        for j in range(i + 1, 5):
            factor = AElim[j][i] / AElim[i][i]
            ATb[j] = ATb[j] - factor * ATb[i]

            for k in range(i, 5):
                AElim[j][k] = AElim[j][k] - factor * AElim[i][k]

    x = np.zeros(5)
    for i in range(4, -1, -1):
        bMinusATemp = ATb[i]
        for j in range(i + 1, 5):
            bMinusATemp -= x[j] * AElim[i][j]

        x[i] = bMinusATemp / AElim[i][i]

    C = np.zeros((5, 8))
    for i in range(32):
        C[int(i % 4 + int(i / 4) % 2)][int(i / 4)] = coeffs_out[i % 8][int(i / 8)]

    coeffs_spline = np.zeros((2, 4))
    for j in range(8):
        coefc = int(j / 2)
        coefr = j % 2
        for i in range(5):
            coeffs_spline[coefr][coefc] += C[i][j] * x[i]

    y_out = np.zeros(len(X))
    for i in range(len(X)):
        second_half = 0 if i < breaks[1] else 1
        y_out[i] = coeffs_spline[second_half][0]

    for i in range(1, 4):
        for j in range(len(X)):
            second_half = 0 if j < breaks[1] else 1
            y_out[j] = (
                y_out[j] * (j - breaks[1] * second_half) + coeffs_spline[second_half][i]
            )

    return y_out


@njit(fastmath=True, cache=True)
def _normalise_series(X, mean):
    std = np.std(X)
    if std > 0:
        return (X - mean) / std
    return X


def _DN_HistogramMode_5(X, smin, smax):
    # Mode of z-scored distribution (5-bin histogram).
    return _histogram_mode(X, 5, smin, smax)


def _DN_HistogramMode_10(X, smin, smax):
    # Mode of z-scored distribution (10-bin histogram).
    return _histogram_mode(X, 10, smin, smax)


@njit(fastmath=True, cache=True)
def _SB_BinaryStats_diff_longstretch0(X, smean):
    # Longest period of consecutive values above the mean.
    mean_binary = np.zeros(len(X))
    for i in range(len(X)):
        if X[i] - smean > 0:
            mean_binary[i] = 1

    return _long_stretch(mean_binary, 1)


def _DN_OutlierInclude_p_001_mdrmd(X):
    # Time intervals between successive extreme events above the mean.
    return _outlier_include(X)


@njit(fastmath=True, cache=True)
def _DN_OutlierInclude_n_001_mdrmd(X):
    # Time intervals between successive extreme events below the mean.
    return _outlier_include(-X)


@njit(fastmath=True, cache=True)
def _CO_f1ecac(X_ac):
    # First 1/e crossing of autocorrelation function.
    threshold = 0.36787944117144233  # 1 / np.exp(1)
    for i in range(1, len(X_ac)):
        if (X_ac[i - 1] - threshold) * (X_ac[i] - threshold) < 0:
            return i
    return len(X_ac)


@njit(fastmath=True, cache=True)
def _CO_FirstMin_ac(X_ac):
    # First minimum of autocorrelation function.
    for i in range(1, len(X_ac) - 1):
        if X_ac[i] < X_ac[i - 1] and X_ac[i] < X_ac[i + 1]:
            return i
    return len(X_ac)


def _SP_Summaries_welch_rect_area_5_1(X, X_fft):
    # Total power in lowest fifth of frequencies in the Fourier power spectrum.
    return _summaries_welch_rect(X, False, X_fft)


def _SP_Summaries_welch_rect_centroid(X, X_fft):
    # Centroid of the Fourier power spectrum.
    return _summaries_welch_rect(X, True, X_fft)


@njit(fastmath=True, cache=True)
def _FC_LocalSimple_mean3_stderr(X):
    # Mean error from a rolling 3-sample mean forecasting.
    if len(X) - 3 < 3:
        return 0
    res = _local_simple_mean(X, 3)
    return np.std(res)


@njit(fastmath=True, cache=True)
def _CO_trev_1_num(X):
    # Time-reversibility statistic, ((x_t+1 − x_t)^3)_t.
    y = np.zeros(len(X) - 1)
    for i in range(len(y)):
        y[i] = np.power(X[i + 1] - X[i], 3)
    return np.mean(y)


@njit(fastmath=True, cache=True)
def _CO_HistogramAMI_even_2_5(X, smin, smax):
    # Automutual information, m = 2, τ = 5.
    new_min = smin - 0.1
    new_max = smax + 0.1
    bin_width = (new_max - new_min) / 5

    histogram = np.zeros((5, 5))
    sumx = np.zeros(5)
    sumy = np.zeros(5)
    v = 1.0 / (len(X) - 2)
    for i in range(len(X) - 2):
        idx1 = int((X[i] - new_min) / bin_width)
        idx2 = int((X[i + 2] - new_min) / bin_width)

        histogram[idx1][idx2] += v
        sumx[idx1] += v
        sumy[idx2] += v

    nsum = 0
    for i in range(5):
        for n in range(5):
            if histogram[i][n] > 0:
                nsum += histogram[i][n] * np.log(histogram[i][n] / sumx[i] / sumy[n])

    return nsum


@njit(fastmath=True, cache=True)
def _IN_AutoMutualInfoStats_40_gaussian_fmmi(X_ac):
    # First minimum of the automutual information function.
    tau = int(min(40, np.ceil(len(X_ac) / 2)))

    diffs = np.zeros(tau - 1)
    prev = -0.5 * np.log(1 - np.power(X_ac[1], 2))
    for i in range(len(diffs)):
        corr = -0.5 * np.log(1 - np.power(X_ac[i + 2], 2))
        diffs[i] = corr - prev
        prev = corr

    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i + 1] < 0 and diffs[i] < 0:
            return i + 1

    return tau


@njit(fastmath=True, cache=True)
def _MD_hrv_classic_pnn40(X):
    # Proportion of successive differences exceeding 0.04σ (Mietus 2002).
    diffs = np.zeros(len(X) - 1)
    for i in range(len(diffs)):
        diffs[i] = np.abs(X[i + 1] - X[i]) * 1000

    nsum = 0
    for diff in diffs:
        if diff > 40:
            nsum += 1

    return nsum / len(diffs)


@njit(fastmath=True, cache=True)
def _SB_BinaryStats_mean_longstretch1(X):
    # Longest period of successive incremental decreases.
    diff_binary = np.zeros(len(X) - 1)
    for i in range(len(diff_binary)):
        if X[i + 1] - X[i] >= 0:
            diff_binary[i] = 1

    return _long_stretch(diff_binary, 0)


@njit(fastmath=True, cache=True)
def _SB_MotifThree_quantile_hh(X):
    # Shannon entropy of two successive letters in equiprobable 3-letter
    # symbolization.
    indices = np.argsort(X)
    bins = np.zeros(len(X))
    q1 = int(len(X) / 3)
    q2 = q1 * 2
    l1 = np.zeros(q1, dtype=np.int_)
    for i in range(q1):
        l1[i] = indices[i]
    l2 = np.zeros(q1, dtype=np.int_)
    c1 = 0
    for i in range(q1, q2):
        bins[indices[i]] = 1
        l2[c1] = indices[i]
        c1 += 1
    l3 = np.zeros(len(indices) - q2, dtype=np.int_)
    c2 = 0
    for i in range(q2, len(indices)):
        bins[indices[i]] = 2
        l3[c2] = indices[i]
        c2 += 1

    found_last = False
    nsum = 0
    for i in range(3):
        if i == 0:
            o = l1
        elif i == 1:
            o = l2
        else:
            o = l3

        if not found_last:
            for n in range(len(o)):
                if o[n] == len(X) - 1:
                    o = np.delete(o, n)
                    break

        for n in range(3):
            nsum2 = 0

            for v in o:
                if bins[v + 1] == n:
                    nsum2 += 1

            if nsum2 > 0:
                nsum2 /= len(X) - 1
                nsum += nsum2 * np.log(nsum2)

    return -nsum


def _FC_LocalSimple_mean1_tauresrat(X, acfz):
    # Change in correlation length after iterative differencing.
    if len(X) < 2:
        return 0
    res = _local_simple_mean(X, 1)
    mean = np.mean(res)

    nfft = int(np.power(2, np.ceil(np.log(len(res)) / np.log(2))))
    fft = np.fft.fft(res - mean, n=nfft)
    ac = _autocorr(res, fft)

    return _ac_first_zero(ac) / acfz


@njit(fastmath=True, cache=True)
def _CO_Embed2_Dist_tau_d_expfit_meandiff(X, acfz):
    # Exponential fit to successive distances in 2-d embedding space.
    tau = acfz
    if tau > len(X) / 10:
        tau = int(len(X) / 10)

    d = np.zeros(len(X) - tau - 1)
    d_mean = 0
    for i in range(len(d)):
        n = np.sqrt(
            np.power(X[i + 1] - X[i], 2) + np.power(X[i + tau + 1] - X[i + tau], 2)
        )
        d[i] = n
        d_mean += n
    d_mean /= len(X) - tau - 1

    smin = np.min(d)
    smax = np.max(d)
    srange = smax - smin
    std = np.std(d)

    if std == 0:
        return np.nan

    num_bins = int(
        np.ceil(srange / (3.5 * np.std(d) / np.power(len(d), 0.3333333333333333)))
    )

    if num_bins == 0:
        return np.nan
    bin_width = srange / num_bins

    histogram = np.zeros(num_bins)
    for val in d:
        idx = int((val - smin) / bin_width)
        if idx >= num_bins:
            idx = num_bins - 1
        histogram[idx] += 1

    sum = 0
    for i in range(num_bins):
        center = ((smin + bin_width * i) * 2 + bin_width) / 2
        n = np.exp(-center / d_mean) / d_mean
        if n < 0:
            n = 0

        sum += np.abs(histogram[i] / len(d) - n)

    return sum / num_bins


@njit(fastmath=True, cache=True)
def _SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(X):
    # Proportion of slower timescale fluctuations that scale with DFA (50%
    # sampling).
    cs = np.zeros(int(len(X) / 2))
    cs[0] = X[0]
    for i in range(1, len(cs)):
        cs[i] = cs[i - 1] + X[i * 2]

    return _fluct_prop(cs, len(X), True)


@njit(fastmath=True, cache=True)
def _SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(X):
    # Proportion of slower timescale fluctuations that scale with linearly rescaled
    # range fits.
    cs = np.zeros(len(X))
    cs[0] = X[0]
    for i in range(1, len(X)):
        cs[i] = cs[i - 1] + X[i]

    return _fluct_prop(cs, len(X), False)


@njit(fastmath=True, cache=True)
def _SB_TransitionMatrix_3ac_sumdiagcov(X, acfz):
    # Trace of covariance of transition matrix between symbols in 3-letter
    # alphabet.
    ds = np.zeros(int((len(X) - 1) / acfz + 1))
    for i in range(len(ds)):
        ds[i] = X[i * acfz]
    indices = np.argsort(ds)

    bins = np.zeros(len(ds), dtype=np.int32)
    q1 = int(len(ds) / 3)
    q2 = q1 * 2
    for i in range(q1 + 1, q2 + 1):
        bins[indices[i]] = 1
    for i in range(q2 + 1, len(indices)):
        bins[indices[i]] = 2

    t = np.zeros((3, 3))
    for i in range(len(ds) - 1):
        t[bins[i + 1]][bins[i]] += 1
    t /= len(ds) - 1

    means = np.zeros(3)
    for i in range(3):
        means[i] = np.mean(t[i])

    cov = np.zeros((3, 3))
    for i in range(3):
        for n in range(3):
            covariance = 0
            for j in range(3):
                covariance += (t[i][j] - means[i]) * (t[n][j] - means[n])
            covariance /= 2

            cov[i][n] = covariance
            cov[n][i] = covariance

    ssum = 0
    for i in range(3):
        ssum += cov[i][i]

    return ssum


@njit(fastmath=True, cache=True)
def _PD_PeriodicityWang_th0_01(X):
    # Periodicity measure of (Wang et al. 2007).
    y_spline = _spline_fit(X)

    y_sub = np.zeros(len(X))
    for i in range(len(X)):
        y_sub[i] = X[i] - y_spline[i]

    acmax = int(np.ceil(len(X) / 3.0))
    acf = np.zeros(acmax)
    for tau in range(1, acmax + 1):
        covariance = 0
        for i in range(len(X) - tau):
            covariance += y_sub[i] * y_sub[i + tau]
        acf[tau - 1] = covariance / (len(X) - tau)

    troughs = np.zeros(acmax, dtype=np.int32)
    peaks = np.zeros(acmax, dtype=np.int32)
    n_troughs = 0
    n_peaks = 0
    for i in range(1, acmax - 1):
        slope_in = acf[i] - acf[i - 1]
        slope_out = acf[i + 1] - acf[i]

        if slope_in < 0 and slope_out > 0:
            troughs[n_troughs] = i
            n_troughs += 1
        elif slope_in > 0 and slope_out < 0:
            peaks[n_peaks] = i
            n_peaks += 1

    out = 0
    for i in range(n_peaks):
        j = -1
        while troughs[j + 1] < peaks[i] and j + 1 < n_troughs:
            j += 1

        if j == -1 or acf[peaks[i]] - acf[troughs[j]] < 0.01 or acf[peaks[i]] < 0:
            continue

        out = peaks[i]
        break

    return out
