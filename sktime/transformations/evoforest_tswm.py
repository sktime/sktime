"""EvoForest-TS-WM: a frozen, interpretable, closed-form time-series feature transform.

No learned weights; discovered by EvoForest under a world-model objective. numba-only;
the seeded banks are embedded (no torch, no data file). See the class docstring.
"""

__author__ = ["kayuksel"]
__all__ = ["EvoForestTSWM"]

import base64
import io
import math
import zlib
from functools import cache

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

# ===================== frozen seeded banks (embedded; ~8 KB) =====================
_B64 = (
    "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"
    "tqpvuPs3h16RYw=="
)


def load_banks():
    """Return the 6 banks as a dict of float64 numpy arrays."""
    raw = zlib.decompress(base64.b64decode(_B64))
    with np.load(io.BytesIO(raw)) as z:
        return {k: z[k].astype(np.float64) for k in z.files}


L = 64
_EPS = 1e-8


# ---------- constant banks (numpy, built once) ----------
def build_consts(banks=None):
    if banks is None:
        banks = load_banks()
    trf_mu = np.asarray(banks["trf_mu"], np.float64)
    trf_sig = np.asarray(banks["trf_sig"], np.float64)
    t = np.arange(L, dtype=np.float64) / L
    win = np.exp(
        -((t[None, :] - trf_mu[:, None]) ** 2) / (2 * trf_sig[:, None] ** 2 + _EPS)
    )
    GW = win / (win.sum(1, keepdims=True) + _EPS)  # (12, L) normalised windows

    nn = np.arange(L, dtype=np.float64)
    kr = np.arange(L // 2 + 1, dtype=np.float64)
    a = 2 * np.pi * np.outer(kr, nn) / L
    Cr, Ci = np.cos(a), np.sin(a)  # rfft cos/sin (33, L)
    kf = np.arange(L, dtype=np.float64)
    af = 2 * np.pi * np.outer(kf, nn) / L
    Fc, Fs = np.cos(af), np.sin(af)  # full DFT cos/sin (L, L)
    hfir = np.where(
        kf == 0, 1.0, np.where(kf < L / 2, 2.0, np.where(kf == L // 2, 1.0, 0.0))
    )

    crf = np.ascontiguousarray(
        np.asarray(banks["crf_w"], np.float64)[:, 0, :]
    )  # (16, 9)
    x = np.linspace(-7.0, 7.0, 15)
    Rk = np.empty((4, 15), np.float64)
    for i, s in enumerate((1.5, 2.5, 4.0, 6.0)):
        r = (1.0 - (x / s) ** 2) * np.exp(-(x**2) / (2 * s * s))
        Rk[i] = (r - r.mean()) / (np.abs(r).sum() + _EPS)
    srfW = np.ascontiguousarray(np.asarray(banks["srf_W"], np.float64))  # (12,6,12)
    srfb = np.ascontiguousarray(np.asarray(banks["srf_b"], np.float64))  # (12,6)
    srfu = np.ascontiguousarray(np.asarray(banks["srf_u"], np.float64))  # (12,6)
    return (GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu)


# ===================== patchify + pooling =====================
_FAMILIES = [
    ("stats", 12),
    ("srf_mlp", 12),
    ("autocorr", 2),
    ("spectral", 2),
    ("turning", 2),
    ("trf_gausswin", 12),
    ("crf_ppv", 32),
    ("hilbert_env", 2),
    ("crf_max", 32),
    ("morphology_updown", 4),
    ("fftbands", 6),
    ("perm_entropy", 3),
    ("curvature", 3),
    ("conv_position", 3),
    ("ar_residual", 4),
    ("acf_first_min", 3),
    ("histogram_mode", 3),
    ("ricker_wavelet", 4),
]
POOL_MAP = {
    "stats": ["std", "max"],
    "turning": ["std", "max"],
    "trf_gausswin": ["mean", "max"],
    "crf_ppv": ["mean", "std", "max"],
    "crf_max": ["mean", "max"],
    "morphology_updown": ["mean"],
    "fftbands": ["mean"],
    "curvature": ["mean", "max"],
    "conv_position": ["max"],
    "ar_residual": ["mean"],
    "histogram_mode": ["std", "max"],
    "ricker_wavelet": ["std"],
}
_OPS = ("mean", "std", "max")


def _patchify(v, stride=16, resample_short=True):
    v = np.asarray(v, float)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v)
    if len(v) < L:
        if resample_short and len(v) >= 2:
            v = np.interp(np.linspace(0, len(v) - 1, L), np.arange(len(v)), v)
        else:
            v = np.pad(v, (L - len(v), 0), mode="edge")
    st = list(range(0, len(v) - L + 1, stride))
    if st[-1] != len(v) - L:
        st.append(len(v) - L)
    return np.stack([v[s : s + L] for s in st])


@cache
def _consts():
    return build_consts()  # embedded banks, built once


# ===== njit phi kernels; numba imported lazily (it is a soft dependency) =====
@cache
def _kernels():
    """Lazily import numba and compile the phi kernels once (soft dependency)."""
    from numba import njit, prange

    @njit(cache=False)
    def _quantile_lin(sorted_x, q):
        n = sorted_x.shape[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        frac = pos - lo
        if lo + 1 >= n:
            return sorted_x[n - 1]
        return sorted_x[lo] + frac * (sorted_x[lo + 1] - sorted_x[lo])

    @njit(cache=False)
    def _conv_same(w, ker, dilation):
        """cross-correlation, 'same' padding = dilation*(k//2); length L."""
        Lw = w.shape[0]
        k = ker.shape[0]
        pad = dilation * (k // 2)
        out = np.zeros(Lw, np.float64)
        for t in range(Lw):  # out_len == L for these configs
            acc = 0.0
            for j in range(k):
                idx = t + j * dilation - pad
                if 0 <= idx < Lw:
                    acc += ker[j] * w[idx]
            out[t] = acc
        return out

    @njit(cache=False)
    def _phi_one(w, GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu):
        """One length-L patch -> 141 features, filled in family order."""
        n = w.shape[0]
        out = np.empty(141, np.float64)
        o = 0

        mean = 0.0
        for i in range(n):
            mean += w[i]
        mean /= n
        c = w - mean
        ss = 0.0
        for i in range(n):
            ss += c[i] * c[i]
        var1 = ss / (n - 1)  # unbiased (ddof=1), matches torch
        std1 = math.sqrt(var1)
        sden = std1 if std1 > _EPS else _EPS

        sw = np.sort(w)
        q25 = _quantile_lin(sw, 0.25)
        q50 = _quantile_lin(sw, 0.50)
        q75 = _quantile_lin(sw, 0.75)
        dw = np.empty(n - 1, np.float64)
        for i in range(n - 1):
            dw[i] = w[i + 1] - w[i]

        # ===== family 0: stats (12) =====
        skew = 0.0
        kurt = 0.0
        for i in range(n):
            skew += c[i] ** 3
            kurt += c[i] ** 4
        skew = (skew / n) / (sden**3 + _EPS)
        kurt = (kurt / n) / (sden**4 + _EPS) - 3.0
        a1n = 0.0
        d0 = 0.0
        d1 = 0.0
        for i in range(n - 1):
            a1n += c[i] * c[i + 1]
            d0 += c[i] * c[i]
            d1 += c[i + 1] * c[i + 1]
        ac1 = a1n / (math.sqrt(d0) * math.sqrt(d1) + _EPS)
        absdiff = 0.0
        for i in range(n - 1):
            absdiff += abs(dw[i])
        absdiff /= n - 1
        st = np.empty(12, np.float64)
        st[0] = mean
        st[1] = sden
        st[2] = skew
        st[3] = kurt
        st[4] = q25
        st[5] = q50
        st[6] = q75
        st[7] = q75 - q25
        st[8] = ac1
        st[9] = absdiff
        st[10] = sw[0]
        st[11] = sw[n - 1]
        for i in range(12):
            out[o] = st[i]
            o += 1

        # ===== family 1: srf_mlp (12) =====
        for kk in range(12):
            acc = 0.0
            for dd in range(6):
                h = srfb[kk, dd]
                for mm in range(12):
                    h += st[mm] * srfW[kk, dd, mm]
                if h < 0.0:
                    h = 0.0
                acc += h * srfu[kk, dd]
            out[o] = acc
            o += 1

        # ===== family 2: autocorr lags 1,2 (2) =====
        a2n = 0.0
        a2d = 0.0
        for i in range(n - 2):
            a2n += c[i] * c[i + 2]
            a2d += c[i] * c[i]
        out[o] = a1n / (d0 + _EPS)
        o += 1
        out[o] = a2n / (a2d + _EPS)
        o += 1

        # ----- rfft magnitude (shared by spectral & fftbands) -----
        nb = Cr.shape[0]  # 33
        mag = np.empty(nb, np.float64)
        for kb in range(nb):
            re = 0.0
            im = 0.0
            for i in range(n):
                re += w[i] * Cr[kb, i]
                im -= w[i] * Ci[kb, i]
            mag[kb] = math.sqrt(re * re + im * im)

        # ===== family 3: spectral centroid + entropy (2) =====
        psum = 0.0
        for kb in range(1, nb):
            psum += mag[kb]
        psum += _EPS
        cent = 0.0
        ent = 0.0
        for j in range(nb - 1):
            pj = mag[j + 1] / psum
            cent += j * pj
            ent -= pj * math.log(pj + 1e-12)
        out[o] = cent
        o += 1
        out[o] = ent
        o += 1

        # ===== family 4: turning (2) =====
        tp = 0.0
        for i in range(n - 2):
            if dw[i + 1] * dw[i] < 0.0:
                tp += 1.0
        fpos = 0.0
        for i in range(n - 1):
            if dw[i] > 0.0:
                fpos += 1.0
        out[o] = tp / (n - 2)
        o += 1
        out[o] = fpos / (n - 1)
        o += 1

        # ===== family 5: trf_gausswin (12) = w @ GW.T =====
        for kk in range(12):
            acc = 0.0
            for i in range(n):
                acc += w[i] * GW[kk, i]
            out[o] = acc
            o += 1

        # conv outputs at dilations 2,4 (cached for ppv/max/position)
        conv2 = np.empty((16, n), np.float64)
        conv4 = np.empty((16, n), np.float64)
        for ci in range(16):
            conv2[ci] = _conv_same(w, crf[ci], 2)
            conv4[ci] = _conv_same(w, crf[ci], 4)

        # ===== family 6: crf_ppv (32) = ppv@dil2 (16) then ppv@dil4 (16) =====
        for ci in range(16):
            p = 0.0
            for t in range(n):
                if conv2[ci, t] > 0.0:
                    p += 1.0
            out[o] = p / n
            o += 1
        for ci in range(16):
            p = 0.0
            for t in range(n):
                if conv4[ci, t] > 0.0:
                    p += 1.0
            out[o] = p / n
            o += 1

        # ===== family 7: hilbert_env (2) =====
        yre = np.empty(L, np.float64)
        yim = np.empty(L, np.float64)
        for kb in range(L):
            re = 0.0
            im = 0.0
            for i in range(n):
                re += w[i] * Fc[kb, i]
                im -= w[i] * Fs[kb, i]
            yre[kb] = re * hfir[kb]
            yim[kb] = im * hfir[kb]
        emean = 0.0
        env = np.empty(L, np.float64)
        for ni in range(L):
            ir = 0.0
            ii = 0.0
            for kb in range(L):
                ir += yre[kb] * Fc[kb, ni] - yim[kb] * Fs[kb, ni]
                ii += yre[kb] * Fs[kb, ni] + yim[kb] * Fc[kb, ni]
            ir /= L
            ii /= L
            env[ni] = math.sqrt(ir * ir + ii * ii)
            emean += env[ni]
        emean /= L
        evar = 0.0
        eabs = 0.0
        for i in range(L):
            evar += (env[i] - emean) ** 2
        for i in range(L - 1):
            eabs += abs(env[i + 1] - env[i])
        out[o] = math.sqrt(evar / (L - 1)) / (emean + _EPS)
        o += 1
        out[o] = (eabs / (L - 1)) / (emean + _EPS)
        o += 1

        # ===== family 8: crf_max (32) = max@dil2 (16) then max@dil4 (16) =====
        for ci in range(16):
            m = conv2[ci, 0]
            for t in range(1, n):
                if conv2[ci, t] > m:
                    m = conv2[ci, t]
            out[o] = m
            o += 1
        for ci in range(16):
            m = conv4[ci, 0]
            for t in range(1, n):
                if conv4[ci, t] > m:
                    m = conv4[ci, t]
            out[o] = m
            o += 1

        # ===== family 9: morphology_updown (4) =====
        pmax = 0.0
        nmax = 0.0
        pe = 0.0
        ne = 0.0
        for i in range(n - 1):
            d = dw[i]
            if d > 0.0:
                if d > pmax:
                    pmax = d
                pe += d * d
            elif d < 0.0:
                if -d > nmax:
                    nmax = -d
                ne += d * d
        out[o] = pmax
        o += 1
        out[o] = nmax
        o += 1
        out[o] = math.log((pmax + 1e-6) / (nmax + 1e-6))
        o += 1
        out[o] = math.log((pe + 1e-6) / (ne + 1e-6))
        o += 1

        # ===== family 10: fftbands (6) =====
        s2 = 0.0
        pw = np.empty(nb - 1, np.float64)
        for j in range(nb - 1):
            pw[j] = mag[j + 1] * mag[j + 1]
            s2 += pw[j]
        s2 += _EPS
        for b0 in range(0, nb - 1, 6):
            b1 = b0 + 6
            if b1 > nb - 1:
                b1 = nb - 1
            acc = 0.0
            for j in range(b0, b1):
                acc += pw[j] / s2
            if acc < 1e-8:
                acc = 1e-8
            out[o] = math.log(acc)
            o += 1

        # ===== family 11: perm_entropy + Hjorth mobility/complexity (3) =====
        H = np.zeros(8, np.float64)
        for i in range(n - 2):
            a = w[i]
            b = w[i + 1]
            cc = w[i + 2]
            code = 0
            if a < b:
                code += 4
            if b < cc:
                code += 2
            if a < cc:
                code += 1
            H[code] += 1.0
        pent = 0.0
        for k in range(8):
            Hk = H[k] / (n - 2)
            pent -= Hk * math.log(Hk + 1e-12)
        pent /= math.log(6.0)
        # var1(d1) and var1(d2), ddof=1
        md = 0.0
        for i in range(n - 1):
            md += dw[i]
        md /= n - 1
        vd1 = 0.0
        for i in range(n - 1):
            vd1 += (dw[i] - md) ** 2
        vd1 /= n - 2
        m2d = 0.0
        d2 = np.empty(n - 2, np.float64)
        for i in range(n - 2):
            d2[i] = w[i + 2] - 2.0 * w[i + 1] + w[i]
            m2d += d2[i]
        m2d /= n - 2
        vd2 = 0.0
        for i in range(n - 2):
            vd2 += (d2[i] - m2d) ** 2
        vd2 /= n - 3
        mob = math.sqrt((vd1 + _EPS) / (var1 + _EPS))
        comp = math.sqrt((vd2 + _EPS) / (vd1 + _EPS)) / (mob + _EPS)
        out[o] = pent
        o += 1
        out[o] = mob
        o += 1
        out[o] = comp
        o += 1

        # ===== family 12: curvature (3) =====
        cabsm = 0.0
        cabsx = 0.0
        cpe = 0.0
        cne = 0.0
        for i in range(n - 2):
            cc = d2[i]
            ac = abs(cc)
            cabsm += ac
            if ac > cabsx:
                cabsx = ac
            if cc > 0.0:
                cpe += cc * cc
            elif cc < 0.0:
                cne += cc * cc
        out[o] = cabsm / (n - 2)
        o += 1
        out[o] = cabsx
        o += 1
        out[o] = math.log((cpe + 1e-6) / (cne + 1e-6))
        o += 1

        # ===== family 13: conv_position (3) from dil2 argmax over 16 kernels =====
        pmean = 0.0
        posv = np.empty(16, np.float64)
        for ci in range(16):
            m = conv2[ci, 0]
            am = 0
            for t in range(1, n):
                if conv2[ci, t] > m:
                    m = conv2[ci, t]
                    am = t
            posv[ci] = am / n
            pmean += posv[ci]
        pmean /= 16
        pvar = 0.0
        pmn = posv[0]
        pmx = posv[0]
        for ci in range(16):
            pvar += (posv[ci] - pmean) ** 2
            if posv[ci] < pmn:
                pmn = posv[ci]
            if posv[ci] > pmx:
                pmx = posv[ci]
        out[o] = pmean
        o += 1
        out[o] = math.sqrt(pvar / 15)
        o += 1  # std ddof=1 over 16 positions
        out[o] = pmx - pmn
        o += 1

        # ===== family 14: ar_residual AR(2)+AR(3) (4) =====
        out[o] = 0.0
        out[o + 1] = 0.0
        out[o + 2] = 0.0
        out[o + 3] = 0.0
        _ar_fit(c, 2, out, o)
        _ar_fit(c, 3, out, o + 2)
        o += 4

        # ===== family 15: acf_first_min (3) =====
        cc2 = 0.0
        for i in range(n):
            cc2 += c[i] * c[i]
        acf = np.empty(32, np.float64)
        for k in range(1, 33):
            s = 0.0
            for i in range(n - k):
                s += c[i] * c[i + k]
            acf[k - 1] = s / (cc2 + _EPS)
        has = False
        fm = 0
        for j in range(1, 31):
            if acf[j] < acf[j - 1] and acf[j] <= acf[j + 1]:
                has = True
                fm = j - 1  # argmax of ismin (first True), index into length-30
                break
        first_min = ((fm + 2.0) if has else 32.0) / n
        fz = 0
        for j in range(32):
            if acf[j] < 0.0:
                fz = j
                break
        first_zero = fz / n
        idx_va = fm + 1
        if idx_va > 31:
            idx_va = 31
        out[o] = first_min
        o += 1
        out[o] = first_zero
        o += 1
        out[o] = acf[idx_va]
        o += 1

        # ===== family 16: histogram_mode (3) =====
        zden = std1 if std1 > 1e-6 else 1e-6
        z = np.empty(n, np.float64)
        for i in range(n):
            z[i] = (w[i] - mean) / zden
        inv = 1.0 / (5.0 / 9.0)
        e = np.empty(10, np.float64)
        maxl = -1e30
        for j in range(10):
            ctr = -2.5 + 5.0 * j / 9.0
            s = 0.0
            for i in range(n):
                u = (ctr - z[i]) * inv
                s += math.exp(-0.5 * u * u)
            e[j] = 4.0 * s
            if e[j] > maxl:
                maxl = e[j]
        den = 0.0
        for j in range(10):
            e[j] = math.exp(e[j] - maxl)
            den += e[j]
        m10 = 0.0
        for j in range(10):
            ctr = -2.5 + 5.0 * j / 9.0
            m10 += (e[j] / den) * ctr
        zs = np.sort(z)
        zmed = zs[(n - 1) // 2]  # torch lower median
        cmass = 0.0
        for i in range(n):
            if abs(z[i]) < 0.5:
                cmass += 1.0
        out[o] = m10
        o += 1
        out[o] = m10 - zmed
        o += 1
        out[o] = cmass / n
        o += 1

        # ===== family 17: ricker_wavelet (4) =====
        for ri in range(4):
            oc = _conv_same(w, Rk[ri], 1)
            p = 0.0
            for t in range(n):
                if oc[t] > 0.0:
                    p += 1.0
            out[o] = p / n
            o += 1

        return out

    @njit(cache=False)
    def _ar_fit(c, p, out, off):
        """AR(p) ridge fit on centered c; writes [log resid_var, log |beta|^2]."""
        n = c.shape[0]
        m = n - p  # rows
        A = np.zeros((p, p), np.float64)
        b = np.zeros(p, np.float64)
        for r in range(m):
            t = r + p  # target index c[t]; predictors c[t-1..t-p]
            for a in range(p):
                xa = c[t - 1 - a]
                b[a] += xa * c[t]
                for d in range(p):
                    A[a, d] += xa * c[t - 1 - d]
        for a in range(p):
            A[a, a] += 1e-3
        beta = np.linalg.solve(A, b)
        # residual variance (ddof=1) and coeff norm
        rm = 0.0
        resid = np.empty(m, np.float64)
        for r in range(m):
            t = r + p
            pred = 0.0
            for a in range(p):
                pred += beta[a] * c[t - 1 - a]
            resid[r] = c[t] - pred
            rm += resid[r]
        rm /= m
        rv = 0.0
        for r in range(m):
            rv += (resid[r] - rm) ** 2
        rv /= m - 1
        bn = 0.0
        for a in range(p):
            bn += beta[a] * beta[a]
        out[off] = math.log(rv + _EPS)
        out[off + 1] = math.log(bn if bn > _EPS else _EPS)

    @njit(parallel=True, cache=False)
    def _phi_batch(W, GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu):
        B = W.shape[0]
        out = np.empty((B, 141), np.float64)
        for b in prange(B):
            out[b] = _phi_one(W[b], GW, Cr, Ci, Fc, Fs, hfir, crf, Rk, srfW, srfb, srfu)
        return out

    return _phi_batch


def _phi(W):
    return _kernels()(np.ascontiguousarray(np.asarray(W, np.float64)), *_consts())


def _pool(p, pooling):
    if pooling == "full":
        return np.concatenate([p.mean(0), p.std(0), p.max(0)])
    cols, idx = [], 0
    for nm, w in _FAMILIES:
        seg = p[:, idx : idx + w]
        idx += w
        keep = POOL_MAP.get(nm, [])
        if "mean" in keep:
            cols.append(seg.mean(0))
        if "std" in keep:
            cols.append(seg.std(0))
        if "max" in keep:
            cols.append(seg.max(0))
    return np.concatenate(cols)


def _encode(instances, pooling):
    """Encode instances (each a list of 1-D channels) -> (n, D); one batched phi."""
    all_pats, bounds = [], []
    for inst in instances:
        cnt = 0
        for ch in inst:
            ch = np.asarray(ch, float)
            z = (ch - ch.mean()) / (ch.std() + 1e-8)
            P = _patchify(z)
            all_pats.append(P)
            cnt += len(P)
        bounds.append(cnt)
    feats = _phi(np.concatenate(all_pats, 0))
    out, i = [], 0
    for cnt in bounds:
        out.append(_pool(feats[i : i + cnt], pooling))
        i += cnt
    return np.asarray(out)


class EvoForestTSWM(BaseTransformer):
    """EvoForest Time-Series World-Model encoder (TS-WM).

    A frozen, closed-form feature transform; see the module docstring. No learned
    weights; ``fit`` is empty. A peer of ``Catch22`` / ``MiniRocket``.

    Parameters
    ----------
    pooling : {"full", "pruned"}, default="full"
        423 features ("full") or 245 ("pruned").
    """

    _tags = {
        # packaging info
        # --------------
        "python_dependencies": "numba",
        "authors": ["EvoForest"],
        # estimator type
        # --------------
        "object_type": "transformer",
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "nested_univ",
        "y_inner_mtype": "None",
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": True,
        # test and CI flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(self, pooling="full"):
        self.pooling = pooling
        super().__init__()

    def _transform(self, X, y=None):
        if self.pooling not in ("full", "pruned"):
            raise ValueError(
                f"pooling must be 'full' or 'pruned', got {self.pooling!r}"
            )
        instances = [
            [np.asarray(row[c], dtype=float) for c in X.columns]
            for _, row in X.iterrows()
        ]
        return pd.DataFrame(_encode(instances, self.pooling), index=X.index)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [{"pooling": "full"}, {"pooling": "pruned"}]
