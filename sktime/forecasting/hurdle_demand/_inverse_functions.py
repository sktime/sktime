import numpy as np
from scipy.stats import nbinom, poisson


def _inverse_poisson(dist, u):
    rate = np.asarray(dist.rate)
    u = np.asarray(u)

    density = poisson(rate)
    return density.ppf(u)


def _inverse_neg_binom(dist, u):
    concentration = np.asarray(dist.concentration)
    mean = np.asarray(dist.mean)
    u = np.asarray(u)

    n = concentration
    p = concentration / (concentration + mean)

    density = nbinom(n=n, p=p)
    return density.ppf(u)
