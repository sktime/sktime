import pandas as pd
import numpy as np
from sktime.utils.data_container import detabularise


def generate_df_from_array(array, n_rows=10, n_cols=1):
    return pd.DataFrame([[pd.Series(array) for _ in range(n_cols)] for _ in range(n_rows)],
                        columns=[f'col{c}' for c in range(n_cols)])


def generate_polynomial_series(n, order, coefs=None):
    """Helper function to generate polynomial series of given order and coefficients"""
    if coefs is None:
        coefs = np.ones((order + 1, 1))

    x = np.vander(np.arange(n), N=order + 1).dot(coefs)
    return x.ravel()


def generate_time_series_data_with_trend(n_samples=1, n_obs=100, order=0, coefs=None):
    """Helper function to generate time series/panel data with polynomial trend"""
    samples = []
    for i in range(n_samples):
        s = generate_polynomial_series(n_obs, order=order, coefs=coefs)

        index = np.arange(n_obs)
        y = pd.Series(s, index=index)
        samples.append(y)

    X = pd.DataFrame(samples)
    assert X.shape == (n_samples, n_obs)
    return detabularise(X)


def generate_seasonal_time_series_data_with_trend(n_samples=1, n_obs=100, order=0, sp=1, model='additive'):
    """Helper function to generate time series/panel data with polynomial trend and seasonal component"""
    if sp == 1:
        return generate_time_series_data_with_trend(n_samples=n_samples, n_obs=n_obs, order=order)

    samples = []
    for i in range(n_samples):
        # coefs = np.random.normal(scale=0.01, size=(order + 1, 1))
        s = generate_polynomial_series(n_obs, order)

        if model == 'additive':
            s[::sp] = s[::sp] + 0.1
        else:
            s[::sp] = s[::sp] * 1.1

        index = np.arange(n_obs)
        y = pd.Series(s, index=index)
        samples.append(y)

    X = pd.DataFrame(samples)
    assert X.shape == (n_samples, n_obs)
    return detabularise(X)

