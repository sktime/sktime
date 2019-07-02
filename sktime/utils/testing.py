import pandas as pd
import numpy as np


def generate_df_from_array(array, n_rows=10, n_cols=1):
    """Helper function to generate nested pandas DataFrame from array"""
    return pd.DataFrame([[pd.Series(array) for _ in range(n_cols)] for _ in range(n_rows)],
                        columns=[f'col{c}' for c in range(n_cols)])


def generate_polynomial_series(n, order, coefs=None):
    """Helper function to generate polynomial series of given order and coefficients"""
    if coefs is None:
        coefs = np.ones((order + 1, 1))

    x = np.vander(np.arange(n), N=order + 1).dot(coefs)
    return x
