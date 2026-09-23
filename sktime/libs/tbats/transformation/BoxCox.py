import numpy as np
from sklearn.utils.validation import column_or_1d as c1d

from .._compat import check_array

from .. import error
from . import Guerrero


def find_box_cox_lambda(y, seasonal_periods=None, bounds=(-1, 2)):
    y = c1d(check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_samples=1,
                        copy=False, dtype=np.float64))  # type: np.ndarray

    guerrero = Guerrero()
    return guerrero.find_lambda(y, seasonal_periods=seasonal_periods, bounds=bounds)


def boxcox(y, lam=None, seasonal_periods=None, bounds=(-1, 2)):
    y = c1d(check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_samples=1,
                        copy=False, dtype=np.float64))  # type: np.ndarray
    if lam is None:
        lam = find_box_cox_lambda(y, seasonal_periods=seasonal_periods, bounds=bounds)
    if (lam <= 0 or np.isclose(lam, 0)) and np.any(y <= 0):
        raise error.InputArgsException('y must have only positive values for box-cox transformation.')
    if np.isclose(0.0, lam):
        return np.log(y)
    return (np.sign(y) * (np.abs(y) ** lam) - 1) / lam


def inv_boxcox(y, lam, force_valid=False):
    y = c1d(check_array(y, ensure_2d=False, force_all_finite=True, ensure_min_samples=1,
                        copy=False, dtype=np.float64))  # type: np.ndarray
    if np.isclose(0.0, lam):
        return np.exp(y)
    if lam < 0 and force_valid:
        y[y > -1 / lam] = -1 / lam
    if lam < 0 and np.any(y > -1 / lam):
        raise error.InputArgsException('Not possible to transform back such y values.')
    yy = y * lam + 1
    return np.sign(yy) * (np.abs(yy) ** (1 / lam))
