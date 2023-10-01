#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implement cutoff dataset splitting for model evaluation and selection."""

__all__ = ["temporal_train_test_split"]

from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split as _train_test_split

from sktime.datatypes._utilities import get_time_index
from sktime.split.base._common import (
    ACCEPTED_Y_TYPES,
    FORECASTING_HORIZON_TYPES,
    SPLIT_TYPE,
    _split_by_fh,
)


def temporal_train_test_split(
    y: ACCEPTED_Y_TYPES,
    X: Optional[pd.DataFrame] = None,
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    fh: Optional[FORECASTING_HORIZON_TYPES] = None,
) -> SPLIT_TYPE:
    """Split arrays or matrices into sequential train and test subsets.

    Creates train/test splits over endogenous arrays an optional exogenous
    arrays.

    This is a wrapper of scikit-learn's ``train_test_split`` that
    does not shuffle the data.

    Parameters
    ----------
    y : time series in sktime compatible data container format
    X : time series in sktime compatible data container format, optional, default=None
        y and X can be in one of the following formats:
        Series scitype: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
            for vanilla forecasting, one time series
        Panel scitype: pd.DataFrame with 2-level row MultiIndex,
            3D np.ndarray, list of Series pd.DataFrame, or nested pd.DataFrame
            for global or panel forecasting
        Hierarchical scitype: pd.DataFrame with 3 or more level row MultiIndex
            for hierarchical forecasting
        Number of columns admissible depend on the "scitype:y" tag:
            if self.get_tag("scitype:y")=="univariate":
                y must have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                y must have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions on columns apply
        For further details:
            on usage, see forecasting tutorial examples/01_forecasting.ipynb
            on specification of formats, examples/AA_datatypes_and_datasets.ipynb
    test_size : float, int or None, optional (default=None)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        relative number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the relative number of train samples. If None,
        the value is automatically set to the complement of the test size.
    fh : ForecastingHorizon

    Returns
    -------
    splitting : tuple, length=2 * len(arrays)
        List containing train-test split of `y` and `X` if given.

    References
    ----------
    .. [1]  adapted from https://github.com/alkaline-ml/pmdarima/
    """
    if fh is not None:
        if test_size is not None or train_size is not None:
            raise ValueError(
                "If `fh` is given, `test_size` and `train_size` cannot "
                "also be specified."
            )
        return _split_by_fh(y, fh, X=X)
    else:
        pd_format = isinstance(y, pd.Series) or isinstance(y, pd.DataFrame)
        if pd_format is True and isinstance(y.index, pd.MultiIndex):
            ys = get_time_index(y)
            # Get index to group across (only indices other than timepoints index)
            yi_name = y.index.names
            yi_grp = yi_name[0:-1]

            # Get split into test and train data for timeindex only
            series = (ys,)
            yret = _train_test_split(
                *series,
                shuffle=False,
                stratify=None,
                test_size=test_size,
                train_size=train_size,
            )

            # Convert into list indices
            ysl = ys.to_list()
            yrl1 = yret[0].to_list()
            yrl2 = yret[1].to_list()
            p1 = [index for (index, item) in enumerate(ysl) if item in yrl1]
            p2 = [index for (index, item) in enumerate(ysl) if item in yrl2]

            # Subset by group based on identified indices
            y_train = y.groupby(yi_grp, as_index=False).nth(p1)
            y_test = y.groupby(yi_grp, as_index=False).nth(p2)
            if X is not None:
                X_train = X.groupby(yi_grp, as_index=False).nth(p1)
                X_test = X.groupby(yi_grp, as_index=False).nth(p2)
                return y_train, y_test, X_train, X_test
            else:
                return y_train, y_test
        else:
            series = (y,) if X is None else (y, X)
            return _train_test_split(
                *series,
                shuffle=False,
                stratify=None,
                test_size=test_size,
                train_size=train_size,
            )
