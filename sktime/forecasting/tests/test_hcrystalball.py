# -*- coding: utf-8 -*-
# #!/usr/bin/env python3 -u
# # -*- coding: utf-8 -*-
# # copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
# __author__ = ["Michal Chromcak"]

import numpy as np
import pandas as pd
import pytest

# from sktime.forecasting.hcrystalball import HCrystalBallForecaster
from sktime.forecasting.hcrystalball import _adapt_fit_data
from sktime.forecasting.hcrystalball import _adapt_predict_data
from sktime.forecasting.hcrystalball import _safe_merge

# from sktime.forecasting.hcrystalball import _convert_predictions
# from sktime.forecasting.hcrystalball import _ensure_datetime_index

# n_timepoints = 30
n_train = 20
start_date = "2020-01-01"
# s = pd.Series(np.arange(n_timepoints))
# y_train = s.iloc[:n_train]
# y_test = s.iloc[n_train:]


@pytest.fixture
def y_train(request):
    if "dt" in request.param:
        return pd.Series(
            np.arange(n_train),
            index=pd.date_range(start=start_date, periods=n_train, freq="D"),
        )
    elif "int" in request.param:
        return pd.Series(np.arange(n_train))


@pytest.fixture
def X_train(request):
    dt_ind = pd.date_range(start=start_date, periods=n_train, freq="D")

    if "None" in request.param:
        return None
    if "dt_only" in request.param:
        return pd.DataFrame(index=dt_ind)
    elif "dt_exog" in request.param:
        return pd.DataFrame({"ex": np.arange(n_train)}, index=dt_ind)
    elif "dt_short" in request.param:
        return pd.DataFrame(index=dt_ind).iloc[:-1]
    elif "int_only" in request.param:
        return pd.DataFrame(index=np.arange(n_train))
    elif "int_exog" in request.param:
        return pd.DataFrame({"ex": np.arange(n_train)})
    elif "int_short" in request.param:
        return pd.DataFrame({"ex": np.arange(n_train)}).iloc[:-1]


@pytest.mark.parametrize(
    "y_train, X_train, exp_error",
    [
        ("dt", "None", None),
        ("int", "dt_only", None),
        ("dt", "dt_exog", None),
        ("dt", "dt_short", None),
        ("dt", "int_exog", None),
        ("int", "None", ValueError),
        ("int", "int_only", ValueError),
        ("int", "dt_short", ValueError),
    ],
    indirect=["y_train", "X_train"],
)
def test_adapt_fit_data_no_X_train(y_train, X_train, exp_error):
    if exp_error is not None:
        with pytest.raises(exp_error):
            _adapt_fit_data(y_train, X_train)

    else:
        y_train_real, X_train_real = _adapt_fit_data(y_train, X_train)

        assert isinstance(X_train_real, pd.DataFrame)
        assert isinstance(X_train_real.index, pd.DatetimeIndex)
        assert isinstance(y_train_real, pd.Series)
        assert isinstance(y_train_real.index, pd.DatetimeIndex)


@pytest.mark.parametrize(
    "X_train, exp_error",
    [
        ("None", None),
        ("dt_only", None),
        ("dt_exog", None),
        ("dt_short", None),
        ("int_exog", None),
        ("int_only", None),
        ("int_short", ValueError),
    ],
    indirect=["X_train"],
)
def test_adapt_predict_data(X_train, exp_error):
    index = pd.date_range(start=start_date, periods=n_train, freq="D")

    if exp_error is not None:
        with pytest.raises(exp_error):
            _adapt_predict_data(X_train, index=index)

    else:
        X_transformed = _adapt_predict_data(X_train, index=index)

        assert isinstance(X_transformed, pd.DataFrame)
        assert isinstance(X_transformed.index, pd.DatetimeIndex)
        assert len(X_transformed) == len(index)


@pytest.mark.parametrize(
    "X_train, exp_error",
    [
        ("None", None),
        ("dt_only", None),
        ("dt_exog", None),
        ("dt_short", None),
        ("int_exog", None),
        ("int_only", None),
        ("int_short", ValueError),
    ],
    indirect=["X_train"],
)
def test_safe_merge(X_train, exp_error):
    dummy_df = pd.DataFrame(
        index=pd.date_range(start=start_date, periods=n_train, freq="D")
    )
    X_train = X_train if X_train is not None else pd.DataFrame()

    if exp_error is not None:
        with pytest.raises(exp_error):
            _safe_merge(real_df=X_train, dummy_df=dummy_df)

    else:
        merged_df = _safe_merge(real_df=X_train, dummy_df=dummy_df)

        assert len(merged_df) == len(dummy_df)
        assert isinstance(merged_df, pd.DataFrame)
        assert isinstance(merged_df.index, pd.DatetimeIndex)
