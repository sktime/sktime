import numpy as np
import pandas as pd
import pytest

from sktime.forecasters import ARIMAForecaster
from sktime.forecasters.compose import TransformedTargetForecaster
from sktime.transformers.series_to_series import Detrender
from sktime.datasets import load_shampoo_sales
from sktime.utils.validation import check_consistent_indices


@pytest.mark.parametrize("trend_order", [0, 1, 2])
@pytest.mark.parametrize("arima_order", [(2, 1, 0), (4, 2, 0)])
def test_fit_predict(trend_order, arima_order):
    fh = np.arange(3) + 1
    n_fh = len(fh)

    # load data and split into train/test series
    y = load_shampoo_sales()
    train = pd.Series([y.iloc[0].iloc[:-n_fh]])
    test = pd.Series([y.iloc[0].iloc[-n_fh:]])

    transformer = Detrender(order=trend_order)
    forecaster = ARIMAForecaster(order=arima_order)

    # use meta-estimator
    f = TransformedTargetForecaster(forecaster, transformer)
    f.fit(train)
    actual = f.predict(fh=fh)
    check_consistent_indices(actual, test.iloc[0])

    # manual transform-inverse-transform
    train = pd.DataFrame(train)
    traint = transformer.fit_transform(train)
    traint = traint.iloc[:, 0]

    forecaster.fit(traint)
    pred = forecaster.predict(fh=fh)

    pred = pd.DataFrame(pd.Series([pred]))
    pred = transformer.inverse_transform(pred)
    expected = pred.iloc[0, 0]
    check_consistent_indices(expected, test.iloc[0])

    np.testing.assert_array_equal(actual, expected)



