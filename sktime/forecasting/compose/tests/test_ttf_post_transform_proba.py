import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.transformations.series.exponent import ExponentTransformer


def test_ttf_post_transform_applied_to_quantiles():
    """Post-transformers should apply to quantile predictions same as predict()."""

    # simple positive series
    y = pd.Series(np.arange(1, 21))

    # post-transformer squares outputs
    ttf = TransformedTargetForecaster(
        steps=[
            ("forecaster", NaiveForecaster(strategy="last")),
            ("post", ExponentTransformer(power=2)),
        ]
    )

    fh = [1, 2, 3]
    ttf.fit(y, fh=fh)

    # point prediction (post-transform IS applied)
    y_pred = ttf.predict(fh)

    # median quantile prediction (should match point prediction)
    q_pred = ttf.predict_quantiles(fh=fh, alpha=[0.5])

    # extract median column
    q_median = q_pred.xs(0.5, level=1, axis=1).iloc[:, 0]

    # they should match — currently they DO NOT
    assert np.allclose(y_pred.values, q_median.values)


def test_ttf_post_transform_applied_to_interval():
    y = pd.Series(np.arange(1, 21))

    ttf = TransformedTargetForecaster(
        steps=[
            ("forecaster", NaiveForecaster(strategy="last")),
            ("post", ExponentTransformer(power=2)),
        ]
    )

    fh = [1, 2, 3]
    ttf.fit(y, fh=fh)

    y_pred = ttf.predict(fh)

    i_pred = ttf.predict_interval(fh=fh, coverage=[0.5])

    lower = i_pred.xs(("Coverage", 0.5, "lower"), axis=1).iloc[:, 0]

    assert np.allclose(y_pred.values, lower.values)
