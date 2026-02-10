import numpy as np
import pandas as pd

from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import TransformedTargetForecaster


def test_ttf_predict_proba_delegates_to_inner():
    """TTF predict_proba should delegate to wrapped forecaster."""

    y = pd.Series(np.arange(1, 21))
    fh = [1, 2, 3]

    base = NaiveForecaster(strategy="last")

    ttf = TransformedTargetForecaster([
        ("forecaster", base)
    ])

    base.fit(y, fh=fh)
    ttf.fit(y, fh=fh)

    p_base = base.predict_proba(fh)
    p_ttf = ttf.predict_proba(fh)

    assert type(p_base) is type(p_ttf)

    # compare parameters numerically
    assert p_base.mu.equals(p_ttf.mu)
    assert p_base.sigma.equals(p_ttf.sigma)