# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of OosResidualsWrapper."""

__author__ = ["geetu040"]

import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster


class OosResidualsWrapper(BaseForecaster):
    """Out-of-sample residuals wrapper for forecasters."""

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:insample": True,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "python_version": None,
    }

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.forecaster_ = None

        super().__init__()

        # self.set_tags(**{})

    def _fit(self, y, X, fh):
        self.forecaster_ = clone(self.forecaster).fit(y, X, fh)

    def _predict(self, fh, X):
        """Predicts using the OosResidualsWrapper.

        1. Split fh into in-sample and out-of-sample
        2. Calculate out-of-sample predictions using the wrapped forecaster
        3. Calculate in-sample residuals using cross-validation on the training data
        4. Combine the in-sample and out-of-sample predictions
        """
        _y = self._y

        # Step 1
        fh = self._check_fh(fh)
        fh = fh.to_relative(self.cutoff)

        fh_in_sample_mask = fh._is_in_sample()
        fh_out_of_sample_mask = fh._is_out_of_sample()

        fh_in_sample = fh[fh_in_sample_mask]
        fh_out_of_sample = fh[fh_out_of_sample_mask]

        # Step 2
        preds_out_of_sample = self.forecaster_.predict(fh=fh_out_of_sample, X=X)

        # Step 3
        from sktime.split import CutoffSplitter

        fh_in_sample = _y.shape[0] + fh_in_sample - 1
        cv = CutoffSplitter(fh_in_sample)

        preds_in_sample = self.forecaster_.update_predict(
            _y, cv, X, update_params=False, reset_forecaster=True
        )

        # Step 4
        preds = pd.concat([preds_in_sample, preds_out_of_sample]).sort_index()

        return preds


if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor

    from sktime.datasets import load_longley
    from sktime.forecasting.compose import make_reduction
    from sktime.split import temporal_train_test_split

    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, test_size=4)

    # forecaster = NaiveForecaster(strategy="last")
    forecaster = make_reduction(
        estimator=RandomForestRegressor(),
        # estimator=LinearRegression(),
        strategy="recursive",
        window_length=3,
    )

    forecaster = OosResidualsWrapper(forecaster=forecaster)

    forecaster.fit(y_train)

    # preds = forecaster.predict(fh=[-3, -2, -1, 0, 1, 2, 3])
    # print(preds)
    # plot_series(y_train, y_test, preds)
    # plt.show()

    residuals = forecaster.predict_residuals(y_test)
    print(residuals)

    print("\n\n\n..... Reached the End .....")
