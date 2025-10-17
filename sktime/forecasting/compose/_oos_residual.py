# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of OosResidualsWrapper."""

__author__ = ["geetu040"]

import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster


class OosResidualsWrapper(BaseForecaster):
    """Out-of-sample residuals wrapper for forecasters."""

    _tags = {
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "scitype:y": "univariate",
        "capability:exogenous": True,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "capability:missing_values": True,
        "capability:categorical_in_X": True,
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires-fh-in-fit": False,
        "fit_is_empty": True,
    }

    def __init__(self, forecaster, cv=None):
        self.forecaster = forecaster
        self.cv = cv
        self._in_sample_forecaster = None
        self._out_of_sample_forecaster = None

        super().__init__()

        self.set_tags(
            **{
                "capability:exogenous": self.forecaster.get_tag("capability:exogenous"),
                "capability:pred_int": self.forecaster.get_tag("capability:pred_int"),
                "capability:missing_values": self.forecaster.get_tag(
                    "capability:missing_values"
                ),
                "capability:categorical_in_X": self.forecaster.get_tag(
                    "capability:categorical_in_X"
                ),
            }
        )

    def _fit(self, y, X, fh):
        pass

    def _predict(self, fh, X):
        """Predicts using the OosResidualsWrapper.

        1. Split fh into in-sample and out-of-sample
        2. Calculate out-of-sample predictions using the wrapped forecaster
        3. Calculate in-sample residuals using cross-validation on the training data
        4. Combine the in-sample and out-of-sample predictions
        """
        _y = self._y
        _X = self._X

        # Split fh into in-sample and out-of-sample
        out_of_sample_fh = fh.to_out_of_sample()
        in_sample_fh = fh.to_in_sample()

        # Prepare CV
        cv = self.cv
        if cv is None:
            from sktime.split import CutoffFhSplitter

            cv = CutoffFhSplitter(
                cutoff=(in_sample_fh.to_absolute(self.cutoff) - 1), fh=1
            )

        # Prepare forecasters
        self._in_sample_forecaster = clone(self.forecaster)
        self._out_of_sample_forecaster = clone(self.forecaster)

        # Out-of-sample predictions
        self._out_of_sample_forecaster.fit(y=_y, X=_X, fh=out_of_sample_fh)
        out_of_sample_preds = self._out_of_sample_forecaster.predict(X=X)

        # In-sample predictions using cross-validation

        # fit on the first training windows
        new_y, _ = next(cv.split(_y))
        self._in_sample_forecaster.fit(y=_y.iloc[new_y], X=_X, fh=cv.get_fh())
        # update on all training windows
        in_sample_preds = self._in_sample_forecaster.update_predict(
            _y, cv, X, update_params=True, reset_forecaster=False
        )
        # fix pd.Series output
        if isinstance(in_sample_preds, pd.Series):
            in_sample_preds = in_sample_preds.to_frame(name=_y.columns[0])

        # Combine the predictions
        preds = pd.concat([in_sample_preds, out_of_sample_preds])

        return preds
