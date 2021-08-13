# -*- coding: utf-8 -*-
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.base._meta import _HeterogenousEnsembleForecaster
from sktime.forecasting.compose._pipeline import TransformedTargetForecaster
from sktime.forecasting.compose import ColumnEnsembleForecaster
from sktime.transformations.series.theta import ThetaLinesTransformer
from sktime.forecasting.compose._ensemble import _aggregate


class ThetaForecaster(_HeterogenousEnsembleForecaster):
    _required_parameters = ["forecasters"]

    _tags = {
        "scitype:y": "univariate",
        "univariate-only": False,
        "y_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters, theta_values=(0, 2), aggfunc="mean", weights=None):
        super(ThetaForecaster, self).__init__(forecasters=forecasters)
        self.aggfunc = aggfunc
        self.weights = weights
        self.theta_values = theta_values

    def fit(self, y, X=None, fh=None):

        self.pipe = TransformedTargetForecaster(
            steps=[
                ("transformer", ThetaLinesTransformer(theta=self.theta_values)),
                ("forecaster", ColumnEnsembleForecaster(forecasters=self.forecasters)),
            ]
        )
        self.pipe.fit()
        return self

    def predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):

        Y_pred = self.pipe.predict(fh, X, return_pred_int=return_pred_int, alpha=alpha)
        # what data type will be in self after fit? pd.DataFrame? pd.Series?
        # add checks - theta_values length should be the same as forecasters list

        return _aggregate(Y_pred)
