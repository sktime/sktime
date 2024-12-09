# Copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""SplineTrendForecaster implementation."""

__author__ = ["Dehelaan"]
__all__ = ["SplineTrendForecaster"]

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from sktime.forecasting.trend import TrendForecaster, _DelegatedForecaster


class SplineTrendForecaster(_DelegatedForecaster):
    r"""Forecast time series data using a spline regression model.

    Uses an `sklearn` regressor specified by the `regressor` parameter
    to perform regression on time series values against their corresponding indices,
    after transformation with `SplineTransformer`.

    Parameters
    ----------
    degree : int, default=1
        Degree of the splines (1 for linear, 2 for quadratic, etc.).
    n_knots : int, default=4
        Number of knots for the spline transformation.
    extrapolation : {'constant', 'linear', 'periodic', 'continue'}, default='constant'
        Extrapolation strategy for splines beyond the range of the data.
    include_bias : bool, default=True
        Whether to include a bias term in the spline features.
    regressor : sklearn estimator, default=LinearRegression()
        The regressor to use for fitting the transformed features.

    Attributes
    ----------
    regressor_ : sklearn regressor estimator object
        The fitted regressor object.
        This is a fitted `sklearn` pipeline with steps
        `SplineTransformer(degree, n_knots, include_bias)`, followed by
        a clone of the `regressor`.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sklearn.linear_model import Ridge
    >>> y = load_airline()
    >>> forecaster = SplineTrendForecaster(extrapolation="linear", degree=2)
    >>> forecaster.fit(y)
    SplineTrendForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _delegate_name = "forecaster_"

    _tags = {
        "authors": ["Dehelaan"],
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        regressor=None,
        degree=1,
        n_knots=4,
        extrapolation="constant",
        include_bias=True,
    ):
        self.degree = degree
        self.n_knots = n_knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.regressor = regressor if regressor is not None else LinearRegression()

        spline_regressor = make_pipeline(
            SplineTransformer(
                degree=self.degree,
                n_knots=self.n_knots,
                extrapolation=self.extrapolation,
                include_bias=self.include_bias,
            ),
            clone(self.regressor),
        )

        spline_forecaster = TrendForecaster(spline_regressor)
        super().__init__(forecaster=spline_forecaster)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sklearn.ensemble import RandomForestRegressor

        params_list = [
            {},
            {
                "regressor": RandomForestRegressor(),
                "degree": 1,
                "include_bias": False,
                "n_knots": 4,
            },
            {
                "n_knots": 4,
                "degree": 2,
                "include_bias": True,
                "extrapolation": "periodic",
            },
            {
                "regressor": RandomForestRegressor(),
                "n_knots": 4,
                "degree": 2,
                "include_bias": False,
                "extrapolation": "periodic",
            },
        ]

        return params_list
