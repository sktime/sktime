# Copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""SplineTrendForecaster implementation."""

__author__ = ["jgyasu", "Dehelaan"]
__all__ = ["SplineTrendForecaster"]

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from sktime.forecasting.base._delegate import _DelegatedForecaster


class SplineTrendForecaster(_DelegatedForecaster):
    r"""Forecast time series data with a spline trend.

    Uses an ``sklearn`` regressor specified by the ``regressor`` parameter
    to perform regression on time series values against their corresponding indices,
    after transformation of the indices with ``SplineTransformer``.

    Parameters
    ----------
    regressor : sklearn regressor estimator object, default=None
        Define the regression model type. If not set, defaults to
        ``sklearn.linear_model.LinearRegression``.

    n_knots : int, default=5
        Number of knots of the splines if ``knots`` is one of {'uniform', 'quantile'}.
        Must be at least 2. Ignored if ``knots`` is array-like.

    degree : int, default=1
        Degree of the splines (1 for linear, 2 for quadratic, etc.).
    n_knots : int, default=4
        Number of knots for the spline transformation.

    knots : {'uniform', 'quantile'}or array-like of shape (n_knots, n_features),
        default='uniform'
        Determines knot positions such that first knot <= features <= last knot.

        - 'uniform': `n_knots` are distributed uniformly between the
        min and max values of the features.
        - 'quantile': `n_knots` are distributed uniformly along the quantiles
        of the features.
        - array-like: Specifies sorted knot positions, including the boundary knots.
        Internally, additional knots are added before the first knot and after
        the last knot based on the spline degree.

    extrapolation : {'error', 'constant', 'linear', 'continue', 'periodic'},
        default='constant'
        Determines how to handle values outside the min and max values of the
        training features:

        - 'error': Raises a ValueError.
        - 'constant': Uses the spline value at the minimum or maximum feature as
        constant extrapolation.
        - 'linear': Applies linear extrapolation.
        - 'continue': Extrapolates as is (equivalent to `extrapolate=True` in
        `scipy.interpolate.BSpline`).
        - 'periodic': Uses periodic splines with a periodicity equal to the distance
        between the first and last knot, enforcing equal function values and
        derivatives at these knots.

    with_intercept : bool, default=True
        If True, includes a feature in which all polynomial powers are
        zero (i.e., a column of ones, acting as an intercept term in a linear
        model).

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.trend import SplineTrendForecaster
    >>> y = load_airline()
    >>> forecaster = SplineTrendForecaster(
    ...     n_knots=5,
    ...     degree=2,
    ...     knots="uniform",
    ...     extrapolation="constant"
    ... )
    >>> forecaster.fit(y)
    SplineTrendForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])
    """

    _delegate_name = "forecaster_"

    _tags = {
        "authors": ["jgyasu", "Dehelaan"],
        "maintainers": ["jgyasu"],
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "capability:pred_int": False,
    }

    def __init__(
        self,
        regressor=None,
        n_knots=5,
        degree=1,
        knots="uniform",
        extrapolation="constant",
        with_intercept=True,
    ):
        self.regressor = regressor
        self.degree = degree
        self.n_knots = n_knots
        self.knots = knots
        self.extrapolation = extrapolation
        self.with_intercept = with_intercept

        super().__init__()

        if self.regressor is None:
            regressor = LinearRegression(fit_intercept=False)
        else:
            regressor = clone(self.regressor)

        spline_regressor = make_pipeline(
            SplineTransformer(
                n_knots=self.n_knots,
                degree=self.degree,
                knots=self.knots,
                extrapolation=self.extrapolation,
                include_bias=self.with_intercept,
            ),
            regressor,
        )

        from sktime.forecasting.trend import TrendForecaster

        self.forecaster_ = TrendForecaster(spline_regressor)

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
                "n_knots": 5,
                "degree": 1,
                "knots": "uniform",
                "extrapolation": "constant",
                "with_intercept": False,
            },
            {
                "regressor": RandomForestRegressor(),
                "n_knots": 4,
                "degree": 1,
                "knots": "quantile",
                "extrapolation": "linear",
                "with_intercept": True,
            },
            {
                "regressor": RandomForestRegressor(),
                "n_knots": 3,
                "degree": 2,
                "knots": "uniform",
                "extrapolation": "periodic",
                "with_intercept": False,
            },
        ]

        return params_list
