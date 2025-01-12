"""Use endogeneous as exogenous features transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly", "avishkarsonni"]
__all__ = ["YtoX"]

from sktime.transformations.base import BaseTransformer


class YtoX(BaseTransformer):
    """Create exogenous features from a copy of the endogenous data.

    Replaces exogenous features (``X``) with the endogenous data (``y``), optionally
    applying a transformer to ``y`` before using it as exogenous data.

    To *add* instead of *replace*, use ``FeatureUnion``.

    Common use cases include:

    * creating exogenous variables from transformed endogenous variables
    * creating exogenous data from the index, if no exogenous data is available
    * manual construction of reduction strategies, in combination with ``YfromX``

    Parameters
    ----------
    subset_index : boolean, optional, default=False
        If True, subsets the output of ``transform`` to ``X.index``,
        i.e., outputs ``y.loc[X.index]``.

    transformer : sktime transformer, or callable optional, default=None
        If provided, will be applied to the endogenous data (``y``)
        before moving it to the exogenous data.
        If general callable, must implement ``fit_transform``.

    Examples
    --------
    Use case: creating exogenous data from index if no exogenous data is available.

    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.compose import YtoX
    >>> from sktime.transformations.series.fourier import FourierFeatures
    >>> from sktime.forecasting.arima import ARIMA
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>>
    >>> # data with no exogenous features
    >>> y = load_airline()
    >>>
    >>> # create a pipeline with Fourier features and ARIMA
    >>> pipe = ForecastingPipeline(
    ...             YtoX(),
    ...             FourierFeatures(sp_list=[24, 24 * 7], fourier_terms_list=[10, 5]),
    ...             ARIMA(order=(1, 1, 1))  # doctest: +SKIP,
    ...     ]
    ... )  # doctest: +SKIP
    >>>
    >>> # fit and forecast, using Fourier features as exogenous data
    >>> pred = pipe.fit_predict(y, fh=[1, 2, 3, 4, 5])  # doctest: +SKIP

    Use case: using lagged endogenous variables as exogenous data.

    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.compose import YtoX
    >>> from sktime.transformations.series.lag import Lag
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.forecasting.sarimax import SARIMAX
    >>>
    >>> # data with no exogenous features
    >>> y = load_airline()
    >>>
    >>> # create the pipeline with lagged endogenous data as exogenous data
    >>> lagged_y_trafo = YtoX() * Lag(1, index_out="original") * Imputer()
    >>>
    >>> # specify index_out="original" so ARIMA gets a 1-step-ahead forecast
    >>> # use lagged_y_trafo to generate X
    >>> forecaster = lagged_y_trafo ** SARIMAX()  # doctest: +SKIP
    >>>
    >>> # fit and forecast next value, with lagged y as exogenous data
    >>> forecaster.fit(y, fh=[1])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP

    Use case: using summarized endogenous variables as exogenous data.

    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.summarize import WindowSummarizer
    >>> from sktime.transformations.compose import YtoX
    >>> from sktime.forecasting.compose import make_reduction
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> from sklearn.ensemble import GradientBoostingRegressor  # doctest: +SKIP
    >>>
    >>> # data with no exogenous features
    >>> y = load_airline()
    >>>
    >>> # keyword arguments for WindowSummarizer
    >>> kwargs = {
    ...     "lag_feature": {
    ...         "lag": [1],
    ...         "mean": [[1, 3], [3, 6]],
    ...         "std": [[1, 4]],
    ...     },
    ...     "truncate": 'bfill',
    ... }
    >>>
    >>> # create forecaster from sklearn regressor using make_reduction
    >>> forecaster = make_reduction(
    ...     GradientBoostingRegressor(),
    ...     strategy="recursive",
    ...     pooling="global",
    ...     window_length=12,
    ... )  # doctest: +SKIP
    >>>
    >>> # create the pipeline
    >>> pipe = ForecastingPipeline(
    ...     steps=[
    ...         ("summary_features", YtoX(transformer=WindowSummarizer(**kwargs))),
    ...         ("forecaster", forecaster),
    ...     ]
    ... )  # doctest: +SKIP
    >>>
    >>> # fit and forecast, with summarized y as exogenous data
    >>> preds = pipe.fit_predict(y=y, fh=range(1, 20))  # doctest: +SKIP
    """

    _tags = {
        "authors": ["fkiraly", "avishkarsonni"],
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": False,
        "univariate-only": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "both",
        "fit_is_empty": True,
        "requires_X": False,
        "requires_y": True,
    }

    def __init__(self, subset_index=False, transformer=None):
        self.subset_index = subset_index
        self.transformer = transformer
        super().__init__()

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing core logic, called from fit

        Parameters
        ----------
        X : time series or panel in one of the pd.DataFrame formats
            Data to be fitted
        y : time series or panel in one of the pd.DataFrame formats
            Additional data, e.g., labels for fitting

        Returns
        -------
        self : reference to self
        """
        if self.transformer is not None:
            if hasattr(self.transformer, "clone"):
                self.transformer_ = self.transformer.clone()
            else:
                self.transformer_ = self.transformer
            self.transformer_.fit(y)
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : time series or panel in one of the pd.DataFrame formats
            Data to be transformed
        y : time series or panel in one of the pd.DataFrame formats
            Additional data, e.g., labels for transformation

        Returns
        -------
        y, as a transformed version of X
        """
        if self.subset_index:
            y_transformed = y.loc[X.index.intersection(y.index)]
        else:
            y_transformed = y

        # Apply the transformer if provided
        if self.transformer is not None:
            y_transformed = self.transformer_.transform(y_transformed)

        return y_transformed

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        Drops featurized column that was added in transform().

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be inverse transformed
        y : Series or Panel of mtype y_inner_mtype, optional (default=None)
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        if self.subset_index:
            return y.loc[X.index.intersection(y.index)]
        else:
            return y

    @classmethod

    def get_test_params(cls):
        """Return testing parameter settings for the YtoX transformer.
=======
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.


        Parameters
        ----------
        parameter_set : str, default="default"

            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
=======
        Name of the set of test parameters to return, for use in tests. If no
        special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : list of dict

            Parameters to create testing instances of YtoX
            Each dict can be used to construct a test instance, i.e.,
            ``YtoX(**params[i])`` creates a valid test instance.
            ``create_test_instance`` uses the first dictionary in ``params``

        """
        from sktime.transformations.series.exponent import ExponentTransformer

        return [
            {"subset_index": False, "transformer": ExponentTransformer(power=2)},
            {},
        ]
=======
        Parameters to create testing instances of the class.
        """
        param1 = {"subset_index": False}
        param2 = {"subset_index": True}
        return [param1, param2]
