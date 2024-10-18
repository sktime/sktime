"""Use endogeneous as exogeneous features transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["YtoX"]

from sktime.transformations.base import BaseTransformer


class YtoX(BaseTransformer):
    """Create exogeneous features which are a copy of the endogenous data.

    Replaces exogeneous features (``X``) by endogeneous data (``y``).

    To *add* instead of *replace*, use ``FeatureUnion``.

    Common use cases include:

    * creating exogeneous variables from transformed endogenous variables
    * creating exogeneous data from index, if no exogeneous data is available
    * manual construction of reduction strategies, in combination with ``YfromX``

    Parameters
    ----------
    subset_index : boolean, optional, default=False
        if True, subsets the output of ``transform`` to ``X.index``,
        i.e., outputs ``y.loc[X.index]``

    Examples
    --------
    Use case: creating exogenous data from index, if no exogenous data is available.

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
    ...     [
    ...             YtoX(),
    ...             FourierFeatures(sp_list=[24, 24 * 7], fourier_terms_list=[10, 5]),
    ...             ARIMA(order=(1, 1, 1))  # doctest: +SKIP,
    ...     ]
    ... )  # doctest: +SKIP
    >>>
    >>> # fit and forecast, using Fourier features as exogenous data
    >>> pred = pipe.fit_predict(y, fh=[1, 2, 3, 4, 5])  # doctest: +SKIP

    Use case: using lagged endogenous variables as exogeneous data.

    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.compose import YtoX
    >>> from sktime.transformations.series.lag import Lag
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.forecasting.sarimax import SARIMAX
    >>>
    >>> # data with no exogenous features
    >>> y = load_airline()
    >>>
    >>> # create the pipeline
    >>> lagged_y_trafo = YtoX() * Lag(1, index_out="original") * Imputer()
    >>>
    >>> # we need to specify index_out="original" as otherwise ARIMA gets 1 and 2 ahead
    >>> # use lagged_y_trafo to generate X
    >>> forecaster = lagged_y_trafo ** SARIMAX()  # doctest: +SKIP
    >>>
    >>> # fit and forecast next value, with lagged y as exogenous data
    >>> forecaster.fit(y, fh=[1])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP

    Use case: using summarized endogenous variables as exogeneous data.

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
    ...         ("ytox", YtoX()),
    ...         ("summarizer", WindowSummarizer(**kwargs)),
    ...         ("forecaster", forecaster),
    ...     ]
    ... )  # doctest: +SKIP
    >>>
    >>> # fit and forecast, with summarized y as exogenous data
    >>> preds = pipe.fit_predict(y=y, fh=range(1, 20))  # doctest: +SKIP
    """

    _tags = {
        "authors": ["fkiraly"],
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

    def __init__(self, subset_index=False):
        self.subset_index = subset_index

        super().__init__()

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
            return y.loc[X.index.intersection(y.index)]
        else:
            return y

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
