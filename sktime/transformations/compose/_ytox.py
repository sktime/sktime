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

    >>> from sktime.transformations.series.fourier import FourierFeatures
    >>> from sktime.transformations.series.ytox import YtoX  # doctest: +SKIP
    >>> from sktime.forecasting.compose import ForecastingPipeline  # doctest: +SKIP
    >>> from sktime.forecasting.arima import ARIMA  # doctest: +SKIP
    >>>
    >>> # create a pipeline with Fourier features and ARIMA
    >>> ARIMA = ARIMA(order=(1, 1, 1))  # doctest: +SKIP
    >>>
    >>> pipe = ForecastingPipeline(
    ...     [
    ...         YtoX(),
    ...         FourierFeatures(sp_list=[24, 24 * 7], fourier_terms_list=[10, 5]),
    ...         ARIMA,
    ...     ]
    ... )  # doctest: +SKIP

    Use case: using lagged endogenous variables as exogeneous data.

    >>> from sktime.forecasting.sarimax import SARIMAX
    >>> from sktime.transformations.compose import YtoX
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.transformations.series.lag import Lag
    >>>
    >>> lagged_y_trafo = YtoX() * Lag(1, index_out="original") * Imputer()
    >>> # we need to specify index_out="original" as otherwise ARIMA gets 1 and 2 ahead
    >>> # use laggged_y_trafo to generate X
    >>> forecaster = lagged_y_trafo ** SARIMAX()  # doctest: +SKIP
    >>>
    >>> # fit and forecast next value
    >>> forecaster.fit(y_train, fh=[1])  # doctest: +SKIP
    >>> forecaster.predict()  # doctest: +SKIP
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
