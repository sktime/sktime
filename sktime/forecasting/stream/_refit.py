# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Stream compositor that pools data and refits the inner forecaster."""

__all__ = ["RefitForecaster"]

from sktime.datatypes import ALL_TIME_SERIES_MTYPES, update_data
from sktime.forecasting.base._delegate import _DelegatedForecaster


class RefitForecaster(_DelegatedForecaster):
    """Refit inner forecaster on all data seen so far when ``update`` is called.

    Owns explicit data pools ``_y_pool`` and ``_X_pool`` instead of relying on
    ``BaseForecaster``'s implicit ``self._y`` / ``self._X`` cache. This is the
    recommended pattern for streaming refit behaviour when ``remember_data=False``.

    Parameters
    ----------
    forecaster : sktime forecaster
        Forecaster to refit on the pooled data.

    Attributes
    ----------
    forecaster_ : sktime forecaster
        Clone of ``forecaster``, fitted on the data pool.
    _y_pool : sktime compatible time series
        All endogenous data seen in ``fit`` and ``update``.
    _X_pool : sktime compatible time series or None
        All exogenous data seen in ``fit`` and ``update``.

    Examples
    --------
    >>> from sktime.forecasting.trend import TrendForecaster
    >>> from sktime.forecasting.stream import RefitForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> y0, y1 = y.iloc[:-12], y.iloc[-12:]
    >>> forecaster = RefitForecaster(TrendForecaster())
    >>> forecaster.fit(y0, fh=[1, 2, 3])
    RefitForecaster(...)
    >>> forecaster.update(y1)
    RefitForecaster(...)
    >>> forecaster.predict()
    """

    _delegate_name = "forecaster_"

    _tags = {
        "authors": "Faakhir30",
        "capability:update": True,
        "fit_is_empty": False,
        "requires-fh-in-fit": False,
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "tests:core": True,
    }

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.forecaster_ = forecaster.clone()
        self._y_pool = None
        self._X_pool = None

        super().__init__()
        self._set_delegated_tags(self.forecaster_)

    def _update_y_X(self, y, X=None, enforce_index_type=None):
        """Update cutoff only; data pooling is handled in ``_fit`` / ``_update``."""
        if y is not None:
            from sktime.datatypes import VectorizedDF

            y_for_cutoff = y.X_multiindex if isinstance(y, VectorizedDF) else y
            self._set_cutoff_from_y(y_for_cutoff)

    def _fit(self, y, X, fh):
        self._y_pool = y
        self._X_pool = X
        self.forecaster_.fit(y=y, X=X, fh=fh)
        return self

    def _update(self, y, X=None, update_params=True):
        self._y_pool = update_data(self._y_pool, y)
        if X is not None:
            if self._X_pool is None:
                self._X_pool = X
            else:
                self._X_pool = update_data(self._X_pool, X)

        if update_params:
            self.forecaster_.fit(y=self._y_pool, X=self._X_pool, fh=self._fh)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from sktime.forecasting.trend import TrendForecaster

        return {"forecaster": TrendForecaster.create_test_instance()}
