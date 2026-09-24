"""Pre-fitted forecaster, initialized from serialized object."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import pickle
from copy import deepcopy

from sktime.exceptions import NotFittedError
from sktime.forecasting.base._base import BaseForecaster
from sktime.forecasting.base._delegate import _DelegatedForecaster


class FittedForecaster(_DelegatedForecaster):
    """Pre-fitted forecaster, from serialized object.

    `FittedForecaster` wraps a serialized object that is already fitted.

    Depending on parameters, `fit` is empty, or behaves like an `update`.
    Where the state of this forecaster is updated, the `fitted_forecaster` is unchanged.

    Parameters
    ----------
    fitted_forecaster : serialized forecaster, one of the following
        sktime forecaster object (BaseForecaster) in fitted state, after calling `fit`
        pickled sktime forecaster object
    update_forecaster : bool, optional, default = True
        whether to update the forecaster with new data
        if yes, `fit` and `update` will update a deepcopy of fitted_forecaster
    unwrap_time : str, one of "construct", "fit", optional, default = "construct"
        when to unwrap the forecaster from its serialized state
        "construct" = at construction, i.e., when calling `__init__`
        "fit" = in `fit`

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.serialize import FittedForecaster

    >>> y_train = load_airline()[:36]
    >>> y_deploy = load_airline()[36:]
    >>> pickled_forecaster = pickle.dumps(NaiveForecaster(sp=12).fit(y_train))
    >>> fitted_forecaster = FittedForecaster(pickled_forecaster)
    >>> fitted_forecaster.fit(y_deploy)
    FittedForecaster(...)
    >>> y_pred = fitted_forecaster.predict(fh=12)
    """

    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "scitype:y": "both",
        "y_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "fit_is_empty": False,
    }

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods to those of same name in self.fitted_forecaster_
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "fitted_forecaster_"

    def __init__(
        self, fitted_forecaster, update_forecaster=True, unwrap_time="construct"
    ):
        super().__init__()
        self.fitted_forecaster = fitted_forecaster
        self.update_forecaster = update_forecaster
        self.unwrap_time = unwrap_time

        if unwrap_time == "construct":
            self.fitted_forecaster_ = self._unwrap_serialized(fitted_forecaster)
            self.clone_tags(self.fitted_forecaster_)
        elif unwrap_time == "fit":
            self.fitted_forecaster_ = None
            self.set_tags(**self._unwrap_serialized_tags(fitted_forecaster))
        else:
            raise ValueError(
                f'unwrap_time must be one of "construct" or "fit", but'
                f" found {unwrap_time}"
            )

        if unwrap_time == "fit" or update_forecaster:
            self.set_tags(**{"fit_is_empty": False})
        else:
            self.set_tags(**{"fit_is_empty": True})

    def clone(self):
        """Obtain a clone of the object with same hyper-parameters.

        A clone is a different object without shared references, in post-init state.
        This function is equivalent to returning sklearn.clone of self.
        Equal in value to `type(self)(**self.get_params(deep=False))`.

        Returns
        -------
        instance of type(self), clone of self (see above)
        """
        selfcopy = deepcopy(self)
        delattr(selfcopy, "fitted_forecaster_")
        delattr(selfcopy, "_tags_dynamic")
        return selfcopy

    def _unwrap_serialized(self, fitted_forecaster):
        """Unwraps serialized object to sktime forecaster.

        Parameters
        ----------
        fitted_forecaster : serialized forecaster, one of the following
            sktime forecaster object (BaseForecaster) in fitted state, after `fit`
            pickled sktime forecaster object

        Returns
        -------
        unwrapped_forecaster : sktime forecaster object (BaseForecaster)
        """
        if isinstance(fitted_forecaster, BaseForecaster):
            unwrapped_forecaster = deepcopy(fitted_forecaster)

        if isinstance(fitted_forecaster, bytes):
            unwrapped_forecaster = pickle.loads(fitted_forecaster)

        if not unwrapped_forecaster.is_fitted:
            raise NotFittedError(
                f"fitted_forecaster of type {self.__class__.__name__} has not "
                f"been fitted yet; please call `fit` first."
            )

        return unwrapped_forecaster

    def _unwrap_serialized_tags(self, fitted_forecaster):
        """Get tags from serialized forecaster.

        Parameters
        ----------
        fitted_forecaster : serialized forecaster, one of the following
            sktime forecaster object (BaseForecaster) in fitted state, after `fit`
            pickled sktime forecaster object

        Returns
        -------
        tags : sktime tags dict
        """
        if isinstance(fitted_forecaster, BaseForecaster):
            tags = fitted_forecaster.get_tags()

        if isinstance(fitted_forecaster, bytes):
            tags = pickle.loads(fitted_forecaster).get_tags()

        return tags

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        y : sktime compatible time series object
            Time series to which to fit the forecaster
        X : sktime compatible time series object, optional (default=None)
            Exogeneous time series for fitting the forecaster
        fh : ignored, self is already fitted

        Returns
        -------
        self : returns an instance of self.
        """
        if self.unwrap_time == "fit":
            self.fitted_forecaster_ = self._unwrap_serialized(self.fitted_forecaster)

        if self.update_forecaster:
            self.fitted_forecaster_.update(y=y, X=X, update_params=True)
        return self

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        Parameters
        ----------
        y : sktime compatible time series object
            Time series with which to update the forecaster
        X : sktime compatible time series object, optional (default=None)
            Exogeneous time series for updating the forecaster
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        if self.update_forecaster:
            self.fitted_forecaster_.update(y=y, X=X, update_params=update_params)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.utils._testing.series import _make_series

        y = _make_series()
        fitted_forecaster = NaiveForecaster().fit(y)
        pickled_forecaster = pickle.dumps(fitted_forecaster)

        params1 = {"fitted_forecaster": fitted_forecaster}
        params2 = {"fitted_forecaster": pickled_forecaster, "unwrap_time": "fit"}
        return [params1, params2]
