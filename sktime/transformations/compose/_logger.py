"""Logging transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["Logger"]

from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose._common import CORE_MTYPES


class Logger(BaseTransformer):
    """Logging transformer, writes data to logging, and otherwise leaves it unchanged.

    In methods, logs ``X`` and ``y`` to ``logger`` as ``extra`` log,
    i.e., the full object.

    Parameters
    ----------
    logger : str, optional, default="sktime"
        logger name, logs to ``logging.getLogger(logger)``
    log_methods : str or list of str, default=None
        if None, will log ``fit``, ``transform``, ``inverse_transform``;
        if str or list of str, all strings must be from among the above,
        and will log exactly the methods that are passed as str
    level : logging level, optional, default=logging.INFO
        logging level, one of
        ``logging.INFO``, ``logging.DEBUG``, ``logging.WARNING``,
        ``logging.ERROR``
    log_fitted_params : bool, optional, default=False
        if True, will also write ``X`` and ``y`` seen in ``fit``
        to ``self`` as ``X_`` and ``y_``, these can be retrieved
        by calling ``get_fitted_params``.
        If False, ``get_fitted_params`` will return an empty dict.
    """

    _tags = {
        "authors": "fkiraly",
        "capability:inverse_transform": True,  # can the transformer inverse transform?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": CORE_MTYPES,  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": CORE_MTYPES,  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": True,
        # does transform return have the same time index as input X
        "handles-missing-data": True,  # can estimator handle missing data?
    }

    def __init__(
        self,
        logger="sktime",
        log_methods=None,
        level=None,
        log_fitted_params=False,
    ):
        self.logger = logger
        self.log_methods = log_methods
        self.level = level
        self.log_fitted_params = log_fitted_params
        super().__init__()

        if self.log_methods is None:
            self._log_methods = ["fit", "transform", "inverse_transform"]
        elif isinstance(self.log_methods, str):
            self._log_methods = [self.log_methods]
        else:
            self._log_methods = self.log_methods

        if self.level is None:
            import logging

            self._level = logging.INFO
        else:
            self._level = self.level

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : pd.DataFrame
            if self.get_tag("univariate-only")==True:
                guaranteed to have a single column
            if self.get_tag("univariate-only")==False: no restrictions apply
        y : None, present only for interface compatibility

        Returns
        -------
        self: reference to self
        """
        if "fit" in self._log_methods:
            self.logger.log(self._level, "fit", extra={"X": X, "y": y})
        if self.log_fitted_params:
            self.X_ = X
            self.y_ = y
        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : any sktime compatible data, Series, Panel, or Hierarchical
        y : optional, default=None
            ignored, argument present for interface conformance

        Returns
        -------
        X, identical to input
        """
        if "transform" in self._log_methods:
            self.logger.log(self._level, "transform", extra={"X": X, "y": y})
        return X

    def _inverse_transform(self, X, y=None):
        """Inverse transform X and return an inverse transformed version.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : any sktime compatible data, Series, Panel, or Hierarchical
        y : optional, default=None
            ignored, argument present for interface conformance

        Returns
        -------
        X, identical to input
        """
        if "inverse_transform" in self._log_methods:
            self.logger.log(self._level, "inverse_transform", extra={"X": X, "y": y})
        return X

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict
        """
        return {}
