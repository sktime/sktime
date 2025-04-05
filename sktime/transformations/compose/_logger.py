"""Logging transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["Logger"]

from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose._common import CORE_MTYPES
from sktime.utils.singleton import _multiton


class Logger(BaseTransformer):
    """Logging transformer, writes data to logging, and otherwise leaves it unchanged.

    In methods, logs ``X`` and ``y`` to ``logger``.
    The ``logger`` can us as ``logger_backend`` a python ``logging`` instance,
    primarily for printing, with ``data`` logged as ``extra``,
    or a custom ``DataLog`` instance to retrieve full objects and not just printouts.

    Parameters
    ----------
    logger : str optional, default="sktime"
        logger name to use, passed to ``logger_backend`` to identify
        the unique logger instance referenced by ``get_logger``.

    logger_backend : str, one of "logging" (default), "datalog"
        Backend to use for logging.

        * "logging": uses the standard Python logging module,
          logs to ``logging.getLogger(logger)``
        * "datalog": uses a multiton logger class for easy retrieval of data,
          logs to ``DataLog(logger)``, with ``DataLog`` from
          the ``transformations.compose`` module.

        In either case, a reference to the logger can be retrieved
        by calling ``obj.get_logger``, where ``obj`` is an instance of ``Logger``.

    log_methods : str or list of str, default=``"transform"``
        if ``"all"``, will log ``fit``, ``transform``, ``inverse_transform``;
        if str or list of str, all strings must be from among the above,
        and will log exactly the methods that are passed as str;
        can also be ``"off""`` to disable logging entirely.

    level : logging level, optional, default=logging.INFO
        logging level, one of
        ``logging.INFO``, ``logging.DEBUG``, ``logging.WARNING``,
        ``logging.ERROR``

    log_fitted_params : bool, optional, default=False
        if True, will also write ``X`` and ``y`` seen in ``fit``
        to ``self`` as ``X_`` and ``y_``, these can be retrieved
        by calling ``get_fitted_params``.
        If False, ``get_fitted_params`` will return an empty dict.

    Examples
    --------
    >>> from sktime.transformations.compose import DataLog, Logger
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.series.detrend import Detrender
    >>>
    >>> # create a logger
    >>> logger = Logger(logger="foo", log_methods="all", logger_backend="datalog")
    >>> # create a pipeline that logs after detrending and before forecasting
    >>> pipe = Detrender() * logger * NaiveForecaster(sp=12)
    >>> pipe.fit(load_airline(), fh=[1, 2, 3])
    TransformedTargetForecaster(...)
    >>> # get the log
    >>> log = DataLog("foo").get_log()
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
        logger_backend="logging",
        log_methods="all",
        level=None,
        log_fitted_params=False,
    ):
        self.logger = logger
        self.logger_backend = logger_backend
        self.log_methods = log_methods
        self.level = level
        self.log_fitted_params = log_fitted_params
        super().__init__()

        if self.log_methods == "all":
            self._log_methods = ["fit", "transform", "inverse_transform"]
        elif self.log_methods == "off":
            self._log_methods = []
        elif isinstance(self.log_methods, str):
            self._log_methods = [self.log_methods]
        else:
            self._log_methods = self.log_methods

        if self.level is None:
            import logging

            self._level = logging.INFO
        else:
            self._level = self.level

    @property
    def get_logger(self):
        if self.logger_backend == "logging":
            import logging

            return logging.getLogger(self.logger)
        elif self.logger_backend == "datalog":
            return DataLog(self.logger)

    def _write_log(self, key, obj):
        """Write log to logger.

        Parameters
        ----------
        key : str
            key to identify the object
        obj : any
            object to log
        """
        if self.logger_backend == "logging":
            self.get_logger.log(self._level, key, extra=obj)
        elif self.logger_backend == "datalog":
            self.get_logger.log(key, obj)

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
            self._write_log("fit", {"X": X, "y": y})
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
            self._write_log("transform", {"X": X, "y": y})
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
            self._write_log("inverse_transform", {"X": X, "y": y})
        return X

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
        params : dict or list of dict
        """
        import logging

        params0 = {}
        params1 = {"logger": "bar", "log_methods": "off"}
        params2 = {
            "log_methods": ["transform", "inverse_transform"],
            "log_fitted_params": True,
            "level": logging.ERROR,
        }
        params3 = {"logger": "foo", "level": logging.DEBUG, "log_methods": "all"}
        params4 = {"logger": "foo", "logger_backend": "datalog", "log_methods": "all"}
        return [params0, params1, params2, params3, params4]


@_multiton
class DataLog:
    """Data logger.

    Identified uniquely by a key, logs data in a list.
    List contains tuples of (log_key : str, data : any).

    Methods as below:

    * ``log`` appends a tuple to the list.
    * ``get_log`` returns the list.
    * ``reset`` empties the list.

    Parameters
    ----------
    key : str
        key to identify the logger
    """

    def __init__(self, key):
        self.key = key
        self._log = []

    def log(self, log_key, data):
        """Log data.

        Appends the following tuple to the log list: ``(log_key, data)``.

        The full log can be read by calling ``get_log``.

        Parameters
        ----------
        log_key : str
            key to identify the data
        data : any
            data to log
        """
        self._log.append((log_key, data))
        return self

    def reset(self):
        """Reset the log to the empty list."""
        self._log = []
        return self

    def get_log(self):
        """Read the log.

        Returns
        -------
        log : list of tuple
            Reference to the list containing the logged data,
            list of tuples of (log_key : str, data : any)
        """
        return self._log
