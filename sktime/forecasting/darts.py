# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to estimators from darts by Unit8."""
from typing import Optional, Union

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base.adapters._darts import _DartsAdapter

__author__ = ["yarnabrina", "fnhirwa"]


class DartsXGBModel(_DartsAdapter):
    """Darts XGBModel Estimator.

    This is based on implementation of XGBoost Model in darts [1]_ by Unit8.

    Parameters
    ----------
    past_covariates : Optional[List[str]], optional
        column names in ``X`` which are known only for historical data, by default None
    num_samples : Optional[int], optional
        Number of times a prediction is sampled from a probabilistic model, by default
        1000
    lags
        Lagged target values used to predict the next time step. If an integer is given
        the last `lags` past lags are used (from -1 backward). Otherwise a list of
        integers with lags is required (each lag must be < 0). If a dictionary is given,
        keys correspond to the component names
        (of first series when using multiple series) and
        the values correspond to the component lags(integer or list of integers).
    lags_past_covariates
        Number of lagged past_covariates values used to predict the next time step. If
        an integer is given the last `lags_past_covariates` past lags are used
        (inclusive, starting from lag -1). Otherwise a list of integers
        with lags < 0 is required. If a dictionary is given, keys correspond to the
        past_covariates component names(of first series when using multiple series)
        and the values correspond to the component lags(integer or list of integers).
    lags_future_covariates
        Number of lagged future_covariates values used to predict the next time step. If
        a tuple (past, future) is given the last `past` lags in the past are used
        (inclusive, starting from lag -1) along with the first `future` future lags
        (starting from 0 - the prediction time - up to `future - 1` included). Otherwise
        a list of integers with lags is required. If dictionary is given,
        keys correspond to the future_covariates component names
        (of first series when using multiple series) and the values
        correspond to the component lags(integer or list of integers).
    output_chunk_length
        Number of time steps predicted at once by the internal regression model. Does
        not have to equal the forecast horizon `n` used in `predict()`. However, setting
        `output_chunk_length` equal to the forecast horizon may be useful if the
        covariates don't extend far enough into the future.
    add_encoders
        A large number of past and future covariates can be automatically generated with
        `add_encoders`. This can be done by adding multiple pre-defined index encoders
        and/or custom user-made functions that will be used as index encoders.
        Additionally, a transformer such as Darts' :class:`Scaler` can be added to
        transform the generated covariates. This happens all under one hood and only
        needs to be specified at model creation. Read
        :meth:`SequentialEncoder <darts.dataprocessing.encoders.SequentialEncoder>`
        to find out more about ``add_encoders``. Default: ``None``. An example showing
        some of ``add_encoders`` features:

        .. highlight:: python
        .. code-block:: python

            add_encoders={
                'cyclic': {'future': ['month']},
                'datetime_attribute': {'future': ['hour', 'dayofweek']},
                'position': {'past': ['relative'], 'future': ['relative']},
                'custom': {'past': [lambda idx: (idx.year - 1950) / 50]},
                'transformer': Scaler()
            }
        ..
    likelihood
        Can be set to `poisson` or `quantile`. If set, the model will be probabilistic,
        allowing sampling at prediction time. This will overwrite any `objective`
        parameter.
    quantiles
        Fit the model to these quantiles if the `likelihood` is set to `quantile`.
    random_state
        Control the randomness in the fitting procedure and for sampling. Default:
        ``None``.
    multi_models
        If True, a separate model will be trained for each future lag to predict. If
        False, a single model is trained to predict at step 'output_chunk_length' in the
        future. Default: True.
    use_static_covariates
        Whether the model should use static covariate information in case the input
        `series` passed to ``fit()`` contain static covariates. If ``True``, and static
        covariates are available at fitting time, will enforce that all target `series`
        have the same static covariate dimensionality in ``fit()`` and ``predict()``.
    kwargs
        Additional keyword arguments passed to `xgb.XGBRegressor`.
        Passed as a dictionary to conform to sklearn's API. Default: ``None``.

    References
    ----------
    .. [1] https://github.com/unit8co/darts

    Notes
    -----
    If unspecified, all columns will be assumed to be known during prediction duration.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["yarnabrina", "fnhirwa"],
        "maintainers": ["yarnabrina", "fnhirwa"],
        # estimator type
        # --------------
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "capability:pred_int": True,
        "capability:insample": False,
    }

    def __init__(
        self: "DartsXGBModel",
        past_covariates: Optional[list[str]] = None,
        num_samples: Optional[int] = 1000,
        lags: Optional[Union[int, list[int], dict[str, Union[int, list[int]]]]] = None,
        lags_past_covariates: Optional[
            Union[int, list[int], dict[str, Union[int, list[int]]]]
        ] = None,
        lags_future_covariates: Optional[
            Union[
                tuple[int, int], list[int], dict[str, Union[tuple[int, int], list[int]]]
            ]
        ] = None,
        output_chunk_length: Optional[int] = 1,
        add_encoders: Optional[dict] = None,
        likelihood: Optional[str] = None,
        quantiles: Optional[list[float]] = None,
        random_state: Optional[int] = None,
        multi_models: Optional[bool] = True,
        use_static_covariates: Optional[bool] = True,
        kwargs: Optional[dict] = None,
    ) -> None:
        self.lags = lags
        self.lags_past_covariates = lags_past_covariates
        self.lags_future_covariates = lags_future_covariates
        self.output_chunk_length = output_chunk_length
        self.add_encoders = add_encoders
        self.likelihood = likelihood
        self.quantiles = quantiles
        self.random_state = random_state
        self.multi_models = multi_models
        self.use_static_covariates = use_static_covariates
        self.kwargs = kwargs
        self.handle_kwargs()

        super().__init__(past_covariates=past_covariates, num_samples=num_samples)

    def handle_kwargs(self: "DartsXGBModel") -> None:
        """Handle additional keyword arguments."""
        if self.kwargs is not None:
            for key, value in self.kwargs.items():
                setattr(self, key, value)

    def _create_forecaster(self: "DartsXGBModel"):
        """Create Darts model."""
        from darts.models import XGBModel

        return XGBModel(
            lags=self.lags,
            lags_past_covariates=self.lags_past_covariates,
            lags_future_covariates=self.lags_future_covariates,
            output_chunk_length=self.output_chunk_length,
            add_encoders=self.add_encoders,
            likelihood=self.likelihood,
            quantiles=self.quantiles,
            random_state=self.random_state,
            multi_models=self.multi_models,
            use_static_covariates=self.use_static_covariates,
            kwargs=self.kwargs,
        )

    @property
    def _lags_past_covariates(
        self: "DartsXGBModel",
    ) -> Union[int, list[int], dict[str, Union[int, list[int]]], None]:
        return self.lags_past_covariates

    @property
    def _lags_future_covariates(
        self: "DartsXGBModel",
    ) -> Union[
        tuple[int, int], list[int], dict[str, Union[dict[int, int], list[int]]], None
    ]:
        return self.lags_future_covariates

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """
        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        if _check_soft_dependencies(
            "u8darts", package_import_alias={"u8darts": "darts"}, severity="none"
        ):
            params = [
                {
                    "num_samples": 100,
                    "lags": 12,
                    "output_chunk_length": 1,
                    "add_encoders": None,
                    "likelihood": "quantile",
                    "quantiles": None,
                    "random_state": None,
                    "multi_models": True,
                    "use_static_covariates": True,
                    "kwargs": None,
                }
            ]
        else:
            params = [{}]
        return params


__all__ = ["DartsXGBModel"]
