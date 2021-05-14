# -*- coding: utf-8 -*-
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.base import _HeterogenousMetaEstimator
from sktime.transformations.base import _SeriesToSeriesTransformer
from sktime.utils.validation.forecasting import check_y, check_X
from sktime.forecasting.base._base import DEFAULT_ALPHA


__author__ = ["Viktor Kazakov"]
__all__ = ["NetworkPipelineForecaster"]


class NetworkPipelineForecaster(
    _OptionalForecastingHorizonMixin,
    _SktimeForecaster,
    _HeterogenousMetaEstimator,
    _SeriesToSeriesTransformer,
):
    """
    Prototype of non sequential pipeline mimicking a network.

    Each step in the pipeline can use the output of any of the previous steps or
    the original input values to the fit and predict methods of the pipeline.

    The arguments of the steps in the pipeline must be provided as a dictionary.
    The dictionary is substituted in the fit_transform, fit or predict function calls
    of the step transformer or estimator.

    Parameters
    ----------
    steps : array of lists
        list comprised of three elements:
            1. name of step (string),
            2. algorithm (object),
            3. arguments (dictionary) key value paris for
            fit_transform or predict method of algorithm

    Examples
    --------
    `original_X` and `original_y` are key words referring to the
    X and y arguments of NetworkPipeline.fit
    and NetworkPipeline.predict

    The arguments dictionary can refer to any of the previous
    steps in the pipline by its name.

    Forecasting example. Note convention for specifying
    different arguments depending on whether
    NetworkPipeline.fit or NetworkPipeline.predict is called.
    >>> pipe = NetworkPipeline(
        steps=[
            (
                "imputer",
                Imputer(method="drift"),
                {"fit": {"Z": "original_y"}, "predict": None},
            ),
            (
                "forecaster",
                forecaster,
                {"fit": {"y": "imputer"}, "predict": {"fh": "original_y"}},
            ),
        ]
    )

    Classification example.
    >>>  pipe = NetworkPipeline(
        steps=[
            ("tabularizer", Tabularizer(), {"X": "original_X"}),
            (
                "classifier",
                DummyClassifier(strategy="prior"),
                {
                    "fit": {"X": "tabularizer", "y": "original_y"},
                    "predict": {"X": "tabularizer"},
                },
            ),
        ]

    See the tests folder for a working implementations of these examples.
    """

    _required_parameters = ["steps"]
    _tags = {"univariate-only": True}

    def __init__(self, steps):
        self.steps = steps
        self._step_results = {}
        super().__init__()

    def _iter(self):
        for name, alg, arguments in self.steps:
            yield name, alg, arguments

    def _check_steps_for_values(self, step_name):
        if step_name in self._step_results:
            return self._step_results[step_name]

        return None

    def _process_arguments(self, arguments):
        """
        Checks arguments for consistency.

        Replace key word 'original' with X

        Replace string step name with step fit return result

        Parameters
        ----------
        arguments : dictionary
            key-value for fit() method of algorithm
        """

        returned_arguments_kwarg = arguments.copy()
        if arguments is None:
            return arguments
        for argument_name, argument_value in returned_arguments_kwarg.items():
            if argument_value == "original_X":
                returned_arguments_kwarg[argument_name] = self._X
                continue
            if argument_value == "original_y":
                returned_arguments_kwarg[argument_name] = self._y
                continue
            if argument_value == "original_fh":
                returned_arguments_kwarg[argument_name] = self._fh
                continue

            if type(argument_value) == list:
                out = []
                for list_val in argument_value:
                    returned_step_value = self._check_steps_for_values(list_val)
                    if returned_step_value is not None:
                        out.append(returned_step_value)
                returned_arguments_kwarg[argument_name] = out
                continue
            # go through all steps and look for returned values
            returned_step_value = self._check_steps_for_values(argument_value)
            if returned_step_value is not None:
                returned_arguments_kwarg[argument_name] = returned_step_value

        return returned_arguments_kwarg

    def fit(self, y, X=None, fh=None):
        self._X = X
        self._y = y
        self._set_fh(fh)
        for name, alg, arguments in self._iter():
            processed_arguments = {}

            if "fit" in arguments and arguments["fit"] is None:
                # this step must be skipped
                continue
            elif "fit" in arguments:
                processed_arguments = self._process_arguments(arguments["fit"])
            else:
                processed_arguments = self._process_arguments(arguments)

            if "y" in processed_arguments:
                processed_arguments["y"] = check_y(processed_arguments["y"])
            if "Z" in processed_arguments:
                processed_arguments["Z"] = check_y(processed_arguments["Z"])
            if "X" in processed_arguments and processed_arguments["X"] is not None:
                processed_arguments["X"] = check_X(processed_arguments["X"])
            if "X" in processed_arguments and processed_arguments["X"] is None:
                del processed_arguments["X"]
            # Transformers are instances of BaseTransformer and BaseEstimator
            # Estimators are only instances of BaseEstimator
            if hasattr(alg, "fit_transform"):
                out = alg.fit_transform(**processed_arguments)
                self._step_results[name] = out
                self._is_fitted = True
            # estimators have fit and predict methods
            if hasattr(alg, "fit") and hasattr(alg, "predict"):
                alg.fit(**processed_arguments)
                self._is_fitted = True

        return self

    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if fh is not None:
            self._set_fh(fh)
        if X is not None:
            self._X = X
        for name, alg, arguments in self._iter():
            processed_arguments = {}

            if "predict" in arguments and arguments["predict"] is None:
                # this step must be skipped
                continue
            elif "predict" in arguments:
                processed_arguments = self._process_arguments(arguments["predict"])
            else:
                processed_arguments = self._process_arguments(arguments)
            if hasattr(alg, "transform"):
                self._step_results[name] = alg.transform(**processed_arguments)
            if hasattr(alg, "predict"):
                self._step_results[name] = alg.predict(
                    **processed_arguments, return_pred_int=return_pred_int, alpha=alpha
                )

        return self._step_results[self.steps[-1][0]]

    def update(self, y, X=None, update_params=True):
        """Update fitted parameters

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=True)

        Returns
        -------
        self : an instance of self
        """
        self.check_is_fitted()
        self._update_y_X(y, X)
        for name, alg, arguments in self._iter():
            if "update" in arguments and arguments["update"] is None:
                # this step must be skipped
                continue
            # assume update takes same arguments as fit
            elif "update" in arguments:
                processed_arguments = self._process_arguments(arguments["update"])
            else:
                processed_arguments = self._process_arguments(arguments)
            # Transformers are instances of BaseTransformer and BaseEstimator
            # Estimators are only instances of BaseEstimator

            if hasattr(alg, "update"):
                out = alg.update(**processed_arguments)
                self._step_results[name] = out
                self._is_fitted = True
            # estimators have fit and predict methods
            if hasattr(alg, "update") and hasattr(alg, "predict"):
                alg.update(**processed_arguments)
                self._is_fitted = True

        return self

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """

        return self._get_params("steps", deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params("steps", **kwargs)

    def get_fitted_params(self):
        raise NotImplementedError()
