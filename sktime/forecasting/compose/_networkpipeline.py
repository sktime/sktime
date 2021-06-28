# -*- coding: utf-8 -*-
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.base import _HeterogenousMetaEstimator
from sktime.utils.validation.forecasting import check_y, check_X
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sklearn.base import clone

__author__ = ["Viktor Kazakov"]
__all__ = ["NetworkPipelineForecaster"]


class NetworkPipelineForecaster(
    _OptionalForecastingHorizonMixin, _SktimeForecaster, _HeterogenousMetaEstimator
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
            2. estimator (object),
            3. arguments (dictionary) key value paris for
            fit_transform or predict method of estimator

    Examples
    --------

    `original_X` and `original_y` are key words referring to the
    X and y arguments of NetworkPipelineForecaster.fit
    and NetworkPipelineForecaster.predict

    The arguments dictionary can refer to any of the previous
    steps in the pipline by its name.

    Forecasting example. Note convention for specifying
    different arguments depending on whether
    NetworkPipelineForecaster.fit or
    NetworkPipelineForecaster.predict is called.
    >>> from sktime.transformations.series.impute import Imputer
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> pipe = NetworkPipelineForecaster(steps=[
    ...        (
    ...            "imputer",
    ...            Imputer(method="drift"),
    ...            {"fit": {"Z": "original_y"}, "predict": None, "update": None},
    ...        ),
    ...        (
    ...            "forecaster",
    ...            NaiveForecaster(strategy="mean"),
    ...            {
    ...                "fit": {"y": "imputer", "X": "original_X", "fh": "original_fh"},
    ...                "predict": {"fh": "original_fh"},
    ...                "update": {"y": "imputer"},
    ...            },
    ...        ),
    ...    ])
    """

    _required_parameters = ["steps"]
    _tags = {"univariate-only": True}

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = None
        self._step_results = {}
        self._fitted_estimators = {}
        super().__init__()

    def _iter(self, method, reverse=False):
        """
        Iterates through steps of the pipeline

        Parameters
        ----------

        method: str
            method that called `_iter`.
            Currently supported `fit`, `update` and `predict`

        Returns
        -------

            iterator with steps
        """
        if method not in ["fit", "predict", "update"]:
            raise ValueError(
                f'Iterator called by {method}. \
Iterator can be called by "fit", "predict" and "update" only.'
            )

        if reverse:
            steps = reversed(self.steps_)
        else:
            steps = self.steps_

        for (name, est, arguments) in steps:
            # if arguments were defined using the short form notation
            # without specifying behaviour for fit, predict or update
            # add the fit, predict or update, key in the arguments dictionary
            if method not in arguments:
                out = {}
                out[method] = arguments
                yield (name, est, out)
            else:
                yield (name, est, arguments)

    def _check_steps_for_values(self, step_name):
        if step_name in self._step_results:
            return self._step_results[step_name]

        return None

    def _check_steps_for_consistency(self):
        # TODO implement checks for consistency
        return self.steps

    def _process_arguments(self, arguments):
        """
        Checks arguments for consistency.

        Replace key word 'original' with X

        Replace string step name with step fit return result

        Parameters
        ----------
        arguments : dictionary
            key-value for fit() method of estimator
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
        self.steps_ = self._check_steps_for_consistency()
        self._set_y_X(y, X)
        self._set_fh(fh)
        for name, est, arguments in self._iter(method="fit"):
            # check arguments of pipeline.
            # Inspect the `fit` key of the step arguments
            # check and process corresponding values
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
            # if "Z" in processed_arguments:
            #     processed_arguments["Z"] = check_y(processed_arguments["Z"])
            if "X" in processed_arguments and processed_arguments["X"] is not None:
                if type(processed_arguments["X"]) is list:
                    args = []
                    for arg in processed_arguments["X"]:
                        args.append(check_X(arg))
                    processed_arguments["X"] = args
                else:
                    processed_arguments["X"] = check_X(processed_arguments["X"])
            if "X" in processed_arguments and processed_arguments["X"] is None:
                del processed_arguments["X"]
            # if estimator has `fit_transform()` method it is a transformer
            if hasattr(est, "fit_transform"):
                t = clone(est)
                out = t.fit_transform(**processed_arguments)
                self._step_results[name] = out
                self._fitted_estimators[name] = t  # TODO: delete
            processed_arguments["fh"] = self._fh
            # if estimator has `fit()` and `predict()` methods it s a forecaster
            if hasattr(est, "fit") and hasattr(est, "predict"):
                f = clone(est)

                f.fit(**processed_arguments)
                self._fitted_estimators[name] = f

        self._is_fitted = True

        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if X is not None:
            self._X = X
        # iterate in normal order
        for name, est, arguments in self._iter(method="predict"):
            processed_arguments = {}
            # get fitted estimator
            est = self._fitted_estimators[name]
            if "predict" in arguments and arguments["predict"] is None:
                # this step must be skipped
                continue
            elif "predict" in arguments:
                # y is not used when the steps are executed in normal order.
                if "y" in arguments["predict"]:
                    arguments["predict"].pop("y", None)
                processed_arguments = self._process_arguments(arguments["predict"])
            else:
                processed_arguments = self._process_arguments(arguments)

            # if estimator has a `transform()` method, it is a transformer
            if hasattr(est, "transform"):
                if "X" not in arguments["predict"]:
                    # process only exogenous variables, skip `y`
                    continue
                self._step_results[name] = est.transform(**processed_arguments)
            # if estimator has a `predict()` method, it is a forecaster

            if hasattr(est, "predict"):

                fitted_estimator = self._fitted_estimators[name]
                pred = fitted_estimator.predict(
                    **processed_arguments, return_pred_int=return_pred_int, alpha=alpha
                )
                # pred.index.freq = self._y.index.freq
                self._step_results[name] = pred

        # iterate in reverse order
        i = 0  # used for skipping the first estimator when iterating in
        # reverse order, i.e.
        # the last estimator if iterating in normal order
        for name, est, arguments in self._iter(method="predict", reverse=True):
            if i == 0:
                # skips the first step, i.e. the last step
                # when iterating in reverse order
                continue
            i += 1
            # Iterates through the steps in the pipeline

            processed_arguments = {}
            # get fitted estimator
            # estimator in self.steps_ is not fitted.
            # use fitted estimator in self._fitted_estimators
            est = self._fitted_estimators[name]
            # check arguments of pipeline.
            # Inspect the `predict` key of the step arguments
            # check and process corresponding values
            if "predict" in arguments and arguments["predict"] is None:
                # this step must be skipped
                continue
            if (
                ("predict" in arguments)
                and ("X" in arguments["predict"])
                and (len(arguments["predict"]) == 1)
            ):
                # not inverse transform steps that invovle only exogenous variables
                continue
            else:
                # drop X for inverse transform
                if "X" in arguments["predict"]:
                    arguments["predict"].pop("X", None)
                # process arguments of dictionary without `X` argument
                processed_arguments = self._process_arguments(arguments["predict"])

            # if estimator has a `transform()` method it is a transformer
            if hasattr(est, "inverse_transform"):
                if "y" not in processed_arguments["predict"]:
                    continue
                self._step_results[name] = est.inverse_transform(**processed_arguments)
            # if estimator has a `predict()` method it is a forecaster

            if hasattr(est, "predict"):

                fitted_estimator = self._fitted_estimators[name]
                pred = fitted_estimator.predict(
                    **processed_arguments, return_pred_int=return_pred_int, alpha=alpha
                )
                # pred.index.freq = self._y.index.freq
                self._step_results[name] = pred

        return pred

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
        y_index_frequency = self._y.index.freq
        self._update_y_X(y, X)
        self._y = self._y.asfreq(y_index_frequency)

        for (name, est, arguments) in self._iter(method="update"):
            # check arguments of pipeline.
            # Inspect the `update` key of the step arguments
            # check and process corresponding values

            # estimator in self.steps_ is not fitted.
            # use fitted estimator in self._fitted_estimators
            est = self._fitted_estimators[name]

            if "update" in arguments and arguments["update"] is None:
                # this step must be skipped
                continue
            # assume update takes same arguments as fit
            elif "update" in arguments:

                processed_arguments = self._process_arguments(arguments["update"])

            else:
                processed_arguments = self._process_arguments(arguments)

            processed_arguments["update_params"] = update_params
            # Transformers do not have `predict()`

            if hasattr(est, "update"):

                out = est.update(**processed_arguments)
                out._cutoff = self.cutoff
                self._step_results[name] = out
            else:
                raise ValueError(f"{name} has no update() method.")
            #     continue
            # forecasters have `update()` and `predict()` methods
            # if hasattr(est, "update") and hasattr(est, "predict"):
            #     est.update(**processed_arguments)

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

        return {"steps": self.steps}

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
        # super().set_params(**kwargs)
        # self._set_params("steps", **kwargs)

    def get_fitted_params(self):
        raise NotImplementedError()
