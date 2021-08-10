# -*- coding: utf-8 -*-
# from sktime.forecasting.base._sktime import _SktimeForecaster
# from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
# from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sklearn.base import clone
from sktime.forecasting.base import BaseForecaster

__author__ = ["Viktor Kazakov"]
__all__ = ["NetworkPipelineForecaster"]


class NetworkPipelineForecaster(BaseForecaster):
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
    >>> from sktime.transformations.panel.dataset_manipulation import Selector
    >>> from sktime.transformations.panel.dataset_manipulation import Converter
    >>> from sktime.transformations.panel.dataset_manipulation import Concatenator
    >>> from sktime.transformations.series.boxcox import BoxCoxTransformer
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> from sktime.forecasting.arima import AutoARIMA
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=4)
    >>> X_train, X_test = temporal_train_test_split(X, test_size=4)
    >>> pipe = NetworkPipelineForecaster([
    ... ("feature_X1", Selector(1, return_dataframe=False), { "X": "original_X"}),
    ... ("feature_X2", Selector(2, return_dataframe=False), { "X": "original_X"}),
    ... ("ft1", BoxCoxTransformer(), { "Z": "feature_X1"}),
    ... ("ft1_converted", Converter(), {"obj":"ft1", "to_type": "pd.DataFrame",
    ...     "as_scitype": "Series"}),
    ... ("ft2", TabularToSeriesAdaptor(MinMaxScaler()), { "Z": "feature_X2"}),
    ... ("concat", Concatenator(), { "X": ["ft1_converted","ft2"] }),
    ... ("new_y", TabularToSeriesAdaptor(MinMaxScaler()), {"Z":"original_y"}),
    ... ("y_out", AutoARIMA(suppress_warnings=True), {
    ...    "fh":"original_fh", "y": "new_y", "X": "concat"})
    ... ])
    >>> pipeline = pipe.fit(y_train,X_train)
    >>> predictions = pipe.predict(fh=[1,2,3,4], X=X_test)
    """

    _required_parameters = ["steps"]
    _tags = {"univariate-only": False, "requires-fh-in-fit": False}

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = None
        self._step_results_fit = {}
        self._step_results_predict = {}
        self._step_results_update = {}
        self._fitted_estimators = {}
        self._y_transformers = []
        super().__init__()

    def _iter(self, method):
        """
        Iterate through steps of the pipeline.

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

        for (name, est, arguments) in self.steps_:
            # if arguments were defined using the short form notation
            # without specifying behaviour for fit, predict or update
            # add the fit, predict or update, key in the arguments dictionary
            if method not in arguments:
                out = {}
                out[method] = arguments
                yield (name, est, out)
            else:
                yield (name, est, arguments)

    def _check_steps_for_values(self, step_name, method):
        step_results_dict = {
            "fit": self._step_results_fit,
            "predict": self._step_results_predict,
            "update": self._step_results_update,
        }
        if step_name in step_results_dict[method]:
            return step_results_dict[method][step_name]

        return None

    def _find_step_by_name(self, step_name):
        """returns single step by name

        Parameters
        ----------
        step_name : string
            name of step to look up

        Returns
        -------
            Step or False (if not found)
        """
        for name, est, arguments in self._iter(method="predict"):
            if name == step_name:
                return (name, est, arguments)
        return False

    def _check_steps_for_consistency(self):
        # TODO implement checks for consistency
        return self.steps

    def _process_arguments(self, step_name, arguments, method):
        """
        Check arguments for consistency.

        Replace key word 'original' with X

        Replace string step name with step fit return result

        Parameters
        ----------
        step_name : string
            Used only for appending self._y_transformers
            if a transformer is a transformer of `y`
        arguments : dictionary
            key-value for fit() method of estimator
        method : string
            method from which the call was originated.
            Acceptable values: `fit`, `predict` and `update`
        """
        acceptable_mathods = ["fit", "predict", "update"]

        if method not in acceptable_mathods:
            raise ValueError(f"Only {acceptable_mathods} calls are acceptable.")
        arguments = arguments[method]
        returned_arguments_kwarg = arguments.copy()
        if arguments is None:
            return arguments
        for argument_name, argument_value in returned_arguments_kwarg.items():
            if argument_value == "original_X":
                returned_arguments_kwarg[argument_name] = self._X
                continue
            if argument_value == "original_y":
                if method == "fit":
                    self._y_transformers.append(step_name)
                returned_arguments_kwarg[argument_name] = self._y
                continue
            if argument_value == "original_fh":
                returned_arguments_kwarg[argument_name] = self._fh
                continue

            if (argument_value in self._y_transformers) and method == "fit":
                # used for invese transforming `y` at predict()
                # if a transformer transforms y
                # its argument_value can be either `original_y`
                # or a name of step referring to a transformer of y
                # the first case was covered above
                # if self._y_transformers is of length 1
                # this means that no transformations were applied to `y`
                # this check is handled in predict
                self._y_transformers.append(step_name)
            if type(argument_value) == list:
                out = []
                for list_val in argument_value:
                    returned_step_value = self._check_steps_for_values(list_val, method)
                    if returned_step_value is not None:
                        out.append(returned_step_value)
                returned_arguments_kwarg[argument_name] = out
                continue
            # go through all steps and look for returned values
            returned_step_value = self._check_steps_for_values(argument_value, method)
            if returned_step_value is not None:
                returned_arguments_kwarg[argument_name] = returned_step_value

        return returned_arguments_kwarg

    def _fit(self, y, X=None, fh=None):
        self.steps_ = self._check_steps_for_consistency()
        self._set_y_X(y, X)
        self._set_fh(fh)
        for name, est, arguments in self._iter(method="fit"):
            # check arguments of pipeline.
            # Inspect the `fit` key of the step arguments
            # check and process corresponding values
            processed_arguments = {}

            if arguments["fit"] is None:
                # this step must be skipped
                continue
            processed_arguments = self._process_arguments(
                step_name=name, arguments=arguments, method="fit"
            )

            if "X" in processed_arguments and processed_arguments["X"] is not None:
                if type(processed_arguments["X"]) is list:
                    args = []
                    for arg in processed_arguments["X"]:
                        # args.append(check_X(arg))
                        args.append(arg)
                    processed_arguments["X"] = args
                # else:
                #     processed_arguments["X"] = check_X(processed_arguments["X"])
            if "X" in processed_arguments and processed_arguments["X"] is None:
                del processed_arguments["X"]
            # if estimator has `fit_transform()` method it is a transformer
            if hasattr(est, "fit_transform"):
                t = clone(est)
                out = t.fit_transform(**processed_arguments)
                self._step_results_fit[name] = out
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
            if arguments["predict"] is None:
                # this step must be skipped
                continue

            # y is not used when the steps are executed in normal order.
            if "y" in arguments["predict"]:
                arguments["predict"].pop("y", None)

            processed_arguments = self._process_arguments(
                step_name=name, arguments=arguments, method="predict"
            )

            # if estimator has a `transform()` method, it is a transformer
            if hasattr(est, "transform"):
                if "y" in arguments["predict"]:
                    # process only exogenous variables, skip `y`
                    arguments["predict"].pop("y")
                self._step_results_predict[name] = est.transform(**processed_arguments)
            # if estimator has a `predict()` method, it is a forecaster

            if hasattr(est, "predict"):

                fitted_estimator = self._fitted_estimators[name]
                pred = fitted_estimator.predict(
                    **processed_arguments, return_pred_int=return_pred_int, alpha=alpha
                )
                # pred.index.freq = self._y.index.freq
                self._step_results_predict[name] = pred

        # iterate in reverse order only transormers of y
        if len(self._y_transformers) == 1:
            # no transformations of y were performed
            return pred
        else:
            # ignore the last step.
            # last step is final estimator
            last_step_output = pred
            for y_transformer in reversed(self._y_transformers[0:-1]):
                trained_estimator = self._fitted_estimators[y_transformer]
                if hasattr(trained_estimator, "inverse_transform"):
                    # overwrite step results with inverse transform
                    last_step_output = trained_estimator.inverse_transform(
                        last_step_output
                    )
            return last_step_output

    def _update(self, y, X=None, update_params=True):
        """Update fitted parameters.

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

            if arguments["update"] is None:
                # this step must be skipped
                continue
            # assume update takes same arguments as fit

            processed_arguments = self._process_arguments(
                step_name=name, arguments=arguments, method="update"
            )

            processed_arguments["update_params"] = update_params
            # Transformers do not have `predict()`

            if hasattr(est, "update"):

                out = est.update(**processed_arguments)
                out._cutoff = self.cutoff
                self._step_results_update[name] = out
            else:
                raise ValueError(f"{name} has no update() method.")
            #     continue
            # forecasters have `update()` and `predict()` methods
            # if hasattr(est, "update") and hasattr(est, "predict"):
            #     est.update(**processed_arguments)

        self._is_fitted = True

        return self

    def get_fitted_params(self):
        """Will be implemented."""
        raise NotImplementedError()
