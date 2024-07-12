"""Implementation of the graphpipeline step."""

import inspect
from copy import deepcopy

import pandas as pd

from sktime.forecasting.base import BaseForecaster

ALLOWED_METHODS = [
    "transform",
    "inverse_transform",
    "predict",
    "predict_quantiles",
    "predict_interval",
    "predict_residuals",
]


class StepResult:
    """
    Result of a step.

    Parameters
    ----------
    result : mtype: `np.ndarray` or `pd.Series` or `pd.DataFrame`
        The result of the step.
    mode : str
        The mode of the result. Is it a probabilistic result or not.
    """

    def __init__(self, result, mode):
        self.result = result
        self.mode = mode


class Step:
    """
    A step in the pipeline.

    Parameters
    ----------
    skobject : BaseObject
        The sktime object that is used in the step.
    name : str
        The name of the step.
    input_edges : dict
        A dict with string keys to string values. Identifying the
        predcessors.  The keys of the edges dict specify to which argument
        of fit/predict/.. the output of the predecessors (the value of the
        dict specifies the predecessors name) should be passed.
    method : str
        The method that should be called on the skobject. If None, the pipeline
         selects the method based on the method
         that is called on the pipeline (e.g., predict, transform, ..)
    params : dict
        The parameters that should be passed to the skobject when calling
        method
    """

    def __init__(
        self,
        skobject,
        name,
        input_edges,
        method,
        params,
    ):
        self.buffer = None
        self.mode = ""
        self.skobject = skobject
        self.name = name
        self.method = method
        self.input_edges = input_edges
        self.params = params

    def reset(self, reset_buffer=True):
        """Reset the step."""
        if reset_buffer:
            self.buffer = None
        self.mode = ""

    def get_allowed_method(self):
        """
        Get the allowed method for the step.

        Returns all methods that are allowed to be called on the step.
        """
        if self.skobject is None:
            return ["transform"]
        return set(dir(self.skobject)).intersection(ALLOWED_METHODS)

    def get_result(self, fit, required_method, mro, kwargs):
        """
        Get the results of the pipeline step.

        Parameters
        ----------
        fit : bool
            Whether the pipeline is in fit mode.
        required_method : str
            The method that should be called on the step.
        mro : list
            The method resolution order of the step. I.e., first try to call
            the first method, if not possible second method, etc.
        kwargs : dict
            The kwargs that should be passed to the method.
        """
        if self.buffer is not None or self.input_edges is None:
            return StepResult(self.buffer, mode=self.mode)

        # 1. Get results from all previous steps!
        input_data, self.mode, all_none = self._fetch_input_data(
            fit, required_method, mro, kwargs
        )

        # 2. Get the method that should be called on skobject
        if self.method is not None:
            mro = [self.method]
        if hasattr(self.skobject, "fit") and fit and not self.skobject.is_fitted:
            kwargs_ = self._extract_kwargs("fit", kwargs)
            self.skobject.fit(**input_data, **kwargs_)

            # if the skobject is a forecaster, passby the input
            # data if is fitted in the same call
            if isinstance(self.skobject, BaseForecaster):
                self._store_to_buffer(input_data["y"])
                return StepResult(input_data["y"], mode=self.mode)

        for method in mro:
            if hasattr(self.skobject, method):
                non_default_parameter = any(
                    map(
                        lambda x: x.default == inspect.Parameter.empty,
                        inspect.signature(
                            getattr(self.skobject, method)
                        ).parameters.values(),
                    )
                )
                if all_none and non_default_parameter:
                    # Skip method if all input data is None and the method
                    # requires parameters.
                    continue
                kwargs_ = self._extract_kwargs(method, kwargs)
                if "fh" in kwargs_ and fit:
                    # Perform in sample prediction if the get_result is called
                    # during fitting the pipeline. In the case a provided,
                    # fh should be replaced with the length of the time series.
                    kwargs_["fh"] = (
                        input_data["y"].index
                        if hasattr(input_data["y"], "index")
                        else range(len(input_data["y"]))
                    )
                # 3. Call method on skobject and return result
                if self.mode == "proba":
                    idx = input_data["X"].columns
                    n = idx.nlevels
                    yt = dict()
                    for ix in idx:
                        levels = list(range(1, n))
                        if len(levels) == 1:
                            levels = levels[0]
                        yt[ix] = input_data["X"][ix]
                        yt[ix] = getattr(self.skobject, method)(
                            X=yt[ix], **kwargs_
                        ).to_frame()
                    result = pd.concat(yt.values(), axis=1)
                else:
                    result = getattr(self.skobject, method)(
                        **dict(
                            filter(
                                lambda k: k[0]
                                in inspect.getfullargspec(
                                    getattr(self.skobject, method)
                                ).args,
                                input_data.items(),
                            )
                        ),
                        **kwargs_,
                    )

                self.mode = (
                    "proba"
                    if (
                        method
                        in [
                            "predict_interval",
                            "predict_quantiles",
                            "predict_proba",
                        ]
                    )
                    or (self.mode == "proba")
                    else ""
                )
                self._store_to_buffer(result)
                return StepResult(result, self.mode)

        return StepResult(None, "")

    def _fetch_input_data(self, fit, required_method, mro, kwargs):
        input_data = {}
        all_none = True
        transformer_names = []

        for step_name, steps in self.input_edges.items():
            results = []
            for step in steps:
                transformer_names.append(step.name)
                result = step.get_result(
                    fit=fit, required_method=required_method, mro=mro, kwargs=kwargs
                )
                results.append(result.result)
                if result.mode != "":
                    self.mode = result.mode
                if result.result is not None:
                    all_none = False
            if len(results) != 0 and results[0] is not None:
                if len(results) > 1:
                    input_data[step_name] = pd.concat(
                        results, axis=1, keys=transformer_names
                    )
                else:
                    input_data[step_name] = results[0]
        return input_data, self.mode, all_none

    def _extract_kwargs(self, method_name, kwargs):
        kwargs_ = deepcopy(kwargs)
        kwargs_.update(self.params)
        use_kwargs = {}
        method = getattr(self.skobject, method_name)
        method_signature = inspect.signature(method).parameters

        for name, _param in method_signature.items():
            if name in kwargs:
                use_kwargs[name] = kwargs[name]
        return use_kwargs

    def _store_to_buffer(self, result):
        self.buffer = result
