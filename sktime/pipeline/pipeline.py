# -*- coding: utf-8 -*-
from sktime.base import BaseEstimator


from sktime.transformations.base import BaseTransformer


class OnlineUnsupervisedPipeline(BaseEstimator):
    """
    Parameters
    ----------
    steps : array of lists
        list comprised of three elements:
            1. name of step (string),
            2. algorithm (object),
            3. input (dictionary) key value paris for fit() method of algorithm
    interface: sting
        `simple`(default) or `advanced`.
            If `simple` `fit` method of the object is called.
            If `advanced` name of method can be specified like this:
            (`name of step`, `algorithm`, `input`), where input:
        input_dict = [
            {
                function: name of function to be called (fit, transform, predict ...),
                arguments: key value pairs(value is name of step) ,
            },
            {
                function: name of function to be called,
                arguments: key value pairs(value is name of step),
            },
            ......
        ]

    """

    def __init__(self, steps, interface="simple"):
        self._steps = steps
        self._results_dict = {}
        self._interface = interface

    def _iter(self):
        for name, alg, arguments in self._steps:
            yield name, alg, arguments

    def _check_arguments(self, arguments):
        """
        Checks arguments for consistency.

        Replace key word 'original' with X

        Replace string step name with step fit return result

        Parameters
        ----------
        arguments : dictionary
            key-value for fit() method of algorithm
        """
        if arguments is None:
            return arguments
        for key, value in arguments.items():
            if value == "original":
                arguments[key] = self._X

            if value in self._results_dict:
                arguments[key] = self._results_dict[value]._fit_result

        return arguments

    def fit(self, X):
        self._X = X
        for name, alg, arguments in self._iter():
            if self._interface == "simple":
                arguments = self._check_arguments(arguments)
                # Transformers are instances of BaseTransformer and BaseEstimator
                # Estimators are only instances of BaseEstimator
                if isinstance(alg, BaseTransformer) and isinstance(alg, BaseEstimator):
                    self._results_dict[name] = alg.transform(**arguments)
                if not isinstance(alg, BaseTransformer) and isinstance(
                    alg, BaseEstimator
                ):
                    self._results_dict[name] = alg.fit(**arguments)
            if self._interface == "advanced":
                for arg in arguments:
                    func_name = arg["function"]
                    arguments = self._check_arguments(arg["arguments"])
                    # execute function
                    if arguments is not None:
                        self._results_dict[name] = getattr(alg, func_name)(**arguments)
                    else:
                        self._results_dict[name] = getattr(alg, func_name)()

        return self
