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
        self._step_results = {}

    def _iter(self):
        for name, alg, arguments in self._steps:
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
            if argument_value == "original":
                returned_arguments_kwarg[argument_name] = self._X
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

    def fit(self, X):
        self._X = X
        for name, alg, arguments in self._iter():

            arguments = self._process_arguments(arguments["fit"])
            # Transformers are instances of BaseTransformer and BaseEstimator
            # Estimators are only instances of BaseEstimator
            if isinstance(alg, BaseTransformer) and isinstance(alg, BaseEstimator):
                out = alg.fit_transform(**arguments)
                self._step_results[name] = out
            if not isinstance(alg, BaseTransformer) and isinstance(alg, BaseEstimator):
                alg.fit(**arguments)

        return self

    def predict(self, X):

        self._X = X

        for _, alg, arguments in self._iter():
            arguments = self._process_arguments(arguments["predict"])
            if isinstance(alg, BaseTransformer) and isinstance(alg, BaseEstimator):
                return alg.transform(**arguments)
            if not isinstance(alg, BaseTransformer) and isinstance(alg, BaseEstimator):
                return alg.predict(**arguments)
