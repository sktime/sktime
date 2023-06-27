# -*- coding: utf-8 -*-
"""class that implements a graph pipeline."""
import weakref

from sktime.base import BaseEstimator
from sktime.pipeline.step import Step
from sktime.transformations.series.subset import ColumnSelect


class MethodNotImplementedError(Exception):
    """Exception class to raise if the required method is not supported by the current pipeline instance
    since there is at least one skobject in the pipeline that do not support transform and the required method
    """

    def __init__(self, message):
        super().__init__(message)


class Pipeline(BaseEstimator):
    """This class is a generalized graph pipeline. Generalized means that it can contain
    forecasters, classifiers, etc. The graph pipeline mean that the structure is not linear.
    I.e., the each element of the pipeline can be the input of multiple other steps and not only one
    sucessors.

    Describe methods!
    `fit(y, X, *args)` - changes state by running `fit` on all sktime estimators and transformers
        in the pipeline. Note that depending on the sktime estimators and transformers that are added
        to the pipeline, different keywords are required. E.g., if a forecaster is part of the pipeline,
        a forecast horizon (fh) should be provided.
    `predict(X, *args)` - Results in calling predict on the estimators in the pipeline and transform
        or the specified method on the other skobjects in the pipeline. Depending on the skobject added
        to the pipeline, you might need to pass additional parameters to predict.
    `predict_interval(X, fh)`, `predict_quantiles(X, fh)` - as `predict(X, fh)`,
        with `predict_interval` or `predict_quantiles` substituted for `predict`.
    `predict_var`, `predict_proba` - are currently not supported

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `add_step(skobject, name, edges, **kwargs)` - adds a skobject to the pipeline and setting the name as
        identifier and the steps specified with edges as input. # TODO Finalize this description

    Parameters
    ----------
    # TODO
    param name : what it is, what it does

    Attributes
    ----------
    # TODO
    attribute name : what it is, what it does

    Examples
    --------
    # TODO examples: Classifier, Forecaster, ForecasterX?
    >>> Do all import

        Example 1: string/estimator pairs
    >>> Different examples
    """

    def __init__(self, step_informations=None):
        super().__init__()

        self.id_to_true_id = {}
        self.id_to_obj = {}
        self.counter = 0
        self.steps = {
            "X": Step(None, "X", None, {}),
            "y": Step(None, "y", None, {}),
        }
        self.model_dict = {}
        self.kwargs = {}
        if step_informations is not None:
            for step_info in step_informations:
                self.add_step(**step_info)

    def _get_unique_id(self, skobject):
        self.counter += 1
        # Check if not already an skobject cloned from the provided skobject is part of the pipeline
        if (id(skobject) not in self.id_to_obj) or self.id_to_obj[
            id(skobject)
        ]() is None:
            # In this case set a weakref of that skobject to id_to_obj to prevent that the garbage collector
            # reassigns the id.
            self.id_to_obj[id(skobject)] = weakref.ref(skobject)
            self.id_to_true_id[id(skobject)] = self.counter
        return self.id_to_true_id[id(skobject)]

    def _get_step(self, name):
        if name in self.steps:
            return self.steps[name]
        raise Exception("Required Input does not exist")

    def add_step(self, skobject, name, edges, **kwargs):
        """
        TODO
        """
        # TODO revise kwargs. E.g. method should be an explicit parameter.
        unique_id = self._get_unique_id(skobject)
        if unique_id not in self.model_dict:
            self.model_dict[unique_id] = skobject.clone()
        skobject = self.model_dict[unique_id]

        input_steps = {}
        for key, edge in edges.items():
            edge = edge if isinstance(edge, list) else [edge]
            for edg in edge:
                if "__" in edg and edg not in self.steps:
                    # Just semantic sugar..
                    self._create_subsetter(edg)
                input_steps[key] = [self._get_step(edg) for edg in edge]

        step = Step(
            skobject,
            name,
            input_steps,
            kwargs,
        )
        if name in self.steps:
            raise Exception("Name Conflict")

        self.steps[name] = step
        self._last_step_name = name
        return step

    def fit(self, X, y=None, **kwargs):
        """
        TODO
        """
        # Fits the pipeline
        self.kwargs = kwargs
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y

        # 4. call get_result or something similar on last step!
        self.steps[self._last_step_name].get_result(
            fit=True,
            required_method=None,
            mro=["transform", "predict"],
            kwargs=self.kwargs,
        )
        return self

    def fit_transform(self, X, y=None, **kwargs):
        """
        TODO
        """
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)

    def transform(self, X, y=None, **kwargs):
        """
        TODO
        """
        # Implementation of transform, such methods also are required for predict, ...
        # 1. Check if transform is allowed. I.e., Check method needs to check if
        #    all steps implement transform + If all required params are passed
        self._method_allowed("transform")

        # 3. set data into start steps buffer!
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y
        self.kwargs.update(kwargs)

        # 4. call get_result or something similar on last step!
        return (
            self.steps[self._last_step_name]
            .get_result(
                fit=False,
                required_method="transform",
                mro=["transform"],
                kwargs=self.kwargs,
            )
            .result
        )

    def predict(self, X, y=None, **kwargs):
        """
        TODO
        """
        # Implementation of transform, such methods also are required for predict, ...
        # 1. Check if transform is allowed. I.e., Check method needs to check if all
        #    steps implement transform or predict + If all required params are passed
        # 2. Set predict/transform as global methods
        self._method_allowed("predict")

        # 3. set data into start steps buffer!
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y
        self.kwargs.update(kwargs)

        # 4. call get_result or something similar on last step!
        return (
            self.steps[self._last_step_name]
            .get_result(
                fit=False,
                required_method="predict",
                mro=["predict", "transform"],
                kwargs=self.kwargs,
            )
            .result
        )

    def predict_interval(self, X, y=None, **kwargs):
        """
        TODO
        """
        self._method_allowed("predict_interval")

        # 2. Set transform as global method as well as provide all kwargs to step

        # 3. set data into start steps buffer!
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y
        self.kwargs.update(kwargs)

        # 4. call get_result or something similar on last step!
        return (
            self.steps[self._last_step_name]
            .get_result(
                fit=False,
                required_method="predict_interval",
                mro=["predict_interval", "predict", "transform"],
                kwargs=self.kwargs,
            )
            .result
        )

    def predict_quantiles(self, X, y=None, **kwargs):
        """
        TODO
        """
        self._method_allowed("predict_quantiles")

        # 2. Set transform as global method as well as provide all kwargs to step

        # 3. set data into start steps buffer!
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y
        self.kwargs.update(kwargs)

        # 4. call get_result or something similar on last step!
        return (
            self.steps[self._last_step_name]
            .get_result(
                fit=False,
                required_method="predict_quantiles",
                mro=["predict_quantiles", "predict", "transform"],
                kwargs=self.kwargs,
            )
            .result
        )

    def predict_residuals(self, X, y=None, **kwargs):
        """
        TODO
        """
        self._method_allowed("predict_residuals")

        # 2. Set transform as global method as well as provide all kwargs to step

        # 3. set data into start steps buffer!
        # TODO get rid of this boilerplate
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y
        self.kwargs.update(kwargs)

        # 4. call get_result or something similar on last step!
        return (
            self.steps[self._last_step_name]
            .get_result(
                fit=False,
                required_method="predict_residuals",
                mro=["predict_residuals", "predict", "transform"],
                kwargs=self.kwargs,
            )
            .result
        )

    def _method_allowed(self, method):
        for _step_name, step in self.steps.items():
            if "transform" in step.get_allowed_method():
                pass  # This would be okay
            elif method in step.get_allowed_method():
                pass  # This would be okay
            else:
                raise MethodNotImplementedError(
                    f"Step {_step_name} does not support the methods: `transform` "
                    f"or `{method}`. Thus calling `{method}` on pipeline is not allowed."
                )
        return True

    def _create_subsetter(self, edg):
        keys = edg.split("__")[-1].split("_")
        column_select = ColumnSelect(columns=keys)
        self.add_step(column_select, edg, {"X": edg.split("__")[0]})
