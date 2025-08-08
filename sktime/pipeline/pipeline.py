"""class that implements a graph pipeline."""

import warnings
from copy import copy, deepcopy

from sktime.base import BaseEstimator
from sktime.pipeline.step import Step
from sktime.transformations.series.subset import ColumnSelect

__author__ = ["benHeid"]
__all__ = ["Pipeline"]


class MethodNotImplementedError(Exception):
    """MethodNotImplementedError.

    Exception class to raise if the required method is not supported
    by the current pipeline instance since there is at least one skobject
    in the pipeline that do not support transform and the required method.
    """

    def __init__(self, message):
        super().__init__(message)


class Pipeline(BaseEstimator):
    """Implementation of a Graphpipeline.

    This class is a generalized graph pipeline. Generalized means that it can
    contain forecasters, classifiers, etc. The graph pipeline mean that the structure
    is not linear. I.e., the each element of the pipeline can be the input of multiple
    other steps and not only one successors.

    ``fit(y, X, *args)`` - changes state by running ``fit`` on all sktime estimators and
        transformers in the pipeline. Note that depending on the sktime estimators and
        transformers that are added to the pipeline, different keywords are required.
        E.g., if a forecaster is part of the pipeline, a forecast horizon (fh) should be
        provided.
    ``predict(X, *args)`` - Results in calling predict on the estimators in the pipeline
        and transform or the specified method on the other skobjects in the pipeline.
        Depending on the skobject added to the pipeline, you might need to pass
        additional parameters to predict.
    ``predict_interval(X, fh)``, ``predict_quantiles(X, fh)`` - as ``predict(X, fh)``,
        with ``predict_interval`` or ``predict_quantiles`` substituted for ``predict``.
    ``predict_var``, ``predict_proba`` - are currently not supported
    ``get_params``, ``set_params`` uses ``sklearn`` compatible nesting interface
    ``add_step(skobject, name, edges, method, **kwargs)`` - adds a skobject to the
        pipeline and setting the name as identifier and the steps specified with
        edges as input steps (predecessors). Thereby the method that should be
        called can be overridden using the method kwarg. Further provided kwargs
        are directly provided to the skobject if it is called.

    Parameters
    ----------
    param steps : A list of dicts that specify the steps of the pipeline. Further
        steps can be added by using add_step method.
        The dict requires the following keys:
            * skobject: ``sktime`` object, the skobject that should be added to the
                 pipeline
            * name: str, the name of the step that is created
            * edges: dict, a dict with string keys to string values. Identifying the
                 predcessors.  The keys of the edges dict specify to which argument
                 of fit/predict/.. the output of the predecessors (the value of the
                 dict specifies the predecessors name) should be passed.
            * method: str, an optional argument allowing to determine the method that
                should be executed when the pipeline calls the provided skobject.
                If not specified, the pipeline selects the method based on the method
                that is called on the pipeline (e.g., predict, transform, ..)
            * kwargs: additional kwargs are parameters that are provided to the
                skobject if fit/predict/.. is called.

    Attributes
    ----------
    attribute id_to_true_id : a dict with integer keys and values,
        mapping the python object id to skobject ids.
    attribute id_to_obj : a dict with integer keys and weak references of
        skobjects as values. The values are the weak references of the skobjects
        provided to the ``add_step`` method. We store the weak references to
        avoid that the id of the object is reassigned if the user deletes all it
        references to the object.
    attribute model_dict : a dict with integer keys and skobject values.
        This is a mapping of the id of the skobjects provided to ``add_step``
        to the cloned skobject.
    attribute counter : integer, counts the number of steps in the pipeline.
    attribute steps : a dict with string keys and step object values.
        The key is the name that is specified if a skobject is added to the pipeline.
    attribute kwargs : a dict with str keys and object values. Stores all kwargs
        that are specified and might be passed to skobjects in the pipeline.

    Examples
    --------
    >>> from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
    >>> from sktime.datasets import load_arrow_head, load_longley
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.pipeline import Pipeline
    >>> from sktime.transformations.compose import Id
    >>> from sktime.transformations.series.boxcox import BoxCoxTransformer
    >>> from sktime.transformations.series.exponent import ExponentTransformer

        Example 1: Simple sequential pipeline of transformers using the generalized
        non-sequential pipeline implementation
    >>> y, X = load_longley()
    >>> general_pipeline = Pipeline()
    >>> for step in [
    ...     {"skobject": ExponentTransformer(), "name": "exp", "edges": {"X": "X"}},
    ...     {"skobject": BoxCoxTransformer(), "name": "box", "edges": {"X": "exp"}},
    ...     ]:
    ...     general_pipeline = general_pipeline.add_step(**step)
    >>> general_pipeline.fit(X=X) # doctest: +SKIP
    >>> result_general = general_pipeline.transform(X) # doctest: +SKIP

        Example 2: Classification sequential pipeline using the generalized
         non-sequential pipeline implementation
    >>> X, y = load_arrow_head(split="train", return_X_y=True)
    >>> general_pipeline = Pipeline()
    >>> for step in [
    ...     {"skobject": ExponentTransformer(), "name": "exp", "edges": {"X": "X"}},
    ...     {"skobject": KNeighborsTimeSeriesClassifier(),
    ...      "name": "knnclassifier",
    ...      "edges": {"X": "exp", "y": "y"}}]:
    ...     general_pipeline = general_pipeline.add_step(**step)
    >>> general_pipeline.fit(X=X, y=y) # doctest: +SKIP
    >>> result_general = general_pipeline.predict(X) # doctest: +SKIP

        Example 3: Forecasting pipeline with exogenous features using the
        generalized non-sequential pipeline implementation
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    >>> general_pipeline = Pipeline()
    >>> for step in [
    ...     {"skobject": ExponentTransformer(), "name": "exp", "edges": {"X": "X"}},
    ...     {"skobject": NaiveForecaster(),
    ...      "name": "SARIMAX",
    ...      "edges": {"X": "exp", "y": "y"}}]:
    ...     general_pipeline = general_pipeline.add_step(**step)
    >>> general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4]) # doctest: +SKIP
    >>> result_general = general_pipeline.predict(X=X_test) # doctest: +SKIP

    **Acknowledgements**
    This graphical pipeline is inspired by pyWATTS that is developed by the Institute
    for Automation and Applied Informatics (IAI) at Karlsruhe Institute of Technology.
    The implementation is supported by IAI and the author benHeid is funded by
    HelmholtzAI.
    Furthermore, we also want to credit @ViktorKaz for his independent pipeline design
    that is similar to this one.

    References
    ----------
    .. [1]  @article{heidrich2021pywatts,
      title={pyWATTS: Python workflow automation tool for time series},
      author={Heidrich, Benedikt and Bartschat, Andreas and Turowski, Marian and
              Neumann, Oliver and Phipps, Kaleb and Meisenbacher, Stefan and
              Schmieder, Kai and Ludwig, Nicole and Mikut, Ralf and Hagenmeyer, Veit},
      journal={arXiv preprint arXiv:2106.10157},
      year={2021}
    }
    """

    def __init__(self, steps=None):
        warnings.warn(
            "This generalised graphical pipeline is experimental, "
            "with all the usual risks of edge features. "
            "For mature alternatives, use single-purpose pipelines and compositors, "
            "such as TransformedTargetForecaster, ForecastingPipeline, "
            "ClassificationPipeline, etc., see for instance "
            "notebooks 01_forecasting.ipynb and "
            "02_classification.ipynb at "
            "https://github.com/sktime/sktime/blob/main/examples/.",
            stacklevel=1,
        )
        super().__init__()
        self._assembled = False
        self.id_to_true_id = {}
        self.id_to_obj = {}
        self.counter = 0
        self.assembled_steps = {
            "X": Step(None, "X", None, None, {}),
            "y": Step(None, "y", None, None, {}),
        }

        self.kwargs = {}
        self.steps = steps
        self._steps = steps if steps is not None else []

        object_types = [step["skobject"].get_tag("object_type") for step in self._steps]
        if len(set(object_types)) == 1:
            self.set_tags(**{"object_type": object_types[0]})
        elif len(set(object_types) - {"transformer"}) == 1:
            self.set_tags(
                **{"object_type": list(set(object_types) - {"transformer"})[0]}
            )
        else:
            # Mixture of different object types
            pass

        for step_information in self._steps:
            if "method" not in step_information:
                step_information["method"] = None
            if "kwargs" not in step_information:
                step_information["kwargs"] = {}
            self.clone_tags(step_information["skobject"])

    def _get_unique_id(self, skobject):
        self.counter += 1
        # Check if not already an skobject cloned from the provided
        # skobject is part of the pipeline
        if id(skobject) not in self.id_to_obj:
            # In this case store that skobject to id_to_obj to prevent that
            # the garbage collector reassigns the id.
            self.id_to_obj[id(skobject)] = skobject
            self.id_to_true_id[id(skobject)] = self.counter
        return self.id_to_true_id[id(skobject)]

    def get_params(self, deep=True):
        """Get the parameters of the pipeline.

        Parameters
        ----------
        deep : boolean, optional, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators

        Returns
        -------
        params : dict, parameter names mapped to their values.
        """
        params = {"steps": self.steps}
        if deep:
            for step_information in self._steps:
                for key, value in step_information["skobject"].get_params(deep).items():
                    params.update({step_information["name"] + "__" + key: value})
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with get_params().
        Note if steps is provided the other parameters are ignored.

        Parameters
        ----------
        params : dict, parameter names mapped to their values.

        Returns
        -------
        self : Pipeline, this estimator
        """
        self.kwargs = params
        new_step_infos = copy(self._steps)
        for key, value in params.items():
            keys = key.split("__")
            # keys length is at least three:
            # step_name, skobject_or_edges, parameter_name
            if len(keys) >= 3:
                step_name, skobject_or_edges = keys[:2]
                parameter_name = "__".join(keys[2:])

                # Only one element with the same name is allowed
                step_information = list(
                    filter(lambda x: x["name"] == step_name, new_step_infos)
                )[0]
                if skobject_or_edges == "skobject":
                    step_information["skobject"].set_params(**{parameter_name: value})
                elif skobject_or_edges == "edges":
                    assert parameter_name in [
                        "X",
                        "y",
                    ], "Only X and y are allowed as edges"
                    step_information["edges"][parameter_name] = value
                else:
                    raise Exception("Invalid parameter name")

        if "steps" in params:
            new_step_infos = params["steps"]
        self.__init__(steps=new_step_infos)
        return self

    def _get_step(self, name):
        if name in self.assembled_steps:
            return self.assembled_steps[name]
        raise Exception("Required Input does not exist")

    def add_step(self, skobject, name, edges, method=None, **kwargs):
        """Add a new skobject to the pipeline.

        This method changes the structure of the pipeline. It looks up if
        the cloned skobject already exists. If not it clones the skobject.
        Atfterwards, with the cloned skobject a new pipeline step is created,
        with the provided name. The input of this new step is specified by the
        edges dict.

        Parameters
        ----------
        skobject: ``sktime`` object, the skobject that should be added to the pipeline
        name: str, the name of the step that is created
        edges: dict, a dict with string keys to string values. Identifying the
            predcessors.  The keys of the edges dict specify to which argument
             of fit/predict/.. the output of the predecessors (the value of the
             dict specifies the predecessors name) should be passed.
        method: str, an optional argument allowing to determine the method that
            should be executed when the pipeline calls the provided skobject.
            If not specified, the pipeline selects the method based on the method
            that is called on the pipeline (e.g., predict, transform, ..)
        kwargs: additional kwargs are parameters that are provided to the
            skobject if fit/predict/.. is called.

        """
        step = {key: value for key, value in kwargs.items()}
        step.update(
            {
                "skobject": skobject,
                "name": name,
                "edges": edges,
                "method": method,
                "kwargs": kwargs,
            }
        )
        steps = copy(self._steps)
        steps.append(step)
        return Pipeline(steps=steps)

    def _assemble_steps(self):
        # Reset steps and id mappings
        self.assembled_steps = {
            "X": Step(None, "X", None, None, {}),
            "y": Step(None, "y", None, None, {}),
        }
        self.id_to_true_id = {}
        self.id_to_obj = {}
        self.counter = 0
        model_dict = {}

        for step_info in self._steps:
            skobject = step_info["skobject"]
            edges = step_info["edges"]
            name = step_info["name"]
            method = step_info["method"]
            kwargs = step_info["kwargs"]

            # Ensure that the pipelines contained cloned skobject and that
            # if a skobject is already cloned, that the cloned skobject is used
            unique_id = self._get_unique_id(skobject)
            if unique_id not in model_dict:
                model_dict[unique_id] = skobject.clone()
            cloned_skobject = model_dict[unique_id]

            input_steps = {}
            for key, edge in edges.items():
                edge = edge if isinstance(edge, list) else [edge]
                # Create subsetter for taking a specific column of the input.
                for edg in edge:
                    if "__" in edg and edg not in self.assembled_steps:
                        self._create_subsetter(edg)
                input_steps[key] = [self._get_step(edg) for edg in edge]

            step = Step(
                cloned_skobject, name, input_steps, method=method, params=kwargs
            )
            if name in self.assembled_steps:
                raise ValueError(
                    f"You try to add a step with a name '{name}' to the pipeline"
                    f" that already exists. Try to use an other name."
                )

            self.assembled_steps[name] = step
            self._last_step_name = name
        self._assembled = True

    def fit(self, X=None, y=None, **kwargs):
        """Fit graph pipeline to training data.

        Parameters
        ----------
        X : time series in sktime compatible format, optional (default=None)
                Exogeneous time series to fit to
        y : time series in sktime compatible data container format
                Time series to which to fit the pipeline.
        kwargs : additional key word arguments that might be passed to skobjects in the
            pipeline if they have a parameter that corresponds to a key of kwargs.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" in the skobjects of the pipeline and sets is_fitted flag to True.
        """
        self._assembled = False
        self._initiate_call(X, y, kwargs)

        assert (X is not None) or (y is not None), "Either X or y must be provided."
        self._y = y
        self._X = X

        self.assembled_steps[self._last_step_name].get_result(
            fit=True,
            required_method=None,
            mro=["transform", "predict"],
            kwargs=self.kwargs,
        )
        self._is_fitted = True
        self._set_attributes(X, y)
        return self

    def _set_attributes(self, X, y):
        self._X = X
        self._y = y
        for step in self.assembled_steps.values():
            if hasattr(step.skobject, "cutoff"):
                self.cutoff = step.skobject.cutoff

    def fit_transform(self, X, y=None, **kwargs):
        """Fit graph pipeline to training data and call transform afterward.

        Parameters
        ----------
        X : time series in sktime compatible format, optional (default=None)
                Exogeneous time series to fit to
        y : time series in sktime compatible data container format
                Time series to which to fit the pipeline.
        kwargs : additional key word arguments that might be passed to skobjects in the
            pipeline if they have a parameter that corresponds to a key of kwargs.

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" in the skobjects of the pipeline and sets is_fitted flag to True.

        Raises
        ------
        MethodNotImplementedError if a step in the pipeline does not implement
        ``transform``
        """
        return self.fit(X, y, **kwargs).transform(X, y, **kwargs)

    def transform(self, X, y=None, **kwargs):
        """Call transform on each element in the  graph pipeline.

        Parameters
        ----------
        X : time series in sktime compatible format, optional (default=None)
                Exogeneous time series to fit to
        y : time series in sktime compatible data container format
                Time series to which to fit the pipeline.
        kwargs : additional key word arguments that might be passed to skobjects in the
            pipeline if they have a parameter that corresponds to a key of kwargs.

        Raises
        ------
        MethodNotImplementedError if a step in the pipeline does not implement
         ``transform``
        """
        self._initiate_call(X, y, kwargs)
        self._method_allowed("transform")

        return (
            self.assembled_steps[self._last_step_name]
            .get_result(
                fit=False,
                required_method="transform",
                mro=["transform"],
                kwargs=self.kwargs,
            )
            .result
        )

    def predict(self, X=None, y=None, **kwargs):
        """Perform a prediction.

        I.e. calls predict or transform on each element in the  graph pipeline.

        Parameters
        ----------
        X : time series in sktime compatible format, optional (default=None)
                Exogeneous time series to fit to
        y : time series in sktime compatible data container format
                Time series to which to fit the pipeline.
        kwargs : additional key word arguments that might be passed to skobjects in the
            pipeline if they have a parameter that corresponds to a key of kwargs.

        Raises
        ------
        MethodNotImplementedError if a step in the pipeline does not implement
        ``transform`` or ``predict``
        """
        self._initiate_call(X, y, kwargs)
        self._method_allowed("predict")

        return (
            self.assembled_steps[self._last_step_name]
            .get_result(
                fit=False,
                required_method="predict",
                mro=["predict", "transform"],
                kwargs=self.kwargs,
            )
            .result
        )

    def predict_interval(self, X, y=None, **kwargs):
        """Perform an interval prediction.

        I.e. calls predict,  predict_interval, or transform  on each element
        in the  graph pipeline.

        Parameters
        ----------
        X : time series in sktime compatible format, optional (default=None)
                Exogeneous time series to fit to
        y : time series in sktime compatible data container format
                Time series to which to fit the pipeline.
        kwargs : additional key word arguments that might be passed to skobjects in the
            pipeline if they have a parameter that corresponds to a key of kwargs.

        Raises
        ------
        MethodNotImplementedError if a step in the pipeline does not implement
        ``transform``, ``predict``, or ``predict_interval``
        """
        self._initiate_call(X, y, kwargs)
        self._method_allowed("predict_interval")

        return (
            self.assembled_steps[self._last_step_name]
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
        Perform a quantile prediction.

        I.e. calls predict,  predict_quantiles, or transform  on each element
        in the  graph pipeline.

        Parameters
        ----------
        X : time series in sktime compatible format, optional (default=None)
                Exogeneous time series to fit to
        y : time series in sktime compatible data container format
                Time series to which to fit the pipeline.
        kwargs : additional key word arguments that might be passed to skobjects in the
            pipeline if they have a parameter that corresponds to a key of kwargs.

        Raises
        ------
        MethodNotImplementedError if a step in the pipeline does not implement
        ``transform``, ``predict``, or ``predict_quantiles``
        """
        self._initiate_call(X, y, kwargs)
        self._method_allowed("predict_quantiles")

        return (
            self.assembled_steps[self._last_step_name]
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
        Perform a residuals prediction.

        I.e. calls predict,  predict_residuals, or transform  on each element
        in the  graph pipeline.

        Parameters
        ----------
        X : time series in sktime compatible format, optional (default=None)
                Exogeneous time series to fit to
        y : time series in sktime compatible data container format
                Time series to which to fit the pipeline.
        kwargs : additional key word arguments that might be passed to skobjects in the
            pipeline if they have a parameter that corresponds to a key of kwargs.

        Raises
        ------
        MethodNotImplementedError if a step in the pipeline does not implement
         ``transform``,  ``predict``, or ``predict_residuals``
        """
        # If no y is passed, use the data passed to fit.
        inner_X = X if y is not None else self._X
        inner_y = y if y is not None else self._y
        if y is None:
            kwargs = deepcopy(kwargs)
            kwargs["fh"] = inner_X.index
        y_pred = self.predict(inner_X, inner_y, **kwargs)

        return inner_y - y_pred

    def _initiate_call(self, X, y, kwargs):
        if not self._assembled:
            self._assemble_steps()
        for key, step in self.assembled_steps.items():
            # Empty the buffer of all steps except for the dummy
            # steps X and y (input steps)
            if key in ["X", "y"]:
                step.reset(reset_buffer=False)
            else:
                step.reset()
        # Overwrite the buffer of X and y if data are provided
        if X is not None:
            self.assembled_steps["X"].buffer = X
        if y is not None:
            self.assembled_steps["y"].buffer = y
        self.kwargs.update(kwargs)

    def _method_allowed(self, method):
        for _step_name, step in self.assembled_steps.items():
            if "transform" in step.get_allowed_method():
                pass  # This would be okay
            elif method in step.get_allowed_method():
                pass  # This would be okay
            else:
                raise MethodNotImplementedError(
                    f"Step {_step_name} does not support the methods: `transform` "
                    f"or `{method}`. Thus calling `{method}` on pipeline is not "
                    "allowed."
                )
        return True

    def _create_subsetter(self, edg):
        keys = edg.split("__")[-1].split("_")
        column_select = ColumnSelect(columns=keys)
        self.add_step(column_select, edg, {"X": edg.split("__")[0]})
        step = Step(
            column_select,
            edg,
            {"X": [self.assembled_steps[edg.split("__")[0]]]},
            method="transform",
            params={},
        )
        self.assembled_steps[edg] = step

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.transformations.series.boxcox import BoxCoxTransformer
        from sktime.transformations.series.exponent import ExponentTransformer

        return [
            {
                "steps": [
                    {
                        "skobject": ExponentTransformer(),
                        "name": "exp",
                        "edges": {"X": "X"},
                    },
                    {
                        "skobject": BoxCoxTransformer(),
                        "name": "box",
                        "edges": {"X": "exp"},
                    },
                ]
            },
            {
                "steps": [
                    {
                        "skobject": ExponentTransformer(),
                        "name": "exp",
                        "edges": {"X": "X"},
                    },
                    {
                        "skobject": KNeighborsTimeSeriesClassifier(),
                        "name": "knnclassifier",
                        "edges": {"X": "exp", "y": "y"},
                    },
                ]
            },
            {
                "steps": [
                    {
                        "skobject": ExponentTransformer(),
                        "name": "exp",
                        "edges": {"X": "y"},
                    },
                    {
                        "skobject": NaiveForecaster(),
                        "name": "naive",
                        "edges": {"X": "exp", "y": "y"},
                    },
                ]
            },
        ]
