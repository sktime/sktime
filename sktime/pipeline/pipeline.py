"""class that implements a graph pipeline."""
import weakref

from sktime.base import BaseEstimator
from sktime.pipeline.step import Step
from sktime.transformations.series.subset import ColumnSelect


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
    other steps and not only one sucessors.

    `fit(y, X, *args)` - changes state by running `fit` on all sktime estimators and
        transformers in the pipeline. Note that depending on the sktime estimators and
        transformers that are added to the pipeline, different keywords are required.
        E.g., if a forecaster is part of the pipeline, a forecast horizon (fh) should be
        provided.
    `predict(X, *args)` - Results in calling predict on the estimators in the pipeline
        and transform or the specified method on the other skobjects in the pipeline.
        Depending on the skobject added to the pipeline, you might need to pass
        additional parameters to predict.
    `predict_interval(X, fh)`, `predict_quantiles(X, fh)` - as `predict(X, fh)`,
        with `predict_interval` or `predict_quantiles` substituted for `predict`.
    `predict_var`, `predict_proba` - are currently not supported

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `add_step(skobject, name, edges, method, **kwargs)` - adds a skobject to the pipeline and
        setting the name as identifier and the steps specified with edges as input steps
        (predecessors). Therby the method that should be called can be overriden using the method kwarg.
        Further provided kwargs are directly provided to the skobject if it is called.

    Parameters
    ----------
    param step_informations : what it is, what it does

    Attributes
    ----------
    attribute id_to_true_id : a dict with integer keys and values,
        mapping the python object id to skobject ids.
    attribute id_to_obj : a dict with integer keys and weak references of
        skobjects as values. The values are the weak references of the skobjects
        provided to the `add_step` method. We store the weak references to
        avoid that the id of the object is reassigned if the user deletes all it
        references to the object.
    attribute model_dict : a dict with integer keys and skobject values.
        This is a mapping of the id of the skobjects provided to `add_step`
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
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> from sktime.forecasting.sarimax import SARIMAX
    >>> from sktime.pipeline.pipeline import Pipeline
    >>> from sktime.transformations.compose import Id
    >>> from sktime.transformations.series.boxcox import BoxCoxTransformer
    >>> from sktime.transformations.series.exponent import ExponentTransformer

        Example 1: Simple sequential pipeline of transformers using the generalized
        non-sequential pipeline implementation
    >>>     y, X = load_longley()
    >>>     general_pipeline = Pipeline()
    >>>     for step in [
    ...         {"skobject": ExponentTransformer(), "name": "exp", "edges": {"X": "X"}},
    ...         {"skobject": BoxCoxTransformer(), "name": "box", "edges": {"X": "exp"}},
    ...         ]:
    >>>         general_pipeline.add_step(**step)
    >>>     general_pipeline.fit(X=X)
    >>>     result_general = general_pipeline.transform(X)

        Example 2: Classification sequential pipeline using the generalized
         non-sequential pipeline implementation
    >>>     X, y = load_arrow_head(split="train", return_X_y=True)
    >>>     general_pipeline = Pipeline()
    >>>     for step in [
    ...         {"skobject": ExponentTransformer(), "name": "exp", "edges": {"X": "X"}},
    ...         {"skobject": KNeighborsTimeSeriesClassifier(),
    ...          "name": "knnclassifier",
    ...          "edges": {"X": "exp", "y": "y"}}]:
    >>>         general_pipeline.add_step(**step)
    >>>     general_pipeline.fit(X=X, y=y)
    >>>     result_general = general_pipeline.predict(X)
        Example 3: Forecasting pipeline with exogenous features using the
        generalized non-sequential pipeline implementation

    >>>     y, X = load_longley()
    >>>     y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    >>>     general_pipeline = Pipeline()
    >>>     for step in [
    ...         {"skobject": ExponentTransformer(), "name": "exp", "edges": {"X": "X"}},
    ...         {"skobject": SARIMAX(),
    ...          "name": "SARIMAX",
    ...          "edges": {"X": "exp", "y": "y"}}]:
    >>>         general_pipeline.add_step(**step)
    >>>     general_pipeline.fit(y=y_train, X=X_train, fh=[1, 2, 3, 4])
    >>>     result_general = general_pipeline.predict(X=X_test)
    """

    def __init__(self, step_informations=None):
        super().__init__()

        self.id_to_true_id = {}
        self.id_to_obj = {}
        self.counter = 0
        self.steps = {
            "X": Step(None, "X", None, None, {}),
            "y": Step(None, "y", None, None, {}),
        }
        self.model_dict = {}
        self.kwargs = {}
        self.step_informations = [] if step_informations is None else step_informations
        if step_informations is not None:
            for step_info in step_informations:
                self.add_step(**step_info)

    def _get_unique_id(self, skobject):
        self.counter += 1
        # Check if not already an skobject cloned from the provided
        # skobject is part of the pipeline
        if (id(skobject) not in self.id_to_obj) or self.id_to_obj[
            id(skobject)
        ]() is None:
            # In this case set a weakref of that skobject to id_to_obj to prevent that
            # the garbage collector reassigns the id.
            self.id_to_obj[id(skobject)] = weakref.ref(skobject)
            self.id_to_true_id[id(skobject)] = self.counter
        return self.id_to_true_id[id(skobject)]

    def _get_step(self, name):
        if name in self.steps:
            return self.steps[name]
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
        skobject: `sktime` object, the skobject that should be added to the pipeline
        name: str, the name of the step that is created
        edges: dict, a dict with string keys to string values. Identifying the
            predcessors.  The keys of the edges dict specify to which argument
             of fit/predict/.. the output of the predessors (the value of the
             dict specifies the predessors name) shuold be passed.
        method: str, an optional argument allowing to determine the method that
            should be executed when the pipeline calls the provided skobject.
            If not specified, the pipeline selects the method based on the method
            that is called on the pipeline (e.g., predict, transform, ..)
        kwargs: additional kwargs are parameters that are provided to the
            skobject if fit/predict/.. is called.

        """
        unique_id = self._get_unique_id(skobject)
        if unique_id not in self.model_dict:
            self.model_dict[unique_id] = skobject.clone()
        cloned_skobject = self.model_dict[unique_id]

        input_steps = {}
        for key, edge in edges.items():
            edge = edge if isinstance(edge, list) else [edge]
            for edg in edge:
                if "__" in edg and edg not in self.steps:
                    # Just semantic sugar..
                    self._create_subsetter(edg)
                input_steps[key] = [self._get_step(edg) for edg in edge]

        step = Step(cloned_skobject, name, input_steps, method=method, params=kwargs)
        if name in self.steps:
            raise ValueError(
                f"You try to add a step with a name '{name}' to the pipeline"
                f" that already exists. Try to use an other name."
            )

        self.steps[name] = step
        self._last_step_name = name
        new_step_info = {key: value for key, value in kwargs.items()}
        new_step_info.update(
            {
                "skobject": skobject,
                "name": name,
                "edges": edges,
            }
        )
        self.step_informations.append(new_step_info)
        return step

    def fit(self, X, y=None, **kwargs):
        """Fit graph pipeline to training data.

        Parameters
        ----------
        X : TODO What types are really allowed here?
        y : TODO what types are really allowed?

        Notes
        -----
        Changes state by creating a fitted model that updates attributes
        ending in "_" in the skobjects of the pipeline and sets is_fitted flag to True.
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
        """Compute/return quantile forecasts.

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
        Return residuals of time series forecasts.

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
                    f"or `{method}`. Thus calling `{method}` on pipeline is not "
                    "allowed."
                )
        return True

    def _create_subsetter(self, edg):
        keys = edg.split("__")[-1].split("_")
        column_select = ColumnSelect(columns=keys)
        self.add_step(column_select, edg, {"X": edg.split("__")[0]})
