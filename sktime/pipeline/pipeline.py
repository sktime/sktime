from sktime.base import BaseEstimator


class Pipeline(BaseEstimator):

    def __init__(self, steps=None):
        super().__init__()
        # Initialise the method
        self.steps = [] if steps is None else steps

    @staticmethod
    def _check_validity(step, method_name, **kwargs):
        # Checks if the method_name is allowed to call on the pipeline.
        # Thus, it uses ducktyping
        # Returns the all kwargs that are provided to the pipeline and needed by method_name.
        pass

    def add_step(self, skobject, name, edges, **kwargs):
        # Adds a step to the pipeline and store the step informations
        pass

    def fit(self, **kwargs):
        # Fits the pipeline
        pass

    def transform(self, *args, **kwargs):
        # Implementation of transform, such methods also are required for predict, ...
        pass

    def predict(self, *args, **kwargs):
        # Implementation of transform, such methods also are required for predict, ...
        pass

    def predict_quantiles(self, *args, **kwargs):
        # Implementation of transform, such methods also are required for predict, ...
        pass

    def predict_proba(self, *args, **kwargs):
        # Implementation of transform, such methods also are required for predict, ...
        pass

    # TODO Weitere Methoden
    ...
