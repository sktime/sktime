from copy import deepcopy

from sktime.base import BaseEstimator
from sktime.pipeline.computation_setting import ComputationSetting
from sktime.pipeline.step import Step


class Pipeline(BaseEstimator):

    def __init__(self, step_informations=None):
        super().__init__()
        self.computation_setting = ComputationSetting()

        # Initialise the method
        self.steps = {"X": Step(None, "X", None, {}, compuatation_setting=self.computation_setting),
                      "y" : Step(None, "y", None, {}, compuatation_setting=self.computation_setting)}
        self.model_dict = {}
        if step_informations is not None:
            for step_info in step_informations:
                self.add_step(**step_info)

    @staticmethod
    def _check_validity(step, method_name, **kwargs):
        # Checks if the method_name is allowed to call on the pipeline.
        # Thus, it uses ducktyping
        # Returns the all kwargs that are provided to the pipeline and needed by method_name.
        pass

        ## TODO How to check allowed methods, we need to go through the whole graph. Rules should be:
        #    * Start at the end at the beginning only transform is allowed.
        #    * The first non-transformer determines the type of the pipeline!
        #    * During operation:
        #       * On transformer transform or inverse_transform is called
        #           * Inverse_transform is first expicitly added. Laterly, we can make it implicitly if it is on the y path.
        #       * On non-transformer, predict<X> is called
        #           * What happens with PI or something like that after deterministic forecasts?

    def _get_unique_id(self, skobject):
        return -1

    def _get_step(self, name):
        if name in self.steps:
            return self.steps[name]
        raise Exception("Required Input does not exist")

    def add_step(self, skobject, name, edges, **kwargs):
        """
        TODO
        """
        unique_id = self._get_unique_id(skobject)
        if not unique_id in self.model_dict:
            self.model_dict[unique_id] = skobject.clone()

        input_steps = {key: self._get_step(edge) for key, edge in edges.items()}
        step = Step(skobject, name, input_steps, kwargs, compuatation_setting=self.computation_setting)
        if name in self.steps:
            raise Exception("Name Conflict")

        self.steps[name] = step
        self._last_step_name = name
        return step


    def fit(self, X, y=None, **kwargs):
        # Fits the pipeline
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y

        self.computation_setting.method_resolution_order = ["transform", "predict"]
        self.computation_setting.kwargs.update(kwargs)
        # TODO Store additional kwargs

        # 4. call get_result or something similar on last step!
        self.steps[self._last_step_name].get_result(fit=True)

    def transform(self, X, y=None, **kwargs):
        # Implementation of transform, such methods also are required for predict, ...
        # 1. Check if transform is allowed. I.e., Check method needs to check if all steps implement transform + If all required params are passed
        if not self._method_allowed("transform"):
            raise Exception("TODO")

        # 2. Set transform as global method as well as provide all kwargs to step
        self.computation_setting.required_method = "transform"
        self.computation_setting.method_resolution_order = ["transform"]
        self.computation_setting.kwargs.update(kwargs)

        # 3. set data into start steps buffer!
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y

        # 4. call get_result or something similar on last step!
        return self.steps[self._last_step_name].get_result(fit=False).result


    def predict(self, X, y=None, **kwargs):
        # Implementation of transform, such methods also are required for predict, ...
        # 1. Check if transform is allowed. I.e., Check method needs to check if all steps implement transform or predict + If all required params are passed
        # 2. Set predict/transform as global methods
        if not self._method_allowed("predict"):
            raise Exception("TODO")

        # 2. Set transform as global method as well as provide all kwargs to step
        self.computation_setting.required_method = "predict"
        self.computation_setting.method_resolution_order = ["predict", "transform"]
        self.computation_setting.kwargs.update(kwargs) # TODO Update or overwrite?

        # 3. set data into start steps buffer!
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y

        # 4. call get_result or something similar on last step!
        return self.steps[self._last_step_name].get_result(fit=False).result

    def predict_interval(self, X, y=None, **kwargs):
        if not self._method_allowed("predict_interval"):
            raise Exception("TODO")

        # 2. Set transform as global method as well as provide all kwargs to step
        self.computation_setting.required_method = "predict_interval"
        self.computation_setting.method_resolution_order = ["predict_interval", "predict", "transform"]
        self.computation_setting.kwargs.update(kwargs) # TODO Update or overwrite?

        # 3. set data into start steps buffer!
        self.steps["X"].buffer = X
        self.steps["y"].buffer = y

        # 4. call get_result or something similar on last step!
        return self.steps[self._last_step_name].get_result(fit=False).result

    def predict_quantiles(self, *args, **kwargs):
        # Implementation of transform, such methods also are required for predict, ...
        # 1. Check if transform is allowed. I.e., Check method needs to check if all steps implement transform or predict / predict_quantiles? + If all required params are passed
        # 2. Set predict/transform as global methods
        pass


    def predict_proba(self, *args, **kwargs):
        # Implementation of transform, such methods also are required for predict, ...
        pass

    # TODO Weitere Methoden
    ...

    def _method_allowed(self, method):
        allowed_methods = ["transform"]
        method_resolution_order = ["transform"]
        for step_name, step in self.steps.items():
            print(step.get_allowed_method())
            if "transform" in step.get_allowed_method():
                pass # This would be okay
            elif method in step.get_allowed_method():
                pass # This would be okay
            else:
                # TODO for now raise an exception. However if PI based postprocessing or an Ensemble
                #   exist after a forecaster. There might be the case that predict could be possible.
                return False
        return True

