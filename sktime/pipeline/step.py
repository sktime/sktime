from inspect import _ParameterKind

import pandas as pd

from sktime.pipeline.computation_setting import ComputationSetting
import inspect

class StepResult:
    def __init__(self, result, mode):
        self.result = result
        self.mode = mode



class Step():
    def __init__(self, skobject, name, input_edges, params, compuatation_setting: ComputationSetting):
        self.buffer = None
        self.skobject = skobject
        self.name = name
        self.input_edges = input_edges
        self.params = params
        self.computation_setting = compuatation_setting

    def get_allowed_method(self):
        if self.skobject is None:
            return ["transform"] # TODO very hacky
        "Returns a list of allowed methods of the skobject or the method specified by the user."
        return dir(self.skobject)

    def get_result(self, fit=False):
        # TODO Probably separate get_result from fit?
        #      Fit should call fit only if required (if a module is added multiple times than fit only the first occurence)
        #      Fit asks for input data only if it needs to be fitted or asked for input_data
        if self.input_edges is None:
            # If the input_edges are none that the step is a first step.
            return StepResult(self.buffer, "")
        # 1. Get results from all previous steps!
        # TODO should fetch y only if fit should be called. If not do not fetch it....
        #      Would not work if y can also be passed to predict... For global model forecasting
        #      How can we derive if y should be fetched or not...
        #      Fetch y only if y is passed to pipeline.predict(...) this should solve it.
        input_data = {step_name : step.get_result(fit=fit) for step_name, step in self.input_edges.items()}

        # 2. Get the method that should be called on skobject
        mro = self.computation_setting.method_resolution_order
        if "method" in self.params:
            mro = [self.params["method"]]
        if hasattr(self.skobject, "fit") and fit and not self.skobject.is_fitted:
            kwargs = self._extract_kwargs("fit")
            self.skobject.fit(**{k: in_data.result for k, in_data in input_data.items() }, **kwargs)


        for method in mro:
            if hasattr(self.skobject, method):
                kwargs = self._extract_kwargs(method)
                if "fh" in kwargs and fit:
                    # TODO check this if it works with numpy. Check if this can be done more generalized!
                    #      Here should be nothing that is only focusing on a specific estimator/...
                    kwargs["fh"] = input_data["y"].result.index if hasattr(input_data["y"].result, "index") else range(len(input_data["y"].result))
                all_none = True
                mode = ""
                for inp in input_data.values():
                    if inp.mode != "":
                        mode = inp.mode
                    if not inp.result is None:
                        all_none = False
                        # TODO make this prettier. Returns None if no input data is provided.
                if all_none:
                    return StepResult(None, "")
                # 3. Call method on skobject and return result
                if mode == "proba":
                    # TODO fix the case if we need to apply this to X and y?
                    idx = input_data["X"].result.columns
                    n = idx.nlevels
                    idx_low = idx.droplevel(0).unique()
                    yt = dict()
                    for ix in idx:
                        levels = list(range(1, n))
                        if len(levels) == 1:
                            levels = levels[0]
                        yt[ix] = input_data["X"].result[ix]
                        # deal with the "Coverage" case, we need to get rid of this
                        #   i.d., special 1st level name of prediction objet
                        #   in the case where there is only one variable
                        #if len(yt[ix].columns) == 1:
                        #    temp = yt[ix].columns
                        #    yt[ix].columns = input_data["X"].result.columns
                        yt[ix] = getattr(self.skobject, method)(X=yt[ix],
                            **kwargs
                        ).to_frame()
                    result = pd.concat(yt.values(), axis=1)
                else:
                    result = getattr(self.skobject, method)(
                        **dict(filter(lambda k: k[0] in inspect.getfullargspec(getattr(self.skobject, method)).args,
                                      {k: in_data.result for k, in_data in input_data.items() }.items())),
                        **kwargs
                    )

                mode = "proba" if ("predict_interval" == method) or (mode == "proba") else ""
                return StepResult(result, mode)
            # TODO fill buffer to save

    def predict_classifier(self):
        pass

    def predict_forecaster(self):
        pass

    def transform(self):
        pass

    def inverse_transform(self):
        pass

    def _extract_kwargs(self, method_name):
        use_kwargs = {}
        method = getattr(self.skobject, method_name)
        method_signature = inspect.signature(method).parameters

        for name, param in method_signature.items():
            if name in self.computation_setting.kwargs:
                use_kwargs[name] = self.computation_setting.kwargs[name]
        return use_kwargs
