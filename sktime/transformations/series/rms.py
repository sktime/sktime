import numpy as np
import pandas as pd
from sktime.transformations.base import BaseTransformer


class RMSTransformer(BaseTransformer):
    """Root Mean Square (RMS) transformer.

    Computes RMS value of a time series.
    Useful for vibration and signal energy analysis.
    """

    _tags = {
        "capability:multivariate": True,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": True,
    }

    def __init__(self):
        super().__init__()

    def _fit(self, X, y=None):
        return self

    def _transform(self, X, y=None):
        import numpy as np
        import pandas as pd

        X_np = X.to_numpy().flatten()

        rms = np.sqrt(np.mean(X_np**2))

        return pd.DataFrame([rms])

    @classmethod
    
    def get_test_params(cls, parameter_set="default"):
        return {}