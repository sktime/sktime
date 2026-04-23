import numpy as np
from sktime.transformations.base import BaseTransformer


class RMSTransformer(BaseTransformer):
    """Root Mean Square (RMS) transformer.

    Computes the RMS value of a time series.
    Useful for vibration and signal energy analysis.
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "X_inner_mtype": ["pd.Series","numpy.ndarray"],
        "fit_is_empty": True,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
    import numpy as np

    # Convert pandas Series → numpy
    if hasattr(X, "values"):
        X = X.values

    if X.ndim == 1:
        return np.array([np.sqrt(np.mean(X**2))])

    return np.sqrt(np.mean(X**2, axis=1)).reshape(-1, 1)