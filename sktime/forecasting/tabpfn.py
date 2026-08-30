from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import make_reduction

class TabPFNForecaster(BaseForecaster):
    """TabPFN Forecaster using the reduction delegator pattern."""

    _tags = {
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "python_dependencies": "tabpfn",
    }

    def __init__(self, window_length=10, strategy="recursive"):
        self.window_length = window_length
        self.strategy = strategy
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        from tabpfn import TabPFNClassifier
        
        self.forecaster_ = make_reduction(
            estimator=TabPFNClassifier(),
            window_length=self.window_length,
            strategy=self.strategy
        )
        
        self.forecaster_.fit(y, X, fh)
        return self

    def _predict(self, fh, X=None):
        return self.forecaster_.predict(fh, X)