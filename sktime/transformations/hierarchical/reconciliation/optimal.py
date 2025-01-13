import pandas as pd

from sktime.transformations.base import BaseTransformer

# Add engine for non-constrained reconciliation and for non-negative


class OptimalReconciler(BaseTransformer):
    def __init__(self, error_covariance_matrix: pd.DataFrame = None):
        self.error_covariance_matrix = error_covariance_matrix
        super().__init__()
