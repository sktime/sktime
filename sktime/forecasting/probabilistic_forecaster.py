from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_y
import numpy as np
from scipy.stats import poisson, nbinom

class ProbabilisticIntermittentForecaster(BaseForecaster):
    """
    Probabilistic forecaster for intermittent demand using:
    - Poisson distribution
    - Negative Binomial Hurdle model
    """

    _tags = {
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.Series",
    }

    def __init__(self, model_type="poisson"):
        self.model_type = model_type
        self.lambda_ = None  # for Poisson
        self.mean_ = None    # for NB
        self.dispersion_ = None  # for NB
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        y = check_y(y)

        # Poisson: use mean of non-zero observations
        if self.model_type == "poisson":
            self.lambda_ = y[y > 0].mean()

        # NB Hurdle: fit simple mean and variance for demonstration
        elif self.model_type == "nb_hurdle":
            non_zero = y[y > 0]
            self.mean_ = non_zero.mean()
            self.dispersion_ = max(non_zero.var() - self.mean_, 0.01)

        else:
            raise ValueError(f"Unknown model_type={self.model_type}")

        return self

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=0.05):
        n_steps = len(fh) if fh is not None else 1

        if self.model_type == "poisson":
            # Predict mean for each step
            y_pred = np.full(n_steps, self.lambda_)

        elif self.model_type == "nb_hurdle":
            # Predict mean for each step (simple NB hurdle approach)
            y_pred = np.full(n_steps, self.mean_)

        else:
            raise ValueError(f"Unknown model_type={self.model_type}")

        return y_pred
