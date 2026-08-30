# sktime/forecasting/_causal_var.py

import pandas as pd
from pgmpy.models import BayesianNetwork
from sktime.forecasting.base import BaseForecaster
from statsmodels.tsa.api import VAR


class CausalVAR(BaseForecaster):
    """A Vector Autoregressive (VAR) model that prunes regressors based on a causal graph.

    This forecaster uses a pre-supplied causal graph to determine which time series
    (endogenous variables) act as predictors for others. For each variable being
    forecasted, it fits a separate VAR model that only includes its direct causes
    from the graph as external regressors.

    This approach can lead to simpler, more robust, and more interpretable models
    by eliminating spurious correlations.

    Parameters
    ----------
    causal_graph : pgmpy.models.BayesianNetwork
        A directed acyclic graph representing the causal relationships between
        variables. Node names must correspond to column names in the input `y` series.
    maxlags : int, optional (default=None)
        The maximum number of lags to be included in the VAR model. If None, the
        lag order is selected by the VAR model's default criteria (BIC).
        Passed directly to statsmodels.tsa.api.VAR.
    """

    _tags = {
        "scitype:y": "multivariate",
        "requires-fh-in-fit": False,
        "y_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,  # This forecaster does not use a separate X
        "authors": ["YourGitHubUsername"],
        "maintainers": ["YourGitHubUsername"],
    }

    def __init__(self, causal_graph: BayesianNetwork, maxlags: int = None):
        if not isinstance(causal_graph, BayesianNetwork):
            raise TypeError("causal_graph must be an instance of pgmpy.models.BayesianNetwork")
        self.causal_graph = causal_graph
        self.maxlags = maxlags
        super(CausalVAR, self).__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        For each variable in y, a separate VAR model is fitted. The model
        for variable 'i' is fitted using only the columns of y that are
        identified as direct causes of 'i' in `self.causal_graph`.
        """
        self.models_ = {}
        variables = y.columns
        self._fitted_causal_subsets = {}

        # Validate that all graph nodes are in the data columns
        for node in self.causal_graph.nodes():
            if node not in variables:
                raise ValueError(
                    f"Node '{node}' from causal_graph is not present in the "
                    f"training data columns: {variables.tolist()}"
                )

        for var in variables:
            # A variable's future is caused by its own past and the past of its direct causes.
            causes = [var] + list(self.causal_graph.predecessors(var))
            # Ensure causes are unique and are a subset of available variables
            causal_subset = sorted(list(set(c for c in causes if c in variables)))
            
            y_subset = y[causal_subset]
            self._fitted_causal_subsets[var] = causal_subset
            
            model = VAR(y_subset)
            self.models_[var] = model.fit(maxlags=self.maxlags)

        return self

    def _predict(self, fh, X=None):
        """Generate forecasts for the given forecast horizon.

        This method iteratively predicts one step at a time, as the prediction for
        one variable at step `t` may be needed for predicting another variable at `t+1`.
        However, for a standard VAR model, we can use the `forecast` method which
-        handles this internally. We predict for each causal subset and then combine
+        handles this internally. We predict for each causal subset and then combine.
        """
        # Collect all predictions in a dictionary
        predictions = {}

        for var, model_fit in self.models_.items():
            causal_subset = self._fitted_causal_subsets[var]
            y_subset_hist = self._y[causal_subset]
            
            # Get the required number of lags for prediction
            lags = model_fit.k_ar
            if lags == 0:  # Handle case with no lags (e.g., model on constants)
                # Predict by taking the mean of the subset
                y_forecast_subset = pd.concat(
                    [y_subset_hist.mean().to_frame().T] * len(fh), ignore_index=True
                )
            else:
                y_last_lags = y_subset_hist.values[-lags:]
                forecast_values = model_fit.forecast(y=y_last_lags, steps=len(fh))
                
                # Format the forecast into a pandas DataFrame
                y_forecast_subset = pd.DataFrame(forecast_values, columns=causal_subset)

            # We only care about the forecast for the target variable `var`
            predictions[var] = y_forecast_subset[var]

        # Combine the individual series forecasts into a single DataFrame
        forecast_df = pd.DataFrame(predictions)
        
        # Set the correct forecast index
        forecast_df.index = self.fh.to_absolute(self._y.index[-1])

        return forecast_df