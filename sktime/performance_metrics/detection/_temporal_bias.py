"""Temporal Bias metric for advance event detection."""

import numpy as np

from sktime.performance_metrics.detection._base import BaseDetectionMetric

class TemporalBias(BaseDetectionMetric):
    """Temporal Bias metric as proposed by Escobar et al. (2021).

    Computes the temporal distance between predicted events and real events.
    """

    _tags = {
        "metric_type": "detection",
        "lower_is_better": True,
    }

    def __init__(self):
        super().__init__()

    def _evaluate(self, y_true, y_pred, **kwargs):
        """Calculate the Temporal Bias score.

        Parameters
        ----------
        y_true : 1D np.array or pd.Series
            Actual event locations/timestamps.
        y_pred : 1D np.array or pd.Series
            Predicted event locations/timestamps.

        Returns
        -------
        float
            The total temporal bias score.
        """
        # Convert to numpy arrays if they aren't already
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        if len(y_pred) == 0 or len(y_true) == 0:
            return 0.0 # Handle edge cases
        
        total_bias = 0.0

        # For every prediction, find the distance to the closest actual event
        for pred in y_pred:
            # Calculate absolute distance to all true events
            distances = np.abs(y_true - pred)
            # Find the minimum distance (closest event)
            closest_distance = np.min(distances)
            # Add to our total bias
            total_bias += closest_distance
        
        # If you can average it or return the sum, check the paper's specific format
        return total_bias / len(y_pred)
