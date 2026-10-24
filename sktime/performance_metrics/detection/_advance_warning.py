import numpy as np
import pandas as pd
from sktime.performance_metrics.detection._base import BaseDetectionMetric

class NormalizedAdvanceWarningScore(BaseDetectionMetric):
    """
    Normalized Advance Warning Score for early event detection.
    
    This metric evaluates advance event detection by rewarding predictions 
    that occur before the actual event, up to a specified maximum lead time.
    Detections occurring after the event receive a score of 0.0.
    
    Parameters
    ----------
    max_lead_time : float, default=10.0
        The maximum warning time window. Warnings earlier than or equal to 
        this value achieve a perfect score of 1.0. As the warning gets closer 
        to the event time, the score decreases linearly to 0.0.
    """

    # These tags tell the sktime engine how to handle the data before passing it to _evaluate
    _tags = {
        "lower_is_better": False,  # A higher score (closer to 1.0) is better
        "requires_X": False,       # We don't need external features for this math
        "requires_y_true": True,
        "scitype:y": "points",     # Enforces the 'ilocs' DataFrame structure
    }

    def __init__(self, max_lead_time=10.0):
        self.max_lead_time = max_lead_time
        super().__init__()

    def _evaluate(self, y_true, y_pred, X=None):
        """
        Evaluate the advance warning score.
        The base class guarantees y_true and y_pred are pd.DataFrames with an 'ilocs' column.
        """
        # Extract the raw integer index arrays
        true_locs = y_true["ilocs"].to_numpy()
        pred_locs = y_pred["ilocs"].to_numpy()

        # Handle edge cases (no events happened)
        if len(true_locs) == 0:
            return 1.0 if len(pred_locs) == 0 else 0.0

        scores = []
        
        # Iterate through actual events to find the best early warning for each
        for t_event in true_locs:
            # Find all predictions that happened BEFORE or EXACTLY ON the event
            early_warnings = pred_locs[pred_locs <= t_event]
            
            if len(early_warnings) == 0:
                # The model missed the event entirely, or the warning came too late
                scores.append(0.0)
            else:
                # Grab the closest warning before the event
                best_warning = np.max(early_warnings)
                lead_time = t_event - best_warning
                
                # Calculate the normalized score and clip between 0.0 and 1.0
                score = np.clip(lead_time / self.max_lead_time, a_min=0.0, a_max=1.0)
                scores.append(score)

        # Return the mean score across all actual events
        return np.mean(scores)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator test registry."""
        return [
            {"max_lead_time": 5.0}, 
            {"max_lead_time": 15.0}
        ]