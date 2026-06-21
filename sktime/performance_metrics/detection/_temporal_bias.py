from sktime.performance_metrics.detection._base import BaseDetectionMetric

import numpy as np
import pandas as pd
from scipy.stats import skew

class TemporalBias(BaseDetectionMetric):
    """
    A distance-based metric to evaluate whether a time series detection algorithm tends to detect events before or after the real events happen.
    
    Temporal bias is the distances between the positions of predicted events and real events. Negative values mean early detection,
    while positive values mean late detection.
    For each real event, there are two kinds of distances calculated,
    a posteriori and a priori distances, and they make up the temporal bias.
    The skewness of an array of temporal bias scores indicates the tendency of the algorithm being evaluated.
    Positive skewness indicates the algorithm's tendency to detect an event before it really happens, and
    negative skewness means the algorithm tends to give delayed detections.
    
    Reference: Escobar, L., Salles, R., Lima, J., Gea, C., Baroni, L., Ziviani, A., Pires, P., Delicato, F., Coutinho, R., Assis, L., & Ogasawara, E. (2021).
               Evaluating Temporal Bias in Time Series Event Detection Methods. Journal of Information and Data Management, 12(3).
               https://doi.org/10.5753/jidm.2021.1968
    
    Attributes
    ----------
    temporal_bias: array-like,
        The temporal bias for each real event. The temporal bias is obtained as the minimum between the a posteriori and a priori distances.
    """
    
    _tags = {
        "lower_is_better": False
    }
    
    def __init__(self):
        self.temporal_bias = []
        
        super().__init__()
        
    def _evaluate(self, y_pred, y_true, X=None):
        """Compute the temporal bias for each real event.
        
        Parameters
        ----------
        y_true : time series in ``sktime`` compatible data container format.
            Ground truth (correct) event locations, in ``X``.

            Not required if unsupervised metric,
            that is, if tag ``requires_y_true`` is False.

            Should be ``pd.DataFrame`` of one column named ``ilocs``.

            For further details on data format, see glossary on :term:`mtype`.

        y_pred : time series in ``sktime`` compatible data container format
            Detected events to evaluate against ground truth.
            Must be of same format as ``y_true``, same indices and columns if indexed.
            
        X: ignored for API consistency.
        
        Returns:
        skew: float,
            The skewness of the temporal_bias (or bias for short).
            Positive skewness indicates anticipated detection, while a negative one indicates delayed detections.
        """
           
        bias = []
        E = y_pred['ilocs'].values # detected events set
        R = y_true['ilocs'].values # real events set
        
        for i in range(len(R)):
            reference_event = R[i]
            # dividing E into two subsets: the early and the late detection
            early_detections = E[E <= reference_event]
            late_detections = E[E > reference_event]
            
            # if there is no early detections,
            # that means there are only late detections
            if len(early_detections) == 0:
                post_i = self._posteriori(late_detections, reference_event)
                bias.append(post_i)
            
            # if there is no late detections,
            # that means there are only early detections
            if len(late_detections) == 0:
                prior_i = self._priori(early_detections, reference_event)
                bias.append(-prior_i)
            
            if (len(early_detections) > 0) and (len(late_detections) > 0):           
                prior_i = self._priori(early_detections, reference_event)
                post_i = self._posteriori(late_detections, reference_event)
                bias_i = post_i if post_i < prior_i else -prior_i
                bias.append(bias_i)
        
        self.temporal_bias = bias
        return skew(bias)
        
    def _priori(self, detections, reference_true_event):
        """Compute the priori distance, which is the distance between the early or anticipated detection and the real events.
        
        Parameters:
            detections: array-like,
                a set of predicted events which are detected before the real, reference event happens.
            reference_true_event: int,
                the time point when the true event happens.
        """
        
        return np.abs(reference_true_event - np.max(detections))
    
    def _posteriori(self, detections, reference_true_event):
        """Compute the posteriori distance, which is the distance between the late or delayed detection and the real events.
        
        Parameters:
            detections: array-like,
                a set of predicted events which are detected after the real, reference event happens.
            reference_true_event: int,
                the time point when the true event happens.
        """
        # the case where there is no events detected after the reference_true_event
        
        return np.abs(np.min(detections) - reference_true_event)
    
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        param1 = {}

        return param1
        