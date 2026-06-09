"""Anomaly scores for interval evaluation."""

from ._from_cost import (
    LocalAnomalyScore,
    Saving,
    to_local_anomaly_score,
    to_saving,
)
from ._l2_saving import L2Saving

LOCAL_ANOMALY_SCORES = [
    LocalAnomalyScore,
]
SAVINGS = [
    Saving,
    L2Saving,
]
ANOMALY_SCORES = SAVINGS + LOCAL_ANOMALY_SCORES

__all__ = ANOMALY_SCORES + [
    to_local_anomaly_score,
    to_saving,
]
