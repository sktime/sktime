"""Base module for momentfm TimeSeriesOutputs."""

from dataclasses import dataclass

import numpy.typing as npt


@dataclass
class TimeseriesOutputs:
    """TimeseriesOutput class."""

    forecast: npt.NDArray = None
    anomaly_scores: npt.NDArray = None
    logits: npt.NDArray = None
    labels: int = None
    input_mask: npt.NDArray = None
    pretrain_mask: npt.NDArray = None
    reconstruction: npt.NDArray = None
    embeddings: npt.NDArray = None
    metadata: dict = None
    illegal_output: bool = False
