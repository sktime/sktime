"""Common file for momentfm."""

from dataclasses import dataclass


@dataclass
class TASKS:
    """Tasks modules for momentfm models."""

    RECONSTRUCTION: str = "reconstruction"
    FORECASTING: str = "forecasting"
    CLASSIFICATION: str = "classification"
    EMBED: str = "embedding"
