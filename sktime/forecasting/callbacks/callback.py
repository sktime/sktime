"""Callback class for callbacks during the evaluation of a forecaster."""

import pandas as pd


class Callback:
    """Callback class for callbacks during the evaluation of a forecaster."""

    def __init__(self):
        self._forecaster = None
        self._scores = []

    @property
    def scores(self):
        """Scores of the forecaster during evaluation."""
        return self._scores

    @property
    def forecaster(self):
        """Forecaster being evaluated."""
        return self._forecaster

    @forecaster.setter
    def forecaster(self, forecaster):
        self._forecaster = forecaster

    @scores.setter
    def scores(self, scores):
        self._scores = scores

    def on_iteration_start(self, update=None):
        """Call at the start of each iteration."""

    def on_iteration(self, iteration, y_pred, x, result: pd.DataFrame, update=None):
        """Call after each iteration."""

    def on_iteration_end(self, results=None):
        """Call at the end of each iteration."""
