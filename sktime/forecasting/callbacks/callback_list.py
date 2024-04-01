"""Callback list calls all callbacks in the list."""
from sktime.forecasting.callbacks.callback import Callback


class CallbackList(Callback):
    """Callback list calls all callbacks in the list."""

    def __init__(self, callbacks=None, forecaster=None, scores=None):
        if callbacks is None:
            callbacks = []
        if scores is None:
            scores = []
        self.callbacks = callbacks
        self.scores = scores
        self.forecaster = forecaster

    @property
    def forecaster(self):
        """Forecaster being evaluated."""
        return self._forecaster

    @property
    def scores(self):
        """Scores of the forecaster during evaluation."""
        return self._scores

    @scores.setter
    def scores(self, scores):
        if isinstance(scores, dict):
            scores = [value for sublist in scores.values() for value in sublist]
        for callback in self.callbacks:
            callback.score_metrics = scores

    @forecaster.setter
    def forecaster(self, forecaster):
        for callback in self.callbacks:
            callback.forecaster = forecaster

    def on_iteration(self, iteration, y_pred, x, result, update=None):
        """Call after each iteration."""
        for callback in self.callbacks:
            callback.on_iteration(iteration, y_pred, x, result, update=None)

    def on_iteration_start(self, update=None):
        """Call at the start of each iteration."""
        for callback in self.callbacks:
            callback.on_iteration_start(update=None)

    def on_iteration_end(self, results=None):
        """Call at the end of each iteration."""
        for callback in self.callbacks:
            callback.on_iteration_end(results)
