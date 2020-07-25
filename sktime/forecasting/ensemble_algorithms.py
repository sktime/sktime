from .online_experts import OnlineHedgeDoubling, \
                            OnlineHedgeIncremental, \
                            NormalHedge
from scipy.optimize import nnls
import numpy as np


HEDGE_EXPERTS = (OnlineHedgeDoubling, OnlineHedgeIncremental, NormalHedge)


class EnsembleAlgorithms(object):
    """Wrapper class to take ensemble algorithms and allow them to be easily set

    Parameters
    ----------
    n : float
    loss_func : function
    """
    def __init__(self, n, loss_func=None, **kwargs):
        self.n = n
        self.weights = np.ones(n)/n
        self.loss_func = loss_func

    def _predict(self, expert_predictions):
        """ Performs prediction by taking a weighted average of the expert
            predictions w.r.t the weights vector

        Parameters
        ----------
        expert_predictions : np.array(), shape=(time_axis,experts_axis)

        Returns
        -------
        predictions : np.array(), shape=(time_axis)
        """
        prediction = np.dot(self.weights, expert_predictions)

        return prediction

    def _update(self, expert_predictions, actual_values):
        """ Resets the weights over the experts by passing previous observations
            to the online_expert algorithm

        Parameters
        ----------
        expert_predictions : np.array(), shape=(time_axis,experts_axis)
        actual_values : np.array(), shape=(time_axis)
        """
        raise NotImplementedError()

    def _uniform_weights(self, n):
        """ Resets weights for n expert to uniform weights

        Parameters
        ----------
        n : int
        """
        self.n = n
        self.weights = np.ones(n)/n


class HedgeExpertEnsemble(EnsembleAlgorithms):
    """Wrapper class to take hedge style ensemble algorithms and
       allow them to be updated

    Parameters
    ----------
    n : int, number of experts
    T : int, time horizon
    online_expert : OnlineHedge()
    loss_func : function, loss function
    """

    def __init__(self, n, T, online_expert, loss_func=None):
        super().__init__(n=n, loss_func=loss_func)

        # Check if Online Expert is a Hedge Algorithm
        if online_expert not in HEDGE_EXPERTS:
            raise ValueError("Online Expert is not a Online Hedge Algorithm")

        # Resident Online Expert
        self.online_expert = online_expert(n=n, T=T, loss_func=loss_func)

    def _update(self, expert_predictions, actual_values):
        self.online_expert._update(expert_predictions, actual_values)
        self.weights = self.online_expert.weights


class NNLSEnsemble(EnsembleAlgorithms):
    """ Wrapper class to perform a non-negative least squares to fit to the experts
    Keeps tracks of all observations seen so far and fits to it

    Parameters
    ----------
    n : int, number of experts
    loss_func : function, loss function
    """

    def __init__(self, n, loss_func=None):
        super().__init__(n=n, loss_func=loss_func)
        self.total_expert_predictions = np.empty((n, 0))
        self.total_actual_values = np.empty((0))

    def _update(self, expert_predictions, actual_values):
        self.total_expert_predictions = np.concatenate(
            (self.total_expert_predictions, expert_predictions), axis=1)
        self.total_actual_values = np.concatenate(
            (self.total_actual_values, actual_values))
        weights, loss = nnls(self.total_expert_predictions.T,
                             self.total_actual_values)
        self.weights = weights
