from sklearn.base import BaseEstimator
from scipy.optimize import nnls
import numpy as np


class EnsembleAlgorithm(BaseEstimator):
    """Wrapper class to take ensemble algorithms and allow them to be easily set

    Parameters
    ----------
    n : float
    loss_func : function
    """
    def __init__(self, n=10, loss_func=None, **kwargs):
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

    def _modify_weights(self, new_array):
        """Performs a pointwise multiplication of the current
        weights with a new array of weights.

        Parameters
        ----------
        new_array : np.array()
        """
        self.weights = self.weights * new_array
        self.weights /= np.sum(self.weights)

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


class HedgeExpertEnsemble(EnsembleAlgorithm):
    """Wrapper class to take hedge style ensemble algorithms and
       allow them to be updated

    Parameters
    ----------
    n : float, number of experts
    T : int, forecasting horizon (in terms of timesteps)
    a : float, normalizing constant
    loss_func : function, loss function
    """

    def __init__(self, n=10, T=10, a=1, loss_func=None):
        super().__init__(n=n, loss_func=loss_func)
        self.T = T
        self.a = a
        self._uniform_weights(n)
        self.loss_func = loss_func


class NormalHedgeEnsemble(HedgeExpertEnsemble):
    """Implementation of A Parameter-free Hedging Algorithm,
    Kamalika Chaudhuri, Yoav Freund, Daniel Hsu (2009)

    Parameters
    ----------
    n : float, number of experts
    T : int, forecasting horizon (in terms of timesteps)
    a : float, normalizing constant
    loss_func : function, loss function
    """

    def __init__(self, n=10, a=1, loss_func=None, **kwargs):
        super().__init__(n=n, T=None, a=a, loss_func=loss_func)
        self.R = np.zeros(n)

    def _update(self, expert_predictions, actual_values, low_c=0.01):
        assert expert_predictions.shape[1] == len(actual_values), \
                                            "Time Dimension Matches"
        time_length = expert_predictions.shape[1]

        for i in range(time_length):
            loss_vector = np.array([
                          self.loss_func([prediction], [actual_values[i]])
                          for prediction in expert_predictions[:, i]
                          ])

            average_loss = np.dot(self.weights, loss_vector)

            instant_regret = (average_loss - loss_vector)
            self.R += instant_regret
            self._update_weights(low_c=low_c)

    def _update_weights(self, low_c=0.01):
        """Updates the weights on each of the experts by performing a potential
        function update with a root-finding search. low_c represents the lower
        bound on the window that the root finding is occuring over.

        Parameters
        ----------
        low_c : float
        """

        # Calculating Normalizing Constant
        R_plus = np.array(list(map(lambda x: 0 if 0 > x else x, self.R)))
        normalizing_R = np.max(R_plus)

        R_plus /= normalizing_R

        low_c = low_c
        high_c = (max(R_plus)**2)/2

        def pot(c):
            """Internal Potential Function

            Parameters
            ----------
            low_c : float

            Returns
            -------
            potential: float
            """
            return np.mean(np.exp((R_plus**2)/(2*c)))-np.e

        c_t = bisection(low_c, high_c, pot)

        def prob(r, c_t):
            """Internal Probability Function

            Parameters
            ----------
            r : float
            c_t : float

            Returns
            -------
            prob : float
            """
            return (r/c_t)*np.exp((r**2)/(2*c_t))

        self.weights = np.array([prob(r, c_t) for r in R_plus])
        self.weights /= np.sum(self.weights)


class NNLSEnsemble(EnsembleAlgorithm):
    """ Wrapper class to perform a non-negative least squares to fit to the experts
    Keeps tracks of all observations seen so far and fits to it

    Parameters
    ----------
    n : int, number of experts
    loss_func : function, loss function
    """

    def __init__(self, n=10, loss_func=None):
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


"""
Helper Functions
"""


def bisection(low, high, function, threshold=1e-8):
    """ Uses the bisection method in order to a root of a function
        within a specific interval

    Parameters
    ----------
    low : float, lower bound for the interval
    high : float, higher bound for the interval
    function: function
    threshold: float, abs distance from zero when the algorithm stops at

    Returns
    -------
    mid : float, value at which the funtion takes on a value within
          [-theshold/2,theshold]
    """

    left = low
    right = high

    if function(low) > 0:
        left = high
        right = low

    while abs(left - right) > 1e-8:
        mid = (left + right)/2
        if function(mid) > 0:
            right = mid
        else:
            left = mid

    return (left + right)/2
