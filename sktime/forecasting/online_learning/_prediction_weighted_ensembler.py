# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import bisect
from scipy.optimize import nnls


class _PredictionWeightedEnsembler:
    """Wrapper class to handle ensemble algorithms that use multiple forecasters
    for prediction. We implement default methods for setting uniform weights,
    updating and prediction.

    Parameters
    ----------
    n_estimators : float
        number of estimators
    loss_func : function
        loss function which follows sklearn.metrics API, for updating weights
    """

    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, n_estimators=10, loss_func=None):
        self.n_estimators = n_estimators
        self.weights = np.ones(n_estimators) / n_estimators
        self.loss_func = loss_func
        super(_PredictionWeightedEnsembler, self).__init__()

    def _predict(self, y_pred):
        """Performs prediction by taking a weighted average of the estimator
            predictions w.r.t the weights vector

        Parameters
        ----------
        y_pred : np.array(), shape=(time_axis,estimator_axis)
            array with predictions from the estimators

        Returns
        -------
        predictions : np.array(), shape=(time_axis)
            array with our predictions
        """
        prediction = np.dot(self.weights, y_pred)

        return prediction

    def _modify_weights(self, new_array):
        """Performs a pointwise multiplication of the current
        weights with a new array of weights.

        Parameters
        ----------
        new_array : np.array()
            input array for pointwise multiplication
        """
        self.weights = self.weights * new_array
        self.weights /= np.sum(self.weights)

    def _update(self, y_pred, y_true):
        """Resets the weights over the estimators by passing previous observations
            to the weighting algorithm

        Parameters
        ----------
        y_pred : np.array(), shape=(time_axis,estimator_axis)
            array with predictions from the estimators
        y_true : np.array(), shape=(time_axis)
            array with actual values for predicted quantity
        """
        raise NotImplementedError()

    def _uniform_weights(self, n_estimators):
        """Resets weights for n estimator to uniform weights

        Parameters
        ----------
        n_estimators : int
            number of estimators
        """
        self.n = n_estimators
        self.weights = np.ones(n_estimators) / n_estimators


class HedgeExpertEnsemble(_PredictionWeightedEnsembler):
    """Wrapper class to set parameters for hedge-style ensemble algorithms with
    a forecasting horizon and normalizing constant.

    Parameters
    ----------
    n_estimators : float
        number of estimators
    T : int
        forecasting horizon (in terms of timesteps)
    a : float
        normalizing constant
    loss_func : function
        loss function which follows sklearn.metrics API, for updating weights
    """

    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, n_estimators=10, T=10, a=1, loss_func=None):
        super().__init__(n_estimators=n_estimators, loss_func=loss_func)
        self.T = T
        self.a = a
        self._uniform_weights(n_estimators)
        self.loss_func = loss_func


class NormalHedgeEnsemble(HedgeExpertEnsemble):
    """Implementation of A Parameter-free Hedging Algorithm,
    Kamalika Chaudhuri, Yoav Freund, Daniel Hsu (2009) as a hedge-style
    algorithm.

    Parameters
    ----------
    n_estimators : float
        number of estimators
    T : int
        forecasting horizon (in terms of timesteps)
    a : float
        normalizing constant
    loss_func : function
        loss function which follows sklearn.metrics API, for updating weights
    """

    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, n_estimators=10, a=1, loss_func=None):
        super().__init__(n_estimators=n_estimators, T=None, a=a, loss_func=loss_func)
        self.R = np.zeros(n_estimators)

    def update(self, y_pred, y_true, low_c=0.01):
        """Resets the weights over the estimators by passing previous observations
            and updating based on Normal Hedge.

        Parameters
        ----------
        y_pred : np.array(), shape=(time_axis,estimator_axis)
            array with predictions from the estimators
        y_true : np.array(), shape=(time_axis)
            array with actual values for predicted quantity
        """
        assert y_pred.shape[1] == len(y_true), "Time Dimension Matches"
        time_length = y_pred.shape[1]

        for i in range(time_length):
            loss_vector = np.array(
                [
                    self.loss_func([prediction], [y_true[i]])
                    for prediction in y_pred[:, i]
                ]
            )

            average_loss = np.dot(self.weights, loss_vector)

            instant_regret = average_loss - loss_vector
            self.R += instant_regret
            self._update_weights(low_c=low_c)

    def _update_weights(self, low_c=0.01):
        """Updates the weights on each of the estimators by performing a potential
        function update with a root-finding search. low_c represents the lower
        bound on the window that the root finding is occuring over.

        Parameters
        ----------
        low_c : float
            lowest value that c can take
        """

        # Calculating Normalizing Constant
        R_plus = np.array(list(map(lambda x: 0 if 0 > x else x, self.R)))
        normalizing_R = np.max(R_plus)

        R_plus /= normalizing_R

        low_c = low_c
        high_c = (max(R_plus) ** 2) / 2

        def _pot(c):
            """Internal Potential Function

            Parameters
            ----------
            low_c : float
                lowest value that c can take

            Returns
            -------
            potential: float
            """
            return np.mean(np.exp((R_plus ** 2) / (2 * c))) - np.e

        c_t = bisect(_pot, low_c, high_c)

        def _prob(r, c_t):
            """Internal Probability Function

            Parameters
            ----------
            r : float
                regret
            c_t : float
                current value for c

            Returns
            -------
            prob : float
                probability
            """
            return (r / c_t) * np.exp((r ** 2) / (2 * c_t))

        self.weights = np.array([_prob(r, c_t) for r in R_plus])
        self.weights /= np.sum(self.weights)


class NNLSEnsemble(_PredictionWeightedEnsembler):
    """Ensemble class that performs a non-negative least squares to fit to the
    estimators. Keeps track of all observations seen so far and fits to it.

    Parameters
    ----------
    n_estimators: int
        number of estimators
    loss_func : function
        loss function which follows sklearn.metrics API, for updating weights
    """

    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, n_estimators=10, loss_func=None):
        super().__init__(n_estimators=n_estimators, loss_func=loss_func)
        self.total_y_pred = np.empty((n_estimators, 0))
        self.total_y_true = np.empty(0)

    def update(self, y_pred, y_true):
        self.total_y_pred = np.concatenate((self.total_y_pred, y_pred), axis=1)
        self.total_y_true = np.concatenate((self.total_y_true, y_true))
        weights, loss = nnls(self.total_y_pred.T, self.total_y_true)
        self.weights = weights
