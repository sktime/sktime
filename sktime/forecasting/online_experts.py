__author__ = ["William Zheng"]
# __all__ = ["OnlineHedgeDoubling","OnlineHedgeIncremental", "NormalHedge"]

import numpy as np


class OnlineExperts(object):
    """
    Super-Class for general online expert style algorithms
    """

    def __init__(self, **kwargs):
        pass

    def _predict(self, expert_predictions):
        pass

    def _update(self, expert_predictions, actual_values):
        """Performs a weight update based on the loss between
        expert_predictions and the actual_values

        Parameters
        ----------
        expert_predictions : np.array(shape=(time_axis,experts_axis)
        actual_values : np.array(), shape=(time_axis)
        """
        pass

    def _modify_weights(self, new_array):
        """Performs a pointwise multiplication of the current
        weights with a new array of weights.

        Parameters
        ----------
        new_array : np.array()
        """
        self.weights = self.weights * new_array
        self.weights /= np.sum(self.weights)

    def _set_uniform(self, n):
        """
        Sets the weight vector uniform.

        Parameters
        ----------
        n : int, length of the uniform vector desired
        """
        self.weights = np.ones(n)/n


"""
Hedge Style Update Algorithms
"""


class OnlineHedge(OnlineExperts):
    """
    General Class for Hedge-Style Algorithms

    Parameters
    ----------
    n : float, number of experts
    T : int, forecasting horizon (in terms of timesteps)
    a : float, normalizing constant
    loss_func : function, loss function

    Hedge Papers:
    (1) A Decision-Theoretic Generalization of On-Line Learning \
                            and an Application to Boosting (1997)

    """

    def __init__(self, n=10, T=10, a=1, loss_func=None):
        self.n = n
        self.T = T
        self.a = a
        self._set_uniform(n)
        self.time = 0
        self.loss_func = loss_func

    def _predict(self):
        """Returns a vector over the experts to indicate how the experts
        are weighted (one given all the weight) chosen according to the
        weights probability distribution.

        Returns
        -------
        expert_array : np.array()
        """

        choosen_expert = np.random.choice(np.arange(self.n),
                                          size=1, replace=True, p=self.weights)

        expert_array = np.zeros(self.n)
        expert_array[choosen_expert] = 1

        return expert_array


class OnlineHedgeDoubling(OnlineHedge):
    """Online Hedge Loss Style algorithm with a fixed time horizon
    that doubles once the time horizon is exceeded.

    Parameters
    ----------
    n : float, number of experts
    T : int, forecasting horizon (in terms of timesteps)
    a : float, normalizing constant
    loss_func : function, loss function
    """

    def __init__(self, n=10, T=10, a=1, loss_func=None):
        super().__init__(n=n, T=T, a=a, loss_func=loss_func)
        self.epsilon = _define_epsilon(n, T, a=a)

    def _update(self, expert_predictions, actual_values):
        assert expert_predictions.shape[1] == len(actual_values), \
               "Time Dimension Matches"
        time_length = expert_predictions.shape[1]

        total_time = time_length+self.time

        t = int(np.floor(np.log2(total_time/self.T)))
        splits = [self.T*2**(i)-self.time for i in range(t)]
        splits = list(filter(lambda x: x >= 0, splits))
        partitions = np.split(np.arange(total_time-self.time), splits)

        for i in range(len(partitions)):
            self.time += len(partitions[i])

            if self.time > self.T:
                self.T = 2*self.T
                self.epsilon = _define_epsilon(self.n, self.T, self.a)

            losses = np.array([self.loss_func(expert_predictions[:, part],
                               actual_values[part]) for part in partitions[i]])
            self._modify_weights(np.prod(exp_loss(losses, self.epsilon), 0))


class OnlineHedgeIncremental(OnlineHedge):
    """Online Hedge Loss Style algorithm with a fixed time horizon
       that incrementally increases as time passes.

    Parameters
    ----------
    n : float, number of experts
    a : float, normalizing constant
    loss_func : function, loss function
    """

    def __init__(self, n=10, a=1, loss_func=None):
        super().__init__(n=n, T=None, a=a, loss_func=loss_func)

    def _update(self, expert_predictions, actual_values, loss_func=None):
        assert expert_predictions.shape[1] == len(actual_values), \
                                             "Time Dimension Matches"
        time_length = expert_predictions.shape[1]

        for i in range(time_length):
            self.time += 1
            epsilon = _define_epsilon(self.n, self.time, self.a)
            losses = self.loss_func(expert_predictions[:, i], actual_values[i])
            self._modify_weights(exp_loss(losses, epsilon))


class NormalHedge(OnlineHedge):
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
                          self.loss_func(prediction, actual_values[i])
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


def se(actual, expected):
    """
    Returns the squared error between the two arguments:

    Parameters
    ----------
    actual : float, actual value to be compared with
    expected : float, expected value to be compared with

    Returns
    -------
    se: squared error between the two values
    """
    return np.power(np.subtract(actual, expected), 2)


def mse(actual, expected):
    """
    Will return the mean squared error between the two arguments

    Parameters
    ----------
    actual : float, actual value to be compared with
    expected : float, expected value to be compared with

    Returns
    -------
    mse: mean squared error between the two values
    """
    return np.mean(se(actual, expected))


def _define_epsilon(n, T, a=1):
    """
    Calculates a factor that is used in determining loss in the hedge algorithm

    Parameters
    ----------
        n : int, number of experts present
        T : int, number of time steps taken
        a : float, value that we can use to scale our epsilon

    Returns
    -------
        epsilon : float, the theoretical epsilon (learning rate)
    """

    return np.sqrt(np.log(n)/T)*a


def exp_loss(x, epsilon):
    """ Exponential loss function

    Parameters
    ----------
        x : float
        epsilon : float, the theoretical epsilon (learning rate)

    Returns
    -------
        loss : float
    """

    return np.exp(-epsilon*x)
