"""Module for online anomaly detection using Bayesian methods.

References
----------
TODO

"""

import numpy as np
import pandas as pd
from scipy import stats

from sktime.annotation.base import BaseSeriesAnnotator


class StudentTDistribution:
    """Represent a Student's T distribution for modeling online anomalies."""

    def __init__(self, alpha=0.1, beta=0.001, kappa=1.0, mu=0.0):
        """
        Initialize the distribution parameters.

        Parameters
        ----------
        alpha : float
            Shape parameter of the distribution.
        beta : float
            Scale parameter of the distribution.
        kappa : float
            Degrees of freedom of the distribution.
        mu : float
            Mean of the distribution.
        """
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def pdf(self, data):
        """
        Calculate the probability density function for the random variable.

        More details can be found in the scipy documentation:
        http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html
        """
        return stats.t.pdf(
            x=data,
            df=2 * self.alpha,
            loc=self.mu,
            scale=np.sqrt((self.beta * (self.kappa + 1)) / (self.alpha * self.kappa)),
        )

    def updateTheta(self, data):
        """
        Update the distribution parameters based on new data.

        Parameters
        ----------
        data : float
            New data point to update the distribution's parameters.
        """
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (data - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


class OnlineBayesCPD(BaseSeriesAnnotator):
    """
    Implement a custom series annotator for online Bayesian changepoint detection.

    Parameters
    ----------
    observation_likelihood : callable
        A function computes the likelihood of observing the data.
    max_run_length : int, optional
        The maximum length of the run without a changepoint, by default 500.
    lambda_const : float, optional
        The decay rate for hazard function, by default 250.

    Attributes
    ----------
    max_run_length : int
        The maximum run length considered for changepoint detection.
    max_length_probs : np.ndarray
        A matrix holding the probabilities of run lengths.
    record_number : int
        The current record number in the sequence being analyzed.
    previous_max_run : int
        The maximum run length detected in the previous step.
    anomaly_scores : list
        A list accumulating anomaly scores for each observation.
    """

    def __init__(
        self,
        observation_likelihoood,
        max_run_length=500,
        lambda_const=250,
    ):
        super().__init__()
        # Setup the matrix that will hold our beliefs about the current
        # run lengths. We'll initialize it all to zero at first. For efficiency
        # we preallocate a data structure to hold only the info we need to detect
        # change points: columns for the current and next recordNumber, and a
        # sufficient number of rows (where each row represents probabilites of a
        # run of that length).
        self.max_run_length = max_run_length
        self.max_length_probs = np.zeros((self.max_run_length + 2, 2))
        # Record 0 is a boundary condition, where we know the run length is 0.
        self.max_length_probs[0, 0] = 1.0

        # Init variables for state.
        self.record_number = 0
        self.previous_max_run = 1

        # Define algorithm's helpers.
        self.observation_likelihoood = observation_likelihoood
        self.lambda_const = lambda_const

        self.anomaly_scores = []

    def _fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series

        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        return self

    def _predict(self, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Anomaly scores predictions
        """
        for x in X:
            if self.record_number > 0:
                self.max_length_probs[:, 0] = self.max_length_probs[:, 1]
                self.max_length_probs[:, 1] = 0

            # Evaluate the predictive distribution for the new datum under each of
            # the parameters. This is standard Bayesian inference.
            pred_probs = self.observation_likelihoood.pdf(x)

            # Evaluate the hazard function for this interval
            hazard = self._hazard_function(self.record_number + 1, self.lambda_const)

            # We only care about the probabilites up to max_run_length.
            run_length_index = min(self.record_number, self.max_run_length)

            # Evaluate the growth probabilities -- shift the probabilities down and to
            # the right, scaled by the hazard function and the predictive probabilities.
            # for r in range(run_length_index + 1):
            #     pred_prob = pred_probs[r]
            #     p_no_change = (1 - hazard[r]) * pred_prob
            #     self.max_length_probs[r + 1, 1] = p_no_change
            #                   * self.max_length_probs[r, 0]
            self.max_length_probs[1 : run_length_index + 2, 1] = (
                self.max_length_probs[: run_length_index + 1, 0]
                * pred_probs[: run_length_index + 1]
                * (1 - hazard)[: run_length_index + 1]
            )

            # Evaluate the probability that there *was* a changepoint and we're
            # accumulating the probability mass back down at run length = 0.
            # for r in range(run_length_index + 1):
            #     self.max_length_probs[0, 1] += \
            #     self.max_length_probs[r, 0] * hazard[r] * pred_probs[r]
            self.max_length_probs[0, 1] = np.sum(
                self.max_length_probs[: run_length_index + 1, 0]
                * pred_probs[: run_length_index + 1]
                * hazard[: run_length_index + 1]
            )

            # Renormalize the run length probabilities for improved numerical stability.
            self.max_length_probs[:, 1] = (
                self.max_length_probs[:, 1] / self.max_length_probs[:, 1].sum()
            )
            self.observation_likelihoood.updateTheta(x)
            # Get the current run length with the highest probability.
            max_recursive_run_length = self.max_length_probs[:, 1].argmax()

            # To calculate anomaly scores from run length probabilites we have several
            # options, implemented below:
            #   1. If the max probability for any run length is the run length of 0, we
            #   have a changepoint, thus anomaly score = 1.0.

            # is_anomaly = np.argmax(self.max_length_probs[:, 0]) == 0
            # anomaly_score = 1.0 if is_anomaly else 0.0

            # #   2. The anomaly score is the probability of run length 0.

            # anomaly_score = self.max_length_probs[0, 0]

            # #   3. Compute a score by assuming a change in sequence from a previously
            # #   long run is more anomalous than a change from a short run.
            if max_recursive_run_length < self.previous_max_run:
                anomaly_score = 1 - (
                    float(max_recursive_run_length) / self.previous_max_run
                )
            else:
                anomaly_score = 0.0
            # # Option 3 results in the best anomaly detections (by far)

            self.anomaly_scores.append(anomaly_score)

            # Update state vars.
            self.record_number += 1
            self.previous_max_run = max_recursive_run_length

        y_pred = pd.Series(self.anomaly_scores)
        return y_pred

    def _hazard_function(self, arraySize, lambdaConst):
        """Estimate the changepoint prior."""
        return np.ones(arraySize) / float(lambdaConst)

    # todo: return default parameters, so that a test instance can be created
    # required for automated unit and integration testing of estimator

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        student_t = StudentTDistribution()
        params = {
            "observation_likelihoood": student_t,
            "max_run_length": 500,
            "lambda_const": 250,
        }
        return params
