__all__ = ["Accuracy", "MSE"]
__author__ = ["Viktor Kazakov", "Markus Löning"]


import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sktime.benchmarking.base import BaseMetric


class Accuracy(BaseMetric):
    """
    Calculates the accuracy between the true and predicted lables.
    """

    def calculate(self, y_true, y_pred):
        """
        Main method for performing the calculations.

        Parameters
        ----------
        y_true: array
            True dataset labels.
        y_pred: array
            predicted labels.


        Returns
        -------
        float
            The accuracy of the prediction.
        """
        return accuracy_score(y_true, y_pred)

    def calculate_per_dataset(self, y_true, y_pred):
        """
        Calculates the loss per dataset

        Parameters
        ----------
        y_true: array
            True dataset labels.
        y_pred: array: 
            predicted labels.
        
        
        Returns
        -------
        tuple
            Tuple with average score and stderr error of the score
        """
        n_instances = len(y_true)
        pointwise_metrics = y_true == y_pred

        stderr = np.std(pointwise_metrics) / np.sqrt(n_instances - 1)
        mean = np.mean(pointwise_metrics)
        return mean, stderr


class MSE(BaseMetric):
    """
    Calculates the mean squared error between the true and predicted labels.
    """

    def calculate(self, y_true, y_pred):
        """
        Main method for performing the calculations.

        Parameters
        ----------
        y_true : array
            True dataset labels.
        y_pred : array
            predicted labels.

        Returns
        -------
        float
            The mean squared error of the prediction.
        """
        return mean_squared_error(y_true, y_pred)

    def calculate_per_dataset(self, y_true, y_pred):
        """
        Calculates the loss per dataset

        Parameters
        ----------
        y_true: array
            True dataset labels.
        y_pred: array: 
            predicted labels.
        
        Returns
        -------
        float
            Returns the result of the metric.
        """
        n_instances = len(y_true)
        pointwise_metrics = (y_true - y_pred) ** 2

        mean = np.mean(pointwise_metrics)
        stderr = np.stderr(pointwise_metrics) / np.sqrt(n_instances - 1)
        return mean, stderr
