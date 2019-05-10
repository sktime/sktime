from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score
import numpy as np
class SKTimeScore(ABC):
    @abstractmethod
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
            Returns the result of the metric.
        """

    @abstractmethod
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

class ScoreAccuracy(SKTimeScore):
    """
    Calculates the accuracy between the true and predicted lables.
    """
    def __init__(self, round_predictions=True):
        """
        Parameters
        ----------
        round_predictions: Boolean
            Should the predictions be rounded before claculating the accuracy score. This is useful when the accuracy score is used on outputs produced by regressors.
        """
        self._round_predictions = round_predictions
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

        
        if self._round_predictions is True:
            y_pred = np.rint(y_pred)
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
            Tuple with average score and std error of the score
        """
        errors = (np.array(y_true) - np.array(y_pred)) ** 2
        errors = np.where(errors > 0, 1, 0)
        n = len(errors)
        
        std_score = np.std(errors)/np.sqrt(n) 
        sum_score = np.sum(errors)
        avg_score = sum_score/n

        return avg_score, std_score

class ScoreMSE(SKTimeScore):
    """
    Calculates the mean squared error between the true and predicted lables.
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
        errors = (y_true - y_pred) ** 2
        n = len(errors)
        
        std_score = np.std(errors)/np.sqrt(n) 
        sum_score = np.sum(errors)
        avg_score = sum_score/n

        return avg_score, std_score
