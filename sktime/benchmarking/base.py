from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self, path, name):
        self.path = path
        self.name = name

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(name={self.name})"

    @abstractmethod
    def load(self):
        pass


class BaseResult(ABC):

    def __init__(self, predictions_path, fitted_strategies_path=None):
        # assigned in construction
        self._predictions_path = predictions_path
        self._fitted_strategies_path = predictions_path if fitted_strategies_path is None else fitted_strategies_path

        # assigned during fitting of orchestration
        self.strategy_names = []
        self.dataset_names = []

    @property
    def fitted_strategies_path(self):
        return self._fitted_strategies_path

    @property
    def path(self):
        return self._predictions_path

    @abstractmethod
    def save_predictions(self, y_true, y_pred, y_proba, index, strategy_name=None, dataset_name=None,
                         train_or_test="test", cv_fold=0):
        pass

    @abstractmethod
    def load_predictions(self, train_or_test="test", fold=0):
        """Loads predictions for all datasets and strategies iteratively"""
        pass

    @abstractmethod
    def check_predictions_exist(self, strategy, dataset_name, cv_fold, train_or_test="test"):
        pass

    @abstractmethod
    def save_fitted_strategy(self, strategy, dataset_name, fold):
        pass

    @abstractmethod
    def load_fitted_strategy(self, strategy_name, dataset_name, fold):
        """Load fitted strategies for all datasets and strategies iteratively"""
        pass

    @abstractmethod
    def check_fitted_strategy_exists(self, strategy, dataset_name, cv_fold, train_or_test="test"):
        pass

    @abstractmethod
    def save(self):
        """Save results object as master file"""
        pass
