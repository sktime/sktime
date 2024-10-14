
from sktime.forecasting.base._base import _BaseGlobalForecaster, ForecastingHorizon
import abc
import torch

class BasePeftable(_BaseGlobalForecaster, abc.ABC):
    """Base global forecaster template class.

    This class is a temporal solution, might be merged into BaseForecaster later.

    The base forecaster specifies the methods and method signatures that all
    global forecasters have to implement.

    Specific implementations of these methods is deferred to concrete forecasters.

    """

    @abc.abstractmethod
    def get_model(self, fh: ForecastingHorizon):
        """Return the underlying model.
        ForecastingHorizon is required to initialize models sometimes correctly.

        """
        raise NotImplementedError()
    
    @abc.abstractmethod
    def set_model(self, model):
        """Return the parameters of the underlying model."""
        raise NotImplementedError()

    def get_train_dataset(self, y, X, fh) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Return the training dataset."""
        raise NotImplementedError()
