from sktime.forecasting.base import BaseForecaster


class BaseFoundationForecaster(BaseForecaster):
    """
    Base class for foundation model forecasters in sktime.

    Unified interface for:
    - loading pre-trained models
    - preprocessing time series data
    - performing inference
    - postprocessing predictions

    Supports:
    - zero-shot inference
    - fine-tuning workflows
    - extensible model registry
    """

    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device

        self.model = None
        self.tokenizer = None

        super().__init__()

    def load_model(self):
        raise NotImplementedError()

    def preprocess(self, y):
        raise NotImplementedError()

    def postprocess(self, preds):
        raise NotImplementedError()

    def _fit(self, y, X=None, fh=None):
        self.load_model()
        self._y = y
        return self

    def _predict(self, fh, X=None):
        self.fh_len = len(fh)

        inputs = self.preprocess(self._y)
        raw_preds = self._infer(inputs, fh)

        return self.postprocess(raw_preds)

    def _infer(self, inputs, fh):
        """
        Run model inference.

        Parameters:
        - inputs: preprocessed input data
        - fh: forecasting horizon

        Returns:
        - raw predictions
        """
        raise NotImplementedError()

    def save(self, path):
        self.model.save_pretrained(path)

    def load(self, path):
        raise NotImplementedError()

    def fine_tune(self, dataset):
        raise NotImplementedError()