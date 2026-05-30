import numpy as np
from sktime.regression.deep_learning.mcdcnn import MCDCNNRegressorTorch

class RecordingCallback:
    def __init__(self):
        self.on_train_start_called = False
        self.on_epoch_start_called = 0
        self.on_epoch_end_called = 0
        self.on_train_end_called = False

    def on_train_start(self, trainer):
        self.on_train_start_called = True

    def on_epoch_start(self, trainer):
        self.on_epoch_start_called += 1

    def on_epoch_end(self, trainer, logs=None):
        self.on_epoch_end_called += 1

    def on_train_end(self, trainer):
        self.on_train_end_called = True

def test_callbacks_are_called_regressor():
    np.random.seed(42)
    X_train = np.random.randn(10, 1, 20)  # 10 samples, 1 dimension, 20 timestamps
    y_train = np.random.randn(10)

    callback = RecordingCallback()
    model = MCDCNNRegressorTorch(
        n_epochs=3,
        batch_size=2,
        callbacks=[callback],
        verbose=False
    )
    model.fit(X_train, y_train)

    assert callback.on_train_start_called
    assert callback.on_epoch_start_called == 3
    assert callback.on_epoch_end_called == 3
    assert callback.on_train_end_called