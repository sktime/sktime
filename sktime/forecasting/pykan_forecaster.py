# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Dummy forecasters."""

__author__ = ["fkiraly"]

import pandas as pd

from sktime.datatypes import convert_to
from sktime.forecasting.base import BaseForecaster
from sktime.split import temporal_train_test_split

from kan import *

class PyKANForecaster(BaseForecaster):
    """
    """

    _tags = {
        # TODO
    }

    def __init__(self, ):
        self.hidden_layers =  (5,5)
        self.input_layer_size = 12
        super().__init__()

    def _fit(self, y, X, fh):
        """
        """
        output_size = max(fh._values)
        y_train, y_test = temporal_train_test_split(y, test_size=36)
        ds = PyTorchTrainDataset(y_train, self.input_layer_size, output_size)
        train_y = [ds[i][0] for i in range(len(ds))]
        train_target = [ds[i][1] for i in range(len(ds))]

        ds = PyTorchTrainDataset(y_test, self.input_layer_size, output_size)
        test_y = [ds[i][0] for i in range(len(ds))]
        test_target = [ds[i][1] for i in range(len(ds))]
        # no fitting, we already know the forecast values



        ds_new = {
            "train_input":torch.stack(train_y),
            "train_label":torch.stack(train_target),
            "test_input":torch.stack(test_y),
            "test_label":torch.stack(test_target)
        }


        grids = np.array([5,10,20,50,100])

        train_losses = []
        test_losses = []
        steps = 20
        k = 3

        for i in range(grids.shape[0]):
            if i == 0:
                self.model = KAN(width=[self.input_layer_size,*self.hidden_layers,output_size], grid=grids[i], k=k)
            if i != 0:
                self.model = KAN(width=[self.input_layer_size,*self.hidden_layers,output_size], grid=grids[i], k=k).initialize_from_another_model(self.model, ds_new['train_input'])
            results = self.model.train(ds_new, opt="LBFGS", steps=steps, stop_grid_update_step=30)
            train_losses += results['train_loss']
            test_losses += results['test_loss']

        return self

    def _predict(self, fh, X):
        """
        """
        input_ = torch.from_numpy(self._y.values[-self.input_layer_size:])
        return self.model(torch.stack(input_)).detach().numpy()

  

from torch.utils.data import Dataset
class PyTorchTrainDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(self, y, seq_len, fh=None, X=None):
        self.y = y.values
        self.X = X.values if X is not None else X
        self.seq_len = seq_len
        self.fh = fh

    def __len__(self):
        """Return length of dataset."""
        return max(len(self.y) - self.seq_len - self.fh + 1, 0)

    def __getitem__(self, i):
        """Return data point."""
        from torch import from_numpy, tensor
        hist_y = tensor(self.y[i : i + self.seq_len]).float()
        if self.X is not None:
            exog_data = tensor(
                self.X[i + self.seq_len : i + self.seq_len + self.fh]
            ).float()
        else:
            exog_data = tensor([])
        return (
            torch.cat([hist_y, exog_data]),
            from_numpy(self.y[i + self.seq_len : i + self.seq_len + self.fh]).float(),
        )