from sktime.forecasting.base import BaseForecaster
from transformers import AutoformerForPrediction
import torch
from torch.utils.data import DataLoader, Dataset
from skpro.distributions import Normal, TDistribution

class HuggingfaceTransformerForecaster(BaseForecaster):
    def __init__(self, model_path : str, 
                 config=None, 
                 fit_strategy="minimal",
                 patience=5,
                 delta=0.0001,
                 validation_split=0.2,
                 batch_size=32,
                 epochs=10,
                 verbose=True):
        super().__init__()
        self.model_path = model_path
        self.config = config if config is not None else {}
        self.fit_strategy = fit_strategy
        self.patience = patience
        self.delta = delta
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose


    def _fit(self, y, X, fh):

        # Load model and extract config
        config = AutoformerForPrediction.from_pretrained(self.model_path).config
        
        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self.config)
        _config["num_dynamic_real_features"] = X.shape[-1]
        config = config.from_dict(_config)
    
        # Load model with the updated config
        self.model, info = AutoformerForPrediction.from_pretrained(self.model_path, 
                                                config=config,
                                                output_loading_info=True,
                                                ignore_mismatched_sizes=True)

        # Freeze all loaded parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Clamp all loaded parameters to avoid NaNs due to large values
        for param in self.model.model.parameters():
            param.clamp_(-1000, 1000)

        # Reininit the weights of all layers that have mismatched sizes
        for key, _, _ in info["mismatched_keys"]:
            _model = self.model
            for attr_name in key.split(".")[:-1]:
                _model = getattr(_model, attr_name)
            _model.weight = torch.nn.Parameter(_model.weight.masked_fill(_model.weight.isnan(), 0.001), requires_grad=True)

        split = int(len(y) * (1 - self.validation_split))

        dataset = PyTorchDataset(y[:split], config.context_length+ max(config.lags_sequence), X=X[:split], fh=config.prediction_length)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = PyTorchDataset(y[split:], config.context_length+ max(config.lags_sequence), X=X[split:], fh=config.prediction_length)
        val_data_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        self.model.model.train()
        
        early_stopper = EarlyStopper(patience=self.patience, min_delta=self.delta)
        self.optim = torch.optim.Adam(self.model.parameters())

        if self.fit_strategy == "minimal":
            if len(info["mismatched_keys"]) == 0:
                return # No need to fit
            val_loss = float('inf')
            for epoch in range(self.epochs):
                if not early_stopper.early_stop(val_loss, self.model):
                    val_loss = self._run_epoch(data_loader, val_data_loader, epoch)
        elif self.fit_strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
            val_loss = float('inf')
            for epoch in self.epochs:
                if not early_stopper.early_stop(val_loss, self.model):
                    val_loss = self._run_epoch(data_loader, val_data_loader, epoch)
        else:
            raise Exception("Unknown fit strategy")

        self.model = early_stopper._best_model
        

    
    def _run_epoch(self, data_loader, val_data_loader, epoch):
        epoch_loss = 0
        for i, _input in enumerate(data_loader):
            (hist, hist_x, x_, y_) = _input
            pred = self.model(past_values=hist,
                        past_time_features=hist_x,
                        future_time_features=x_,
                        past_observed_mask=None,
                        future_values=y_)
            self.optim.zero_grad()
            pred.loss.backward()
            self.optim.step()
            if i % 100 == 0:
                hist, hist_x, x_, y_ = next(iter(val_data_loader))
                val_pred = self.model(past_values=hist,
                        past_time_features=hist_x,
                        future_time_features=x_,
                        past_observed_mask=None,
                        future_values=y_)
                epoch_loss = val_pred.loss.detach().numpy()
                if self.verbose:
                    print(epoch, i, pred.loss.detach().numpy(), val_pred.loss.detach().numpy())

        return epoch_loss

    def _predict(self, fh, X=None):

        hist = self.y_.values.reshape((1,-1))
        hist_x = self.X_.values.reshape((1, -1, self.X_.shape[-1]))
        x_ = X.values.reshape((1, -1, self.X_.shape[-1]))
        pred = self.model.generate(past_values=hist,
                past_time_features=hist_x,
                future_time_features=x_,
                past_observed_mask=None,)
        
        return pred.mean(dim=1).detach().numpy()

    def _predict_proba(self, fh, X=None):
        hist = self.y_.values.reshape((1,-1))
        hist_x = self.X_.values.reshape((1, -1, self.X_.shape[-1]))
        x_ = X.values.reshape((1, -1, self.X_.shape[-1]))
        pred = self.model(past_values=hist,
                past_time_features=hist_x,
                future_time_features=x_,
                past_observed_mask=None,)
        
        if self.model.config.distribution == "normal":
            return Normal(pred.params[0].detach().numpy(), pred.params[1].detach().numpy())
        elif self.model.config.distribution == "student_t":
            return TDistribution(pred.param[s0].detach().numpy(), 
                                 pred.params[1].detach().numpy(),
                                 pred.params[2].detach().numpy())
        elif self.model.config.distribution == "negative_binomial":
            raise Exception("Not implemented yet")
        else:
            raise Exception("Unknown distribution")



class PyTorchDataset(Dataset):
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
            hist_exog = tensor(self.X[i : i + self.seq_len]).float()
        else:
            exog_data = tensor([])
            hist_exog = tensor([])
        return (
            hist_y, 
            hist_exog,
            exog_data,
            from_numpy(self.y[i + self.seq_len : i + self.seq_len + self.fh]).float(),
        )
    
from copy import deepcopy

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            self._best_model = deepcopy(model)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False