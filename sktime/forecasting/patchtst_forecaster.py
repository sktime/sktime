"""Adapter for using the huggingface PatchTST for forecasting."""

# documentation for PatchTST:
# https://huggingface.co/docs/transformers/main/en/model_doc/patchtst#transformers.PatchTSTConfig

import pandas as pd

# required imports for now
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    Trainer,
    TrainingArguments,
)

# another external library required
from tsfm_public.toolkit.dataset import ForecastDFDataset
from tsfm_public.toolkit.time_series_preprocessor import TimeSeriesPreprocessor
from tsfm_public.toolkit.util import select_by_index

# from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class HFPatchTSTForecaster:
    """docstring for PatchTST forecaster.

    link to ref: https://github.com/yuqinie98/PatchTST
    link to paper: https://arxiv.org/abs/2211.14730
    link to hf model card:
    https://huggingface.co/docs/transformers/main/en/model_doc/patchtst#transformers.PatchTSTConfig

    Parameters
    ----------
        params: params
    """

    def __init__(
        self,
        # model variables except for forecast_columns
        path,
        forecast_columns,
        pretrained=True,
        patch_length=16,
        context_length=512,
        patch_stride=16,
        random_mask_ratio=0.4,
        d_model=128,
        num_attention_heads=16,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.2,
        head_dropout=0.2,
        pooling_type=None,
        channel_attention=False,
        scaling="std",
        loss="mse",
        pre_norm=True,
        norm_type="batchnorm",
        # dataset and training config
        timestamp_column=None,  # [],
        id_columns=None,  # [],
        num_workers=16,
        batch_size=64,
        learning_rate=None,
        epochs=100,
        train_split_ratio=None,  # [0.7, 0.2, 0.1]
    ):
        self.pretrained = pretrained
        self.path = path
        self.forecast_columns = forecast_columns
        self.patch_length = patch_length
        self.context_length = context_length
        self.patch_stride = patch_stride
        self.random_mask_ratio = random_mask_ratio
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.pooling_type = pooling_type
        self.channel_attention = channel_attention
        self.scaling = scaling
        self.loss = loss
        self.pre_norm = pre_norm
        self.norm_type = norm_type

        # dataset and training config
        self.timestamp_column = timestamp_column
        self.id_columns = id_columns
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_split_ratio = train_split_ratio
        self.eval_on_test = True
        super().__init__()

        if self.pretrained:
            from transformers import PatchTSTForRegression

            model = PatchTSTForRegression.from_pretrained(
                "namctin/patchtst_etth1_regression"
            )
            self._fitted_model = model

    def _fit(self, X, fh, y):
        """Fits the model.

        Parameters
        ----------
        X : pandas DataFrame
            dataframe containing all the time series, univariate,
            multivariate acceptable

        y : pandas DataFrame, default = None
            pandas dataframe containing forecasted horizon to predict
            default None

        fh : Forecasting Horizon object
            used to determine forecasting horizon for predictions

        Returns
        -------
        self : object


        self : a reference to the object
        """
        # fh : pd.Index, pd.TimedeltaIndex, np.array, list, pd.Timedelta, or int
        # y = X
        X[self.timestamp_column] = pd.to_datetime(X[self.timestamp_column])
        num_train = int(len(X) * self.train_split_ratio[0])
        num_val = int(len(X) * self.train_split_ratio[2])
        num_test = int(len(X) * self.train_split_ratio[1])
        if not self.train_split_ratio:
            self.train_split_ratio = [0.7, 0.2, 0.1]

        border1s = [
            0,
            num_train - self.context_length,
            len(X) - num_test - self.context_length,
        ]
        border2s = [num_train, num_train + num_val, len(X)]

        train_start_index = border1s[0]  # None indicates beginning of dataset
        train_end_index = border2s[0]

        # we shift the start of the evaluation period back by context length so that
        # the first evaluation timestamp is immediately following the training data
        valid_start_index = border1s[1]
        valid_end_index = border2s[1]

        test_start_index = border1s[2]
        test_end_index = border2s[2]

        train_data = select_by_index(
            X,
            id_columns=self.id_columns,
            start_index=train_start_index,
            end_index=train_end_index,
        )
        valid_data = select_by_index(
            X,
            id_columns=self.id_columns,
            start_index=valid_start_index,
            end_index=valid_end_index,
        )
        test_data = select_by_index(
            X,
            id_columns=self.id_columns,
            start_index=test_start_index,
            end_index=test_end_index,
        )

        time_series_preprocessor = TimeSeriesPreprocessor(
            timestamp_column=self.timestamp_column,
            id_columns=self.id_columns,
            input_columns=self.forecast_columns,
            output_columns=self.forecast_columns,
            scaling=True,
        )
        self.time_series_preprocessor = time_series_preprocessor.train(train_data)

        # create the forecasting dataset object
        train_dataset = ForecastDFDataset(
            self.time_series_preprocessor.preprocess(train_data),
            id_columns=self.id_columns,
            timestamp_column="date",
            target_columns=self.forecast_columns,
            context_length=self.context_length,
            prediction_length=fh,
        )
        valid_dataset = ForecastDFDataset(
            self.time_series_preprocessor.preprocess(valid_data),
            id_columns=self.id_columns,
            timestamp_column="date",
            target_columns=self.forecast_columns,
            context_length=self.context_length,
            prediction_length=fh,
        )
        test_dataset = ForecastDFDataset(
            self.time_series_preprocessor.preprocess(test_data),
            id_columns=self.id_columns,
            timestamp_column="date",
            target_columns=self.forecast_columns,
            context_length=self.context_length,
            prediction_length=fh,
        )

        # initialize model
        config = PatchTSTConfig(
            num_input_channels=len(self.forecast_columns),
            context_length=self.context_length,
            patch_length=self.patch_length,
            patch_stride=self.patch_length,
            prediction_length=fh,
            random_mask_ratio=self.random_mask_ratio,
            d_model=self.d_model,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            head_dropout=self.head_dropout,
            pooling_type=self.pooling_type,
            channel_attention=self.channel_attention,
            scaling=self.scaling,
            loss=self.loss,
            pre_norm=self.pre_norm,
            norm_type=self.norm_type,
        )
        model = PatchTSTForPrediction(config)

        # initialize training_args
        training_args = TrainingArguments(
            output_dir=self.path,
            overwrite_output_dir=True,
            # learning_rate=0.001,
            num_train_epochs=self.epochs,
            do_eval=True,
            evaluation_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            dataloader_num_workers=self.num_workers,
            save_strategy="epoch",
            logging_strategy="epoch",
            save_total_limit=3,
            logging_dir=self.path,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,  # For loss
            label_names=["future_values"],
        )

        # Create the early stopping callback
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=10,
            early_stopping_threshold=0.0001,
        )

        # define trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            callbacks=[early_stopping_callback],
            # compute_metrics=compute_metrics,
        )
        # pretrain
        trainer.train()
        self._fitted_model = trainer.model

        if self.eval_on_test:
            testset_eval = trainer.evaluate(test_dataset)
            self.testset_eval_dict = testset_eval
            # forward eval metrics as a param attribute

        return self

    def _predict(self, X, fh):
        # fh : pd.Index, pd.TimedeltaIndex, np.array, list, pd.Timedelta, or int
        """Predicts the model.

        Parameters
        ----------
        X : pandas DataFrame
            dataframe containing all the time series, univariate,
            multivariate acceptable

        y : pandas DataFrame, default = None
            pandas dataframe containing forecasted horizon to predict
            default None

        fh : Forecasting Horizon object
            used to determine forecasting horizon for predictions
            expected to be the same as the one used in _fit

        Returns
        -------
        y_pred : predictions outputted from the fitted model
        """
        X[self.timestamp_column] = pd.to_datetime(X[self.timestamp_column])
        test_dataset = ForecastDFDataset(
            self.time_series_preprocessor.preprocess(X),
            id_columns=self.id_columns,
            timestamp_column="date",
            target_columns=self.forecast_columns,
            context_length=self.context_length,
            prediction_length=fh,
        )
        y_pred = self._fitted_model.predict(test_dataset)

        return y_pred
