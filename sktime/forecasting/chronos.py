"""Implements Chronos forecaster.

References
----------
.. [1] https://github.com/amazon-science/chronos-forecasting
"""

__author__ = ["Z-Fran"]
__all__ = ["ChronosForecaster"]

import itertools
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.hf_transformers_forecaster import HFTransformersForecaster

if _check_soft_dependencies("torch", severity="none"):
    import torch
    from torch.utils.data import IterableDataset, get_worker_info
else:

    class IterableDataset:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies("transformers", severity="none"):
    import transformers
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        GenerationConfig,
        T5Config,
        Trainer,
        TrainingArguments,
    )

if _check_soft_dependencies("gluonts", severity="none"):
    from gluonts.dataset.common import ListDataset
    from gluonts.itertools import Cyclic, Filter, Map
    from gluonts.transform import (
        ExpectedNumInstanceSampler,
        FilterTransformation,
        InstanceSplitter,
        LastValueImputation,
        LeavesMissingValues,
        MissingValueImputation,
        TestSplitSampler,
        ValidationSplitSampler,
    )
else:

    class MissingValueImputation:
        """Dummy class if gluonts is unavailable."""


def left_pad_and_stack_1D(tensors: list):
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(
            size=(max_len - len(c),), fill_value=torch.nan, device=c.device
        )
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
):
    """
    Get the next available path in a directory.

    For example, if `base_fname="results"` and `base_dir` has files
    ["results-0.yaml", "results-1.yaml"], this function returns "results-2.yaml".
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    next_num = max(run_nums) + 1
    fname = f"{base_fname}{separator}{next_num}" + (
        f".{file_type}" if file_type != "" else ""
    )

    return base_dir / fname


def has_enough_observations(
    entry: dict, min_length: int = 0, max_missing_prop: float = 1.0
) -> bool:
    """
    Check if the given entry has enough observations in the ``"target"`` attribute.

    Parameters
    ----------
    entry
        The data entry (dictionary) to be tested.
    min_length
        The minimum length the ``"target"`` attribute must have.
    max_missing_prop
        The maximum proportion of missing data allowed in the ``"target"``
        attribute.
    """
    if (
        len(entry["target"]) >= min_length
        and np.isnan(entry["target"]).mean() <= max_missing_prop
    ):
        return True
    return False


def load_model(
    model_id="google/t5-efficient-tiny",
    model_type="seq2seq",
    vocab_size=4096,
    random_init=False,
    tie_embeddings=False,
    pad_token_id=0,
    eos_token_id=1,
):
    """
    Load the model.

    Load the specified HuggingFace model, adjusting the vocabulary size,
    special token IDs, and initialization options.

    This allows to set a model up for training on a new vocabulary
    of tokens.
    """
    assert model_type in ["seq2seq", "causal"]
    AutoModelClass = (
        AutoModelForSeq2SeqLM if model_type == "seq2seq" else AutoModelForCausalLM
    )
    if random_init:
        config = AutoConfig.from_pretrained(model_id)
        if isinstance(config, T5Config):
            # The default initializer_factor (1.0) in transformers is too large
            config.initializer_factor = 0.05
        config.tie_word_embeddings = tie_embeddings
        model = AutoModelClass.from_config(config)
    else:
        model = AutoModelClass.from_pretrained(model_id)

    model.resize_token_embeddings(vocab_size)

    model.config.pad_token_id = model.generation_config.pad_token_id = pad_token_id
    model.config.eos_token_id = model.generation_config.eos_token_id = eos_token_id

    return model


@dataclass
class ChronosConfig:
    """Holds all the chronos configuration parameters."""

    tokenizer_class: str
    tokenizer_kwargs: dict[str, Any]
    context_length: int
    prediction_length: int
    n_tokens: int
    n_special_tokens: int
    pad_token_id: int
    eos_token_id: int
    use_eos_token: bool
    model_type: Literal["causal", "seq2seq"]
    num_samples: int
    temperature: float
    top_k: int
    top_p: float

    def __post_init__(self):
        assert (
            self.pad_token_id < self.n_special_tokens
            and self.eos_token_id < self.n_special_tokens
        ), f"Special token id's must be smaller than {self.n_special_tokens=}"

    def create_tokenizer(self) -> "ChronosTokenizer":
        class_ = eval(self.tokenizer_class)
        return class_(**self.tokenizer_kwargs, config=self)


class ChronosTokenizer:
    """
    Definines how time series are mapped into token IDs and back.

    For details, see the ``input_transform`` and ``output_transform`` methods,
    which concrete classes must implement.
    """

    def context_input_transform(self, context):
        """
        Turn a batch of time series into token IDs, attention map, and tokenizer_state.

        Parameters
        ----------
        context
            A tensor shaped (batch_size, time_length), containing the
            timeseries to forecast. Use left-padding with ``torch.nan``
            to align time series of different lengths.

        Returns
        -------
        token_ids
            A tensor of integers, shaped (batch_size, time_length + 1)
            if ``config.use_eos_token`` and (batch_size, time_length)
            otherwise, containing token IDs for the input series.
        attention_mask
            A boolean tensor, same shape as ``token_ids``, indicating
            which input observations are not ``torch.nan`` (i.e. not
            missing nor padding).
        tokenizer_state
            An object that can be passed to ``label_input_transform``
            and ``output_transform``. Contains the relevant information
            to decode output samples into real values,
            such as location and scale parameters.
        """
        raise NotImplementedError()

    def label_input_transform(self, label, tokenizer_state):
        """
        Turn a batch of label slices of time series into token IDs and attention map.

        Parameters
        ----------
        label
            A tensor shaped (batch_size, time_length), containing the
            timeseries to forecast. Use left-padding with ``torch.nan``
            to align time series of different lengths.
        tokenizer_state
            An object returned by ``context_input_transform`` containing
            relevant information to preprocess data, such as location and
            scale. The nature of this depends on the specific tokenizer.
            This is used for tokenizing the label, in order to use the same
            scaling used to tokenize the context.

        Returns
        -------
        token_ids
            A tensor of integers, shaped (batch_size, time_length + 1)
            if ``config.use_eos_token`` and (batch_size, time_length)
            otherwise, containing token IDs for the input series.
        attention_mask
            A boolean tensor, same shape as ``token_ids``, indicating
            which input observations are not ``torch.nan`` (i.e. not
            missing nor padding).
        """
        raise NotImplementedError()

    def output_transform(self, samples, tokenizer_state):
        """
        Turn a batch of sample token IDs into real values.

        Parameters
        ----------
        samples
            A tensor of integers, shaped (batch_size, num_samples, time_length),
            containing token IDs of sample trajectories.
        tokenizer_state
            An object returned by ``input_transform`` containing
            relevant context to decode samples, such as location and scale.
            The nature of this depends on the specific tokenizer.

        Returns
        -------
        forecasts
            A real tensor, shaped (batch_size, num_samples, time_length),
            containing forecasted sample paths.
        """
        raise NotImplementedError()


class MeanScaleUniformBins(ChronosTokenizer):
    def __init__(
        self, low_limit: float, high_limit: float, config: ChronosConfig
    ) -> None:
        self.config = config
        self.centers = torch.linspace(
            low_limit,
            high_limit,
            config.n_tokens - config.n_special_tokens - 1,
        )
        self.boundaries = torch.concat(
            (
                torch.tensor([-1e20], device=self.centers.device),
                (self.centers[1:] + self.centers[:-1]) / 2,
                torch.tensor([1e20], device=self.centers.device),
            )
        )

    def _input_transform(self, context, scale=None):
        attention_mask = ~torch.isnan(context)

        if scale is None:
            scale = torch.nansum(
                torch.abs(context) * attention_mask, dim=-1
            ) / torch.nansum(attention_mask, dim=-1)
            scale[~(scale > 0)] = 1.0

        scaled_context = context / scale.unsqueeze(dim=-1)
        token_ids = (
            torch.bucketize(
                input=scaled_context,
                boundaries=self.boundaries,
                # buckets are open to the right, see:
                # https://pytorch.org/docs/2.1/generated/torch.bucketize.html#torch-bucketize
                right=True,
            )
            + self.config.n_special_tokens
        )
        token_ids[~attention_mask] = self.config.pad_token_id

        return token_ids, attention_mask, scale

    def _append_eos_token(self, token_ids, attention_mask):
        batch_size = token_ids.shape[0]
        eos_tokens = torch.full((batch_size, 1), fill_value=self.config.eos_token_id)
        token_ids = torch.concat((token_ids, eos_tokens), dim=1)
        eos_mask = torch.full((batch_size, 1), fill_value=True)
        attention_mask = torch.concat((attention_mask, eos_mask), dim=1)

        return token_ids, attention_mask

    def context_input_transform(self, context):
        length = context.shape[-1]

        if length > self.config.context_length:
            context = context[..., -self.config.context_length :]

        token_ids, attention_mask, scale = self._input_transform(context=context)

        if self.config.use_eos_token and self.config.model_type == "seq2seq":
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask, scale

    def label_input_transform(self, label, scale):
        length = label.shape[-1]

        assert length == self.config.prediction_length
        token_ids, attention_mask, _ = self._input_transform(context=label, scale=scale)

        if self.config.use_eos_token:
            token_ids, attention_mask = self._append_eos_token(
                token_ids=token_ids, attention_mask=attention_mask
            )

        return token_ids, attention_mask

    def output_transform(self, samples, scale):
        scale_unsqueezed = scale.unsqueeze(-1).unsqueeze(-1)
        indices = torch.clamp(
            samples - self.config.n_special_tokens - 1,
            min=0,
            max=len(self.centers) - 1,
        )
        return self.centers[indices] * scale_unsqueezed


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them in buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(
                    len(shuffle_buffer), size=(), generator=self.generator
                )
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """Mix-in class that datasets can inherit from to get shuffling functionality."""

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class ChronosDataset(IterableDataset, ShuffleMixin):
    """
    ChronosDataset.

    Dataset wrapper, using a ``ChronosTokenizer`` to turn data from a time series into
    a HuggingFace-compatible set of ``input_ids``, ``attention_mask`` and ``labels``.
    Entries from the original datasets are assumed to have a ``"start"`` attribute
    (of type ``pd.Period``), and a ``"target"`` attribute (of type ``np.ndarray``).

    Parameters
    ----------
    datasets
        Datasets containing the original time series data.
    probabilities
        In training mode, data will be sampled from each of the original datasets
        with these probabilities.
    tokenizer
        Tokenizer to be used to turn sequences of real numbers into token IDs.
    context_length
        Samples context will be limited to this length.
    prediction_length
        Samples labels will be limited to this length.
    drop_prob
        In training mode, observations from a sample will be turned into ``np.nan``,
        i.e. turned into missing values, with this probability.
    min_past
        Data samples will be considered only if there's at least ``min_past``-many
        historical observations.
    mode
        One of ``"training"``, ``"validation"``, or ``"test"``.
    np_dtype
        Numpy float data type.
    """

    def __init__(
        self,
        datasets: list,
        probabilities: list[float],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        drop_prob: float = 0.2,
        min_past: Optional[int] = None,
        model_type: str = "seq2seq",
        imputation_method: Optional[MissingValueImputation] = None,
        mode: str = "training",
        np_dtype=np.float32,
    ) -> None:
        super().__init__()

        assert len(probabilities) == len(datasets)
        assert mode in ("training", "validation", "test")
        assert model_type in ("seq2seq", "causal")

        self.datasets = datasets
        self.probabilities = probabilities
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.drop_prob = drop_prob
        self.min_past = min_past or prediction_length
        self.model_type = model_type
        self.imputation_method = imputation_method or LeavesMissingValues()
        self.mode = mode
        self.np_dtype = np_dtype

    def preprocess_entry(self, entry: dict, mode: str) -> dict:
        entry = {f: entry[f] for f in ["start", "target"]}
        entry["target"] = np.asarray(entry["target"], dtype=self.np_dtype)
        assert entry["target"].ndim == 1, f"got {entry['target'].ndim=}, expected 1"

        if self.model_type == "causal":
            # Causal models do not play nice with missing values, so it is
            # recommended to use an imputation method, e.g., LastValueImputation
            entry["target"] = self.imputation_method(entry["target"])

        if mode == "training" and self.drop_prob > 0:
            target = entry["target"].copy()
            drop_p = np.random.uniform(low=0.0, high=self.drop_prob)
            mask = np.random.choice(
                [True, False], size=len(target), p=[drop_p, 1 - drop_p]
            )
            target[mask] = np.nan
            entry["target"] = target

        return entry

    def _create_instance_splitter(self, mode: str):
        assert mode in ["training", "test", "validation"]

        instance_sampler = {
            "training": ExpectedNumInstanceSampler(
                num_instances=1.0,
                min_instances=1,
                min_past=self.min_past,
                min_future=self.prediction_length,
            ),
            "test": TestSplitSampler(),
            "validation": ValidationSplitSampler(min_future=self.prediction_length),
        }[mode]

        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=self.context_length,
            future_length=self.prediction_length,
            dummy_value=np.nan,
        )

    def create_training_data(self, data):
        data = Cyclic(data)
        split_transform = self._create_instance_splitter(
            "training"
        ) + FilterTransformation(
            condition=lambda entry: (~np.isnan(entry["past_target"])).sum() > 0
        )
        data = split_transform.apply(data, is_train=True)
        return data

    def create_test_data(self, data):
        data = self._create_instance_splitter("test").apply(data, is_train=False)
        return data

    def create_validation_data(self, data):
        data = self._create_instance_splitter("validation").apply(data, is_train=False)
        return data

    def to_hf_format(self, entry: dict) -> dict:
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(
            past_target
        )
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        labels, labels_mask = self.tokenizer.label_input_transform(future_target, scale)
        labels[labels_mask == 0] = -100

        if self.model_type == "causal":
            # The InstanceSplitter pads time series on the left to be equal to the
            # context_length. However, certain models (e.g., GPT2) with absolute
            # position embeddings should not be trained with left padding.
            # The following piece of code moves padding from left to right.

            assert input_ids.shape[-1] == entry["past_is_pad"].shape[0]

            # Find the index where padding starts
            pad_start_idx = np.searchsorted(1 - entry["past_is_pad"], 1)
            padded_input_ids, obs_input_ids = torch.tensor_split(
                input_ids, [pad_start_idx], dim=-1
            )
            padded_attention_mask, obs_attention_mask = torch.tensor_split(
                attention_mask, [pad_start_idx], dim=-1
            )

            # Move padding to the right
            input_ids = torch.cat(
                [
                    obs_input_ids,
                    labels,
                    padded_input_ids,
                ],
                axis=-1,
            )
            attention_mask = torch.cat(
                [
                    obs_attention_mask,
                    labels_mask,
                    padded_attention_mask,
                ],
                axis=-1,
            )

            # labels for causal models are same as the input_ids.
            # Internally transformers shifts the labels by one during training.
            labels = input_ids.clone()
            input_ids[~attention_mask] = self.tokenizer.config.pad_token_id
            labels[~attention_mask] = -100

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "labels": labels.squeeze(0),
        }

    def __iter__(self) -> Iterator:
        preprocessed_datasets = [
            Map(
                partial(self.preprocess_entry, mode=self.mode),
                dataset,
            )
            for dataset in self.datasets
        ]

        if self.mode == "training":
            iterables = [
                self.create_training_data(dataset) for dataset in preprocessed_datasets
            ]
        elif self.mode == "test":
            iterables = [
                self.create_test_data(dataset) for dataset in preprocessed_datasets
            ]
        else:
            iterables = [
                self.create_validation_data(dataset)
                for dataset in preprocessed_datasets
            ]

        worker_info = get_worker_info()
        if worker_info is None:
            probs = list(self.probabilities)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iterables = list(itertools.islice(iterables, worker_id, None, num_workers))
            probs = list(
                itertools.islice(self.probabilities, worker_id, None, num_workers)
            )

        probs = [prob / sum(probs) for prob in probs]

        iterators = list(map(iter, iterables))
        if self.mode == "training":
            while True:
                idx = np.random.choice(range(len(iterators)), p=probs)
                try:
                    yield self.to_hf_format(next(iterators[idx]))
                except StopIteration:
                    probs[idx] = 0
                    if sum(probs) == 0:
                        return
                    probs = [prob / sum(probs) for prob in probs]
        else:
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format(entry)


class ChronosForecaster(HFTransformersForecaster):
    """Chronos forecaster.

    Parameters
    ----------
    model_path : str
        Path to the Chronos' huggingface model.
    fit_strategy : str, default="minimal"
        Strategy to use for fitting (fine-tuning) the model.
    validation_split : float, default=0.2
        Fraction of the data to use for validation
    config : dict, default={}
        Configuration to use for the model.
    training_args : dict, default={}
        Training arguments to use for the model.
    compute_metrics : list, default=None
        List of metrics to compute during training.
    deterministic : bool, default=False
        Whether the predictions should be deterministic or not.
    callbacks : list, default=[]
        List of callbacks to use during training.
    peft_config : peft.PeftConfig, default=None
        Configuration for Parameter-Efficient Fine-Tuning.
    random_init: bool, default=False
    tie_embeddings: bool, default=False
    min_past: int, default=64
    shuffle_buffer_length: int, default=100
    max_missing_prop: float, default=0.9
    seed: int, optional, default=None

    References
    ----------
    . [1] https://github.com/amazon-science/chronos-forecasting

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.chronos import ChronosForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> import torch # doctest: +SKIP
    >>> import torch._dynamo # doctest: +SKIP
    >>> torch._dynamo.config.suppress_errors = True # doctest: +SKIP
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> training_args={
    ...     "output_dir": './output',
    ...     'per_device_train_batch_size': 4,
    ...     'gradient_accumulation_steps': 1,
    ...     'max_steps': 100
    ... }
    >>> config={"prediction_length": 32}
    >>> forecaster = ChronosForecaster(
    ...        "amazon/chronos-t5-tiny",
    ...        training_args=training_args,
    ...        config=config,
    ...        tie_embeddings=True,
    ...        shuffle_buffer_length=100_000,
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y_train)  # doctest: +SKIP
    >>> y_pred = forecaster.predict(fh) # doctest: +SKIP
    """

    # tag values are "safe defaults" which can usually be left as-is
    _tags = {
        "python_dependencies": ["torch>=2.4", "gluonts", "transformers"],
        "y_inner_mtype": "pd.Series",
        "scitype:y": "univariate",
        "authors": ["Z-Fran"],
    }

    def __init__(
        self,
        model_path: str,
        fit_strategy="minimal",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        deterministic=False,
        callbacks=None,
        peft_config=None,
        random_init: bool = False,
        tie_embeddings: bool = False,
        min_past: int = 64,
        shuffle_buffer_length: int = 100,
        max_missing_prop: float = 0.9,
        seed: Optional[int] = None,
    ):
        super().__init__(
            model_path=model_path,
            fit_strategy=fit_strategy,
            validation_split=validation_split,
            config=config,
            training_args=training_args,
            compute_metrics=compute_metrics,
            deterministic=deterministic,
            callbacks=callbacks,
            peft_config=peft_config,
        )

        # set the seed
        if seed is None:
            self.seed = np.random.randint(0, 2**31)
        else:
            self.seed = seed

        # set traing_args
        _training_args = {
            "output_dir": "./output",
            "per_device_train_batch_size": 32,  # int
            "learning_rate": 1e-3,  # float
            "lr_scheduler_type": "linear",  # str
            "warmup_ratio": 0.0,  # float
            "optim": "adamw_torch_fused",  # str
            "logging_strategy": "steps",  # str
            "logging_steps": 500,  # int
            "save_strategy": "steps",  # str
            "save_steps": 50_000,  # int
            "report_to": ["tensorboard"],  # list
            "max_steps": 200_000,  # int
            "gradient_accumulation_steps": 2,  # int
            "dataloader_num_workers": 1,  # int
            "tf32": True,  # bool, remove this if not using Ampere GPUs (e.g., A100)
            "torch_compile": True,  # bool
            "ddp_find_unused_parameters": False,  # bool
            "remove_unused_columns": False,  # bool
        }
        if training_args is not None:
            _training_args.update(training_args)
        self.training_args = _training_args

        # set config
        _config = {
            "model_type": "seq2seq",  # str
            "tokenizer_class": "MeanScaleUniformBins",  # str
            "tokenizer_kwargs": {"low_limit": -15.0, "high_limit": 15.0},  # dict
            "n_tokens": 4096,  # int
            "n_special_tokens": 2,  # int
            "pad_token_id": 0,  # int,
            "eos_token_id": 1,  # int,
            "use_eos_token": True,  # bool
            "context_length": 512,  # int
            "prediction_length": 64,  # int
            "num_samples": 20,  # int
            "temperature": 1.0,  # float
            "top_k": 50,  # int
            "top_p": 1.0,  # float
        }
        if config is not None:
            _config.update(config)
        self.config = _config
        self.chronos_config = ChronosConfig(**_config)

        # initialize attributes
        self.random_init = random_init
        self.tie_embeddings = tie_embeddings
        self.min_past = min_past
        self.shuffle_buffer_length = shuffle_buffer_length
        self.max_missing_prop = max_missing_prop

        # load the model
        self.model = load_model(
            model_id=self.model_path,
            model_type=self.chronos_config.model_type,
            vocab_size=self.chronos_config.n_tokens,
            random_init=self.random_init,
            tie_embeddings=self.tie_embeddings,
            pad_token_id=self.chronos_config.pad_token_id,
            eos_token_id=self.chronos_config.eos_token_id,
        )
        self.model.config.chronos_config = self.chronos_config.__dict__

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        self : reference to self
        """
        # set random seed
        transformers.set_seed(seed=self.seed)

        # check if TF32 format is available
        training_args = self.training_args
        if training_args["tf32"] and not (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        ):
            # TF32 floating point format is available only on NVIDIA GPUs
            # with compute capability 8 and above. See link for details.
            # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
            training_args["tf32"] = False

        # load data
        dataset_item = {"start": y.index[0], "target": list(y)}

        train_datasets = [
            Filter(
                partial(
                    has_enough_observations,
                    min_length=self.min_past + self.chronos_config.prediction_length,
                    max_missing_prop=self.max_missing_prop,
                ),
                ListDataset([dataset_item], freq=y.index.freqstr),
            )
        ]

        shuffled_train_dataset = ChronosDataset(
            datasets=train_datasets,
            probabilities=[1],
            tokenizer=self.chronos_config.create_tokenizer(),
            context_length=self.chronos_config.context_length,
            prediction_length=self.chronos_config.prediction_length,
            min_past=self.min_past,
            model_type=self.chronos_config.model_type,
            imputation_method=LastValueImputation()
            if self.chronos_config.model_type == "causal"
            else None,
            mode="training",
        ).shuffle(shuffle_buffer_length=self.shuffle_buffer_length)

        # define training args
        training_args = TrainingArguments(**training_args)

        # create Trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=shuffled_train_dataset,
        )

        trainer.train()

        return self

    def _prepare_and_validate_context(self, context):
        if isinstance(context, list):
            context = left_pad_and_stack_1D(context)
        assert isinstance(context, torch.Tensor)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        assert context.ndim == 2

        return context

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored.

        Returns
        -------
        y_pred : pd.DataFrame
            Predicted forecasts.
        """
        transformers.set_seed(self.seed)
        self.model.eval()
        context = torch.tensor(self._y.values)
        context_tensor = self._prepare_and_validate_context(context=context)

        prediction_length = len(fh)
        if prediction_length is None:
            prediction_length = self.chronos_config.prediction_length

        predictions = []
        remaining = prediction_length
        tokenizer = self.chronos_config.create_tokenizer()
        while remaining > 0:
            token_ids, attention_mask, scale = tokenizer.context_input_transform(
                context_tensor
            )
            prediction_length = min(remaining, self.chronos_config.prediction_length)
            preds = self.model.generate(
                input_ids=token_ids.to(self.model.device),
                attention_mask=attention_mask.to(self.model.device),
                generation_config=GenerationConfig(
                    min_new_tokens=prediction_length,
                    max_new_tokens=prediction_length,
                    do_sample=True,
                    num_return_sequences=self.chronos_config.num_samples,
                    eos_token_id=self.chronos_config.eos_token_id,
                    pad_token_id=self.chronos_config.pad_token_id,
                    temperature=self.chronos_config.temperature,
                    top_k=self.chronos_config.top_k,
                    top_p=self.chronos_config.top_p,
                ),
            )

            if self.chronos_config.model_type == "seq2seq":
                preds = preds[..., 1:]  # remove the decoder start token
            else:
                assert self.chronos_config.model_type == "causal"
                assert preds.size(-1) == token_ids.size(-1) + prediction_length
                preds = preds[..., -prediction_length:]

            preds = preds.reshape(
                token_ids.size(0), self.chronos_config.num_samples, -1
            )

            prediction = tokenizer.output_transform(preds.to(scale.device), scale)

            predictions.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            context_tensor = torch.cat(
                [context_tensor, prediction.median(dim=1).values], dim=-1
            )

        forecast_result = torch.cat(predictions, dim=-1)

        values = np.median(forecast_result[0].numpy(), axis=0)
        row_idx = self.fh.to_absolute_index(self.cutoff)
        y_pred = pd.Series(values, index=row_idx, name=self._y.name)

        return y_pred

    def save_model(self, path: str):
        """Save the model.

        Parameters
        ----------
        path : str
            The path to save the model.
        """
        self.model.save_pretrained(path)

    def __repr__(self):
        """Repr dunder."""
        class_name = self.__class__.__name__
        return f"{class_name}"

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params = []

        params.append(
            {
                "model_path": "amazon/chronos-t5-tiny",
                "training_args": {
                    "per_device_train_batch_size": 4,
                    "gradient_accumulation_steps": 1,
                    "max_steps": 100,
                },
                "config": {
                    "prediction_length": 32,
                },
                "shuffle_buffer_length": 100_000,
            }
        )

        # params.append(
        #     {
        #         "model_path": "amazon/chronos-t5-tiny",
        #         "training_args": {
        #             "per_device_train_batch_size": 4,
        #             "gradient_accumulation_steps": 1,
        #             "max_steps": 100,
        #         },
        #         "config": {
        #             "prediction_length": 16,
        #         },
        #         "tie_embeddings": True,
        #         "random_init": True,
        #         "shuffle_buffer_length": 50_000,
        #         "seed": 42,
        #     }
        # )

        return params
