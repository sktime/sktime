import os
import random
from argparse import Namespace
from typing import NamedTuple

import numpy as np
import torch


class NamespaceWithDefaults(Namespace):
    @classmethod
    def from_namespace(cls, namespace):
        new_instance = cls()
        for attr in dir(namespace):
            if not attr.startswith("__"):
                setattr(new_instance, attr, getattr(namespace, attr))
        return new_instance

    def getattr(self, key, default=None):
        return getattr(self, key, default)


def parse_config(config: dict) -> NamespaceWithDefaults:
    args = NamespaceWithDefaults(**config)
    return args


def make_dir_if_not_exists(path, verbose=True):
    if not is_directory(path):
        path = path.split(".")[0]
    if not os.path.exists(path=path):
        os.makedirs(path)
        if verbose:
            print(f"Making directory: {path}...")
    return True


def is_directory(path):
    extensions = [".pth", ".txt", ".json", ".yaml"]

    for ext in extensions:
        if ext in path:
            return False
    return True


def control_randomness(seed: int = 13):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dtype_map(dtype: str):
    map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    return map[dtype]


def get_huggingface_model_dimensions(model_name: str = "flan-t5-base"):
    from transformers import T5Config

    config = T5Config.from_pretrained(model_name)
    return config.d_model


def get_anomaly_criterion(anomaly_criterion: str = "mse"):
    if anomaly_criterion == "mse":
        return torch.nn.MSELoss(reduction="none")
    elif anomaly_criterion == "mae":
        return torch.nn.L1Loss(reduction="none")
    else:
        raise ValueError(f"Anomaly criterion {anomaly_criterion} not supported.")


def _reduce(metric, reduction="mean", axis=None):
    if reduction == "mean":
        return np.nanmean(metric, axis=axis)
    elif reduction == "sum":
        return np.nansum(metric, axis=axis)
    elif reduction == "none":
        return metric


class EarlyStopping:
    def __init__(self, patience: int = 3, verbose: bool = False, delta: float = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, validation_loss):
        score = -validation_loss
        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
