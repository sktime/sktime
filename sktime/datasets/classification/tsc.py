"""Time series classification datasets."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sktime.datasets.base._base import TSDatasetLoader
from sktime.datasets.base._metadata import ExternalDatasetMetadata

DEFAULT_PATH = Path(__file__, "../data/")
CITATION = ""


class TSCDataset(TSDatasetLoader):
    """Base class for all sktime datasets."""

    def __init__(
        self,
        name,
        split: Optional[str] = None,
        save_dir: Optional[str] = None,
    ):
        metadata = ExternalDatasetMetadata(
            name=name,
            task_type="classification",
            url="https://timeseriesclassification.com/",
            backup_urls=["https://github.com/sktime/sktime-datasets/raw/main/TSC"],
            citation=CITATION,
        )

        save_dir = Path(DEFAULT_PATH, name) if save_dir is None else Path(save_dir)
        super().__init__(metadata, save_dir)
        self._split = split

    @classmethod
    def _load_train_test(self, split: str):
        file_path = Path(self._save_dir, f"_{split}.ts")
        return self._load_from_file(file_path)

    def _load(self):
        """Load the dataset into memory."""
        X_train, y_train = self._load_train_test("TRAIN")
        if self._split == "TRAIN":
            return X_train, y_train
        X_test, y_test = self._load_train_test("TEST")
        if self._split == "TEST":
            return X_test, y_test

        return self._preprocess(X_train, y_train, X_test, y_test)

    def _preprocess(self, X_train, y_train, X_test, y_test):
        """Preprocess the dataset."""
        X = pd.concat([X_train, X_test])
        X = X.reset_index(drop=True)
        y = np.concatenate([y_train, y_test])
        return X, y
