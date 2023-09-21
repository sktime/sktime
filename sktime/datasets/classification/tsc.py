"""Time series classification datasets."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sktime.datasets._data_io import load_from_tsfile
from sktime.datasets.base._base import BaseDataset
from sktime.datasets.base._metadata import ExternalDatasetMetadata

DEFAULT_PATH = Path.cwd().parent / "data"
CITATION = ""


class TSCDatasetLoader(BaseDataset):
    """Classification dataset from UCR UEA time series archive."""

    def __init__(
        self,
        name: str,
        split: Optional[str] = None,
        save_dir: Optional[str] = None,
        return_data_type: str = "nested_univ",
    ):
        metadata = ExternalDatasetMetadata(
            name=name,
            task_type="classification",
            url=[
                "https://timeseriesclassification.com/aeon-toolkit",
                "https://github.com/sktime/sktime-datasets/raw/main/TSC",
            ],
            download_file_format="zip",
            citation=CITATION,
        )
        if save_dir is None:
            save_dir = Path(DEFAULT_PATH, name)
        else:
            save_dir = Path(save_dir, name)
        super().__init__(metadata, save_dir, return_data_type)
        self._split = split.upper() if split is not None else split

    def _load_from_file(self, split: str):
        """Load .ts format dataset."""
        file_path = Path(self.save_dir, f"{self._metadata.name}_{split}.ts")
        X, y = load_from_tsfile(
            full_file_path_and_name=file_path, return_data_type=self._return_data_type
        )
        return X, y

    def _preprocess(self, X_train, y_train, X_test, y_test):
        """Preprocess the dataset."""
        X = pd.concat([X_train, X_test])
        X = X.reset_index(drop=True)
        y = np.concatenate([y_train, y_test])
        return X, y

    def _load(self):
        """Load the dataset into memory."""
        X_train, y_train = self._load_train_test("TRAIN")
        if self._split == "TRAIN":
            return X_train, y_train
        X_test, y_test = self._load_train_test("TEST")
        if self._split == "TEST":
            return X_test, y_test

        return self._preprocess(X_train, y_train, X_test, y_test)
