from abc import ABC, abstractmethod
from typing import Tuple

import pandas as pd

from sktime.datasets._data_io import load_from_tsfile  # _alias_mtype_check
from sktime.datasets.base._metadata import BaseDatasetMetadata


class BaseDataset(ABC):
    """Base class for all sktime datasets."""

    def __init__(
        self,
        metadata: BaseDatasetMetadata,
        save_dir: str,
        return_data_type: str = "pd.DataFrame",
    ) -> None:
        super().__init__()
        self._metadata = metadata
        self._save_dir = save_dir
        self._return_data_type = return_data_type

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the dataset."""
        return self._load()

    @abstractmethod
    def _load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the dataset."""
        raise NotImplementedError()

    def download(self) -> None:
        """Download the dataset."""
        raise NotImplementedError()

    def _extract_from_zipfile(self) -> None:
        """Extract the dataset from a zip file."""
        raise NotImplementedError()

    def _fallback_download(self) -> None:
        """Download the dataset from a fallback URL."""
        raise NotImplementedError()

    @property
    def is_dataset_exits(self) -> bool:
        """Check if the dataset exists."""
        raise NotImplementedError()


class TSDatasetLoader(BaseDataset):
    """Base class for .ts fromat datasets."""

    def __init__(self, metadata, save_dir, return_data_type):
        super().__init__(metadata, save_dir, return_data_type)

    def _load_from_file(self, file_path):
        """Load .ts format dataset."""
        X, y = load_from_tsfile(
            full_file_path_and_name=file_path, return_data_type=self._return_data_type
        )
        return X, y

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        """Preprocess the dataset."""
        pass


class CSVDatasetLoader(BaseDataset):
    """Base class for .csv format datasets."""

    def __init__(self, metadata, save_dir):
        super().__init__(metadata, save_dir)

    def _load(self):
        """Load the dataset."""
        dataset = pd.read_csv(self._save_dir)
        return self._preprocess(dataset)

    @abstractmethod
    def _preprocess(self, dataset):
        """Preprocess the dataset."""
        pass
