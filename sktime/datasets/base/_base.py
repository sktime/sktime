import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve

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

    @property
    def get_save_dir(self):
        """Return the save directory."""
        return self._save_dir

    @abstractmethod
    def _load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the dataset."""
        raise NotImplementedError()

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the dataset."""
        # check directory
        # download if not exists
        self.download()
        return self._load()

    def download(self) -> None:
        """Download the dataset."""
        url = f"{self._metadata.url}/{self._metadata.name}.{self._metadata.download_format}"  # noqa
        # TODO: cahce logic here
        if not self._save_dir.exists():
            self._save_dir.mkdir(parents=True, exist_ok=True)
            zip_file, temp_dir = self._download(url)
            if self._metadata.download_file_format == "zip":
                self._extract_zipfile(zip_file, temp_dir)

    def _download(self, url: str) -> None:
        """Download zip file to a temp directory and extract it."""
        temp_dir = tempfile.mkdtemp()  # create a temp directory
        zip_file_save_to = Path(temp_dir, self._metadata.name)
        urlretrieve(url, zip_file_save_to)
        return zip_file_save_to, temp_dir

    def _extract_zipfile(self, zip_file, temp_dir) -> None:
        try:
            zipfile.ZipFile(zip_file, "r").extractall(self._save_dir)
            shutil.rmtree(temp_dir)  # delete temp directory with all its contents
        except zipfile.BadZipFile:
            shutil.rmtree(temp_dir)
            self._fallback_download()
            raise zipfile.BadZipFile(
                "Could not unzip dataset. Please make sure the URL is valid."
            )

    def _fallback_download(self) -> None:
        """Download the dataset from a fallback URL."""
        for url in self._metadata.backup_urls:
            try:
                self._download_extract(url)
                return
            except zipfile.BadZipFile:
                pass
        raise zipfile.BadZipFile(
            "Could not unzip dataset. Please make sure the URL is valid."
        )

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
