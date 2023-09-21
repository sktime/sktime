import shutil
import tempfile
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
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
        return_data_type: str,
    ) -> None:
        super().__init__()
        self._metadata = metadata
        self._save_dir = save_dir
        self._return_data_type = return_data_type

    @property
    def save_dir(self):
        """Return the save directory."""
        return self._save_dir

    @abstractmethod
    def _load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the dataset."""
        raise NotImplementedError()

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the dataset. If not exists, download it first."""
        self.download()
        return self._load()

    def download(self, repeats: Optional[int] = 2, verbose: Optional[bool] = False):
        """Download the dataset."""
        name = self._metadata.name
        format = self._metadata.download_file_format
        urls = [f"{self._metadata.url}/{name}.{format}"]
        if self._metadata.backup_urls:
            urls = urls + [
                f"{url}/{name}.{format}" for url in self._metadata.backup_urls
            ]
        if not self._save_dir.exists():
            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._fallback_download(urls, repeats, verbose)

    def delete(self):
        """Delete the dataset."""
        shutil.rmtree(self._save_dir)

    def _download_extract(self, url: str) -> None:
        """Download zip file to a temp directory and extract it."""
        temp_dir = tempfile.mkdtemp()  # create a temp directory
        zip_file_save_to = Path(temp_dir, self._metadata.name)
        urlretrieve(url, zip_file_save_to)
        try:
            zipfile.ZipFile(zip_file_save_to, "r").extractall(self._save_dir)
        except zipfile.BadZipFile:
            raise zipfile.BadZipFile(
                "Could not unzip dataset. Please make sure the URL is valid."
            )
        finally:
            shutil.rmtree(temp_dir)  # delete temp directory with all its contents

    def _fallback_download(self, urls, repeats, verbose) -> None:
        """Download the dataset from a fallback URL."""
        for url in urls:
            for repeat in range(repeats):
                if verbose:
                    print(  # noqa: T201
                        f"Downloading dataset {self._metadata.name} from {url} "
                        f"to {self._save_dir} (attempt {repeat} of {repeats} total). "
                    )
                try:
                    self._download_extract(url)
                    return  # exit loop when download is successful
                except zipfile.BadZipFile:
                    if verbose:
                        if repeat < repeats - 1:
                            print(  # noqa: T201
                                "Download failed, continuing with next attempt. "
                            )
                        else:
                            print(  # noqa: T201
                                "All attempts for mirror failed, "
                                "continuing with next mirror."
                            )
                            shutil.rmtree(self._save_dir)  # delete directory


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
