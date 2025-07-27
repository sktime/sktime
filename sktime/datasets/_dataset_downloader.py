"""Utilities for downloading datasets."""

__author__ = ["jgyasu"]

__all__ = ["DatasetDownloader"]

import os
import shutil
import tempfile
import warnings
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from sktime.utils.dependencies import _check_soft_dependencies


class DatasetDownloadStrategy:
    """Base class for dataset download strategies."""

    def download(self, dataset_name, download_path=None, force_download=False):
        """Download a dataset using this strategy.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download.
        download_path : str or Path, optional
            Local directory where the dataset should be saved.
        force_download : bool, default=False
            Whether to force re-download the dataset even if it's cached locally.

        Returns
        -------
        Path
            Path to the downloaded dataset folder.
        """
        if download_path is None:
            download_path = Path.cwd() / "local_data"
        else:
            download_path = Path(download_path)

        download_path.mkdir(parents=True, exist_ok=True)
        local_dataset_path = download_path / dataset_name

        if not local_dataset_path.exists() or force_download:
            self._download(dataset_name, download_path, force_download=force_download)

        return local_dataset_path

    def _download(self, dataset_name, download_path, force_download=False):
        """
        Strategy-specific download implementation.

        This method contains the specific logic for downloading a dataset
        using a particular strategy (e.g., from Hugging Face, URLs, etc.).

        The implementation must ensure that the downloaded and extracted dataset
        is located in a directory named `dataset_name` inside the `download_path`.
        For example, if `download_path` is `/path/to/data` and `dataset_name`
        is `MyDataset`, the final data must be in `/path/to/data/MyDataset/`.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download.
        download_path : Path
            The root directory where the dataset folder should be created.
        force_download : bool
            Whether to force the download even if a local copy might exist.
            Subclasses should respect this flag.

        Raises
        ------
        Exception
            Should raise an exception if the download fails, allowing the
            `DatasetDownloader` to try the next strategy.
        """
        raise NotImplementedError("Subclasses must implement _download method")


class HuggingFaceDownloader(DatasetDownloadStrategy):
    """Download datasets using the Hugging Face Hub snapshot mechanism."""

    _tags = {
        "python_dependencies": "huggingface-hub",
    }

    def __init__(self, repo_name, repo_type="dataset", token=None):
        self.repo_name = repo_name
        self.repo_type = repo_type
        self.token = token
        self.available = _check_soft_dependencies("huggingface-hub", severity="none")

    def _download(self, dataset_name, download_path, force_download=False):
        """
        Download a dataset from the Hugging Face Hub.

        This implementation uses `snapshot_download` to fetch a specific
        dataset folder from the configured repository.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset folder to download from the Hub.
        download_path : Path
            The local root directory to save the dataset folder into.
        force_download : bool
            Passed to `snapshot_download` to force re-downloading.

        Raises
        ------
        RepositoryNotFoundError
            If the repository is not found on the Hugging Face Hub.
        HfHubHTTPError
            For other Hugging Face Hub related HTTP errors.
        ValueError
            If the expected dataset folder is not found in the downloaded snapshot.
        """
        local_dataset_path = download_path / dataset_name

        if not self.available:
            return 0

        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

        try:
            snapshot_download(
                repo_id=self.repo_name,
                repo_type=self.repo_type,
                allow_patterns=f"{dataset_name}/**",
                local_dir=download_path,
                force_download=force_download,
                token=self.token,
            )

            if not local_dataset_path.exists():
                raise ValueError(
                    f"Dataset folder '{dataset_name}' not found"
                    f" in repository '{self.repo_name}'"
                )

        except (RepositoryNotFoundError, HfHubHTTPError, ValueError):
            raise


class URLDownloader(DatasetDownloadStrategy):
    """Strategy for downloading datasets from URLs (with zip extraction)."""

    def __init__(self, base_urls):
        self.base_urls = base_urls
        self.available = True

    def _download(self, dataset_name, download_path, force_download=False):
        """
        Download and extract a dataset from a list of URLs.

        This method attempts to download a zip file from each URL in `self.base_urls`
        in order. The first successful download is extracted and used.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset. This is used to name the final output directory.
        download_path : Path
            The local root directory to save the extracted dataset into.
        force_download : bool
            If True, any existing local directory will be removed before extraction.

        Raises
        ------
        RuntimeError
            If downloading from all provided URLs fails.
        """
        last_error = None
        for url in self.base_urls:
            try:
                self._download_and_extract(
                    url, download_path, dataset_name, force=force_download
                )
                return
            except Exception as e:
                last_error = e
                continue

        raise last_error or RuntimeError(
            f"Failed to download dataset '{dataset_name}' from any URL"
        )

    def _download_and_extract(self, url, root_path, dataset_name, force=False):
        """Download a zip file, extract it, and place it in the target directory."""
        file_name = os.path.basename(url)
        dl_dir = tempfile.mkdtemp()
        zip_file_name = os.path.join(dl_dir, file_name)

        extract_path = root_path / dataset_name

        try:
            urlretrieve(url, zip_file_name)

            if extract_path.exists() and force:
                shutil.rmtree(extract_path)

            extract_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            shutil.rmtree(dl_dir)

            return extract_path

        except (URLError, HTTPError, zipfile.BadZipFile) as e:
            shutil.rmtree(dl_dir)
            if extract_path and extract_path.exists():
                shutil.rmtree(extract_path)

            raise RuntimeError(f"Could not download/extract dataset from {url}: {e}")


class DatasetDownloader(DatasetDownloadStrategy):
    """
    Main dataset downloader class with fallback logic built-in.

    Parameters
    ----------
    hf_repo_name : str
        Hugging Face repository name, e.g., "sktime/tsf-datasets".
    fallback_urls : list of str
        List of URLs to attempt if Hugging Face fails.
    retries : int, default=1
        Number of retries for each strategy.
    """

    def __init__(self, hf_repo_name=None, fallback_urls=None, retries=1):
        self.strategies = [
            HuggingFaceDownloader(hf_repo_name),
            URLDownloader(fallback_urls),
        ]
        self.retries = retries

    def _download(self, dataset_name, download_path, force_download=False):
        """
        Download dataset using a sequence of strategies.

        This method iterates through the available download strategies
        (e.g., HuggingFaceDownloader, URLDownloader) and attempts to download
        the dataset using each one, with a specified number of retries.
        The first strategy to succeed will terminate the process.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download.
        download_path : Path
            The root directory where the dataset folder should be created.
        force_download : bool
            Whether to force the download for each attempt.

        Raises
        ------
        RuntimeError
            If all strategies fail after all retries.
        """
        errors = []

        for i, strategy in enumerate(self.strategies):
            if not strategy.available:
                strategy_name = type(strategy).__name__
                soft_dependency = strategy.get_tag("python_dependencies")
                warnings.warn(
                    f"DatasetDownloader skipping {strategy_name} as "
                    f"it requires {soft_dependency}. We recommend this location as "
                    f"download mirror no. {i + 1}, to use this please install "
                    f"{soft_dependency}, by pip install {soft_dependency}."
                )
                continue

            for attempt in range(self.retries):
                try:
                    strategy._download(
                        dataset_name,
                        download_path,
                        force_download=force_download,
                    )
                    return
                except Exception as e:
                    error_msg = (
                        f"Strategy {i + 1} (attempt {attempt + 1}/{self.retries}): "
                        f"{type(strategy).__name__} failed: {e}"
                    )
                    errors.append(error_msg)

                    if attempt == self.retries - 1:
                        break

        raise RuntimeError(
            f"All download strategies failed after {self.retries} retries each.\n"
            + "\n".join(errors)
        )
