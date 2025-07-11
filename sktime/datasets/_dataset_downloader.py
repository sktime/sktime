"""Utilities for downloading datasets."""

__author__ = ["jgyasu"]

__all__ = ["DatasetDownloader", "HuggingFaceDownloader", "URLDownloader"]

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

    def download(
        self, dataset_name, download_path=None, force_download=False, **kwargs
    ):
        """Download a dataset using this strategy.

        This method handles common boilerplate, then delegates to
        _download for strategy-specific implementation.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download.
        download_path : str or Path, optional
            Local directory where the dataset should be saved. If not specified,
            defaults to a 'local_data' folder in the current working directory.
        force_download : bool, default=False
            Whether to force re-download the dataset even if it's cached locally.
        **kwargs
            Additional keyword arguments passed to the strategy implementation.

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

        if local_dataset_path.exists():
            if force_download:
                return self._download(dataset_name, download_path, **kwargs)
            return local_dataset_path

        return self._download(dataset_name, download_path, **kwargs)

    def _download(self, dataset_name, download_path, **kwargs):
        """Strategy-specific download implementation."""
        raise NotImplementedError("Subclasses must implement _download method")


class HuggingFaceDownloader(DatasetDownloadStrategy):
    """Download datasets using the Hugging Face Hub snapshot mechanism.

    Implements a strategy to download datasets stored in Hugging Face
    repositories.

    Parameters
    ----------
    repo_name : str
        Name of the Hugging Face repository containing the datasets.
        In the format: "organisation/repository_name",
        e.g., "sktime/tsf-datasets"
    repo_type : str, default="dataset"
        Type of repository (e.g., "dataset", "model", etc.).
    token : str, optional
        Hugging Face authentication token for private repositories.

    References
    ----------
    https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
    """

    def __init__(self, repo_name, repo_type="dataset", token=None):
        self.repo_name = repo_name
        self.repo_type = repo_type
        self.token = token
        self.available = _check_soft_dependencies("huggingface-hub", severity="none")

    def _download(self, dataset_name, download_path, **kwargs):
        """Download a specific dataset folder from the Hugging Face repository.

        Parameters
        ----------
        dataset_name : str
            Name of the folder (dataset) inside the repository to download.
        download_path : Path
            Local directory where the dataset should be saved.
        **kwargs
            Additional keyword arguments passed to `snapshot_download`.

        Returns
        -------
        Path
            Path to the downloaded dataset folder.
        """
        local_dataset_path = download_path / dataset_name

        if self.available:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

        try:
            snapshot_download(
                repo_id=self.repo_name,
                repo_type=self.repo_type,
                allow_patterns=f"{dataset_name}/**",
                local_dir=download_path,
                force_download=False,
                token=self.token,
                **kwargs,
            )

            if not local_dataset_path.exists():
                raise ValueError(
                    f"Dataset folder '{dataset_name}' not found"
                    " in repository '{self.repo_name}'"
                )

            return local_dataset_path

        except (RepositoryNotFoundError, HfHubHTTPError, ValueError):
            raise


class URLDownloader(DatasetDownloadStrategy):
    """Strategy for downloading datasets from URLs (with zip extraction).

    Parameters
    ----------
    base_urls: List of URLs to attempt downloading from.
    """

    def __init__(self, base_urls):
        self.base_urls = base_urls

    def _download(self, dataset_name, download_path, **kwargs):
        """Download and extract a dataset from URL.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download.
        download_path : Path
            Local directory where the dataset should be saved.
        **kwargs
            Additional keyword arguments (unused in this implementation).

        Returns
        -------
        Path
            Path to the downloaded dataset folder.
        """
        last_error = None
        for url in self.base_urls:
            try:
                result_path = self._download_and_extract(url, download_path)
                if result_path:
                    return result_path

            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        else:
            raise RuntimeError(
                f"Failed to download dataset '{dataset_name}' from any URL"
            )

    def _download_and_extract(self, url: str, extract_path=None):
        """Download and unzip datasets (helper function)."""
        file_name = os.path.basename(url)
        dl_dir = tempfile.mkdtemp()
        zip_file_name = os.path.join(dl_dir, file_name)

        try:
            urlretrieve(url, zip_file_name)

            if extract_path is None:
                extract_path = Path.cwd() / "local_data" / file_name.split(".")[0]
            else:
                extract_path = Path(extract_path) / file_name.split(".")[0]

            extract_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
                zip_ref.extractall(extract_path)

            shutil.rmtree(dl_dir)

            return extract_path

        except (URLError, HTTPError, zipfile.BadZipFile) as e:
            shutil.rmtree(dl_dir)
            if extract_path and extract_path.exists():
                shutil.rmtree(extract_path)

            error_msg = f"Could not download/extract dataset from {url}: {e}"
            raise RuntimeError(error_msg)


class FallbackDownloader(DatasetDownloadStrategy):
    """
    Composite downloader that tries multiple strategies in sequence.

    Parameters
    ----------
    strategies : List[DatasetDownloadStrategy]
        List of download strategies to try in order.
    retries : int, default=1
        Number of retries for each strategy before moving to the next.
    """

    def __init__(self, strategies, retries=1):
        self.strategies = strategies
        self.retries = retries

    def _download(self, dataset_name, download_path, **kwargs):
        """
        Try each strategy in sequence until one succeeds.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download.
        download_path : Path
            Local directory where the dataset should be saved.
        **kwargs
            Arguments passed to each strategy's download method.

        Returns
        -------
        Path
            Path to the downloaded dataset.

        Raises
        ------
        RuntimeError
            If all strategies fail after retries.
        """
        errors = []

        for i, strategy in enumerate(self.strategies):
            if isinstance(strategy, HuggingFaceDownloader) and not strategy.available:
                next_strategy = type(self.strategies[i + 1]).__name__
                warnings.warn(
                    f"huggingface_hub is not available, please install it to"
                    f" use Hugging Face for dataset downloads, skipping "
                    f"HuggingFaceDownloader, using {next_strategy}."
                )
                continue
            for attempt in range(self.retries):
                try:
                    return strategy._download(dataset_name, download_path, **kwargs)
                except Exception as e:
                    strategy_name = type(strategy).__name__
                    error_msg = (
                        f"Strategy {i + 1} (attempt {attempt + 1}/{self.retries}): "
                        f"{strategy_name} failed: {e}"
                    )
                    errors.append(error_msg)

                    if attempt == self.retries - 1:
                        break

        raise RuntimeError(
            f"All download strategies failed after {self.retries} retries each.\n"
            + "\n".join(errors)
        )


class DatasetDownloader:
    """Main dataset downloader class using the composite pattern."""

    def __init__(self, hf_repo_name=None, fallback_urls=None):
        """
        Initialize the Dataset Downloader with multiple strategies.

        Parameters
        ----------
        hf_repo_name : str
            Hugging Face repository name, in the format: "organisation/repo_name"
        fallback_urls : list of str
            List of base URLs for fallback downloads
        """
        strategies = [HuggingFaceDownloader(hf_repo_name), URLDownloader(fallback_urls)]

        self._downloader = FallbackDownloader(strategies)

    def download(
        self, dataset_name, download_path=None, force_download=False, **kwargs
    ):
        """
        Download a dataset using the configured fallback strategy.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to download.
        download_path : str or Path, optional
            Local directory where the dataset should be saved.
        force_download : bool, default=False
            Whether to force re-download the dataset.
        **kwargs
            Additional keyword arguments passed to the download strategies.

        Returns
        -------
        Path
            Path to the downloaded dataset.
        """
        return self._downloader.download(
            dataset_name=dataset_name,
            download_path=download_path,
            force_download=force_download,
            **kwargs,
        )
