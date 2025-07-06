"""Utilities for downloading datasets."""

__author__ = ["jgyasu"]

__all__ = ["DatasetDownloader"]

import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve

from sktime.utils.dependencies import _safe_import

snapshot_download = _safe_import(
    "huggingface_hub.snapshot_download", pkg_name="huggingface-hub"
)
hf_hub_download = _safe_import(
    "huggingface_hub.hf_hub_download", pkg_name="huggingface-hub"
)
HfHubHTTPError = _safe_import(
    "huggingface_hub.utils.HfHubHTTPError", pkg_name="huggingface-hub"
)
RepositoryNotFoundError = _safe_import(
    "huggingface_hub.utils.RepositoryNotFoundError", pkg_name="huggingface-hub"
)


class DatasetDownloadStrategy:
    """Base class for dataset download strategies."""

    def download_dataset(self, **kwargs):
        """Download a dataset using this strategy."""
        raise NotImplementedError("Subclasses must implement download_dataset method")


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

    References
    ----------
    https://huggingface.co/docs/huggingface_hub/en/guides/download#filter-files-to-download
    """

    def __init__(self, repo_name, repo_type="dataset"):
        self.repo_name = repo_name
        self.repo_type = repo_type

    def download_dataset(
        self,
        dataset_folder,
        download_path=None,
        force_download=False,
        token=None,
        **kwargs,
    ):
        """Download a specific dataset folder from the Hugging Face repository.

        Parameters
        ----------
        dataset_folder : str
            Name of the folder (dataset) inside the repository to download.
        download_path : str or Path, optional
            Local directory where the dataset should be saved. If not specified,
            defaults to a 'local_data' folder in the current working directory.
        force_download : bool, default=False
            Whether to force re-download the dataset even if it's cached locally.
        token : str, optional
            Hugging Face authentication token for private repositories.
        **kwargs
            Additional keyword arguments passed to `snapshot_download`.

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

        local_dataset_path = download_path / dataset_folder

        try:
            snapshot_download(
                repo_id=self.repo_name,
                repo_type=self.repo_type,
                allow_patterns=f"{dataset_folder}/**",
                local_dir=download_path,
                force_download=force_download,
                token=token,
            )

            if not local_dataset_path.exists():
                raise ValueError(
                    f"Dataset folder '{dataset_folder}' not found"
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

    def download_dataset(
        self, dataset_name, download_path=None, force_download=False, **kwargs
    ):
        """Download and extract a dataset from URL."""
        if download_path is None:
            download_path = Path.cwd() / "local_data"
        else:
            download_path = Path(download_path)

        extract_path = download_path / dataset_name

        if extract_path.exists() and not force_download:
            return extract_path

        last_error = None
        for url in self.base_urls:
            try:
                dataset_url = f"{url}/{dataset_name}.zip"

                result_path = self._download_and_extract(dataset_url, download_path)
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
            raise Exception(error_msg)


class DatasetDownloader:
    """
    Main dataset downloader class.

    This class implements the strategy pattern to support multiple download methods:
    1. Hugging Face repositories (primary)
    2. URL-based downloads with zip extraction (fallback)
    """

    def __init__(self, hf_repo_name="sktime/tsf-datasets", fallback_urls=None):
        """
        Initialize the Dataset Downloader with multiple strategies.

        Args:
            hf_repo_name (str): Hugging Face repository name,
                in the format: "organisation/repo_name", e.g.,
                "sktime/tsf-datasets"
            fallback_urls (Optional[List[str]]):
                List of base URLs for fallback downloads
        """
        self.hf_strategy = HuggingFaceDownloader(hf_repo_name)

        self.url_strategy = URLDownloader(fallback_urls)

    def download_dataset(
        self, dataset_name, download_path=None, force_download=False, **kwargs
    ):
        """Download a dataset using multiple strategies."""
        try:
            return self.hf_strategy.download_dataset(
                dataset_folder=dataset_name,
                download_path=download_path,
                force_download=force_download,
                **kwargs,
            )
        except Exception as hf_error:
            try:
                return self.url_strategy.download_dataset(
                    dataset_name=dataset_name,
                    download_path=download_path,
                    force_download=force_download,
                    **kwargs,
                )
            except Exception as url_error:
                raise RuntimeError(
                    f"All download strategies failed.\n"
                    f"HuggingFace error: {hf_error}\n"
                    f"URL error: {url_error}"
                )
