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

from skbase.base import BaseObject

from sktime.utils.dependencies import _check_soft_dependencies


class DatasetDownloadStrategy(BaseObject):
    """Abstract base class for dataset download strategies.

    Provides a unified interface to download datasets using various
    strategies (e.g., Hugging Face, direct URLs).

    Subclasses must implement `_download`, which contains the logic for
    downloading and storing datasets.
    """

    def download(self, download_path=None, force_download=False):
        """Download a dataset using this strategy.

        Parameters
        ----------
        download_path : str or Path, optional
            Path where the dataset folder will be created. Defaults to folder
            ``./local_data``, relative to the current working directory.
        force_download : bool, default=False
            If True, deletes and redownloads the dataset folder, even
            if it already exists.

        Side Effects
        ------------
        - Creates the dataset directory at ``{download_path}/`` if it does not exist.
        - Creates dataset files within ``{download_path}/``.
        - If ``force_download=True`` and the dataset folder already exists, the folder
          (and its contents) will be deleted before redownloading.

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

        dataset_folder_path = self._get_dataset_folder_path(download_path)

        if not dataset_folder_path.exists() or force_download:
            if force_download and dataset_folder_path.exists():
                shutil.rmtree(dataset_folder_path)
            self._download(download_path, force_download)

        return self._get_dataset_folder_path(download_path)

    def _get_dataset_folder_path(self, download_path):
        """
        Get the expected path to the dataset folder.

        Should be overridden by subclasses that know the folder structure.
        Default implementation returns the download_path.
        """
        return download_path

    def _download(self, download_path, force_download=False):
        """
        Strategy-specific download implementation.

        This method contains the specific logic for downloading a dataset
        using a particular strategy (e.g., from Hugging Face, URLs, etc.).

        Parameters
        ----------
        download_path : str or Path
            Path where the dataset folder will be created

        Raises
        ------
        Exception
            Should raise an exception if the download fails, allowing the
            `DatasetDownloader` to try the next strategy.
        """
        raise NotImplementedError("Subclasses must implement _download method")


class HuggingFaceDownloader(DatasetDownloadStrategy):
    """Downloads a dataset folder from a Hugging Face Hub repository.

    Uses the `snapshot_download` function from `huggingface_hub` to
    download a specific subdirectory corresponding to `folder_name`
    from a dataset repository.

    Where It Looks
    --------------
    Downloads from a Hugging Face repo such as "sktime/tsc-datasets".
    Only the subdirectory matching `folder_name` is fetched.

    Files and Side Effects
    ----------------------
    The contents of the `folder_name` subdirectory in the repo will be
    downloaded into a new directory:
        {download_path}/{folder_name}/

    This includes all files under that subdirectory, preserving hierarchy.

    Parameters
    ----------
    hf_repo_name : str
        Hugging Face repository in the format "org/repo".
    folder_name : str
        Dataset folder name inside the Hugging Face repository.
    repo_type : str, default="dataset"
        Type of Hugging Face repo.
    token : str, optional
        Authentication token for private repositories.

    Examples
    --------
    >>> from sktime.datasets._dataset_downloader import (  # doctest: +SKIP
    ...     HuggingFaceDownloader
    ... )
    >>> hf_repo_name = "sktime/tsc-datasets"  # doctest: +SKIP
    >>> downloader = HuggingFaceDownloader(
    ...     hf_repo_name, folder_name="Beef"
    ... )  # doctest: +SKIP
    >>> downloader.download()  # doctest: +SKIP

    References
    ----------
    https://huggingface.co/docs/huggingface_hub/en/guides/download
    """

    _tags = {
        "python_dependencies": "huggingface-hub",
    }

    def __init__(self, hf_repo_name, folder_name, repo_type="dataset", token=None):
        super().__init__()
        self.hf_repo_name = hf_repo_name
        self.folder_name = folder_name
        self.repo_type = repo_type
        self.token = token
        self.available = _check_soft_dependencies("huggingface-hub", severity="none")

    def _get_dataset_folder_path(self, download_path):
        """Return the path where the dataset folder will be located."""
        return download_path / self.folder_name

    def _download(self, download_path, force_download=False):
        """
        Download a dataset from the Hugging Face Hub.

        This implementation uses `snapshot_download` to fetch a specific
        dataset folder from the configured repository.

        Parameters
        ----------
        download_path : str or Path, optional
            Path where the dataset folder will be created. Defaults to
            `datasets/local_data`.
        force_download : bool, default=False
            If True, deletes and redownloads the dataset even if it already exists.

        Raises
        ------
        RepositoryNotFoundError
            If the repository is not found on the Hugging Face Hub.
        HfHubHTTPError
            For other Hugging Face Hub related HTTP errors.
        ValueError
            If the expected dataset folder is not found in the downloaded snapshot.
        """
        if not self.available:
            raise ModuleNotFoundError(
                "huggingface-hub not installed, please install it "
                "using pip install huggingface-hub"
            )

        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=self.hf_repo_name,
            repo_type=self.repo_type,
            allow_patterns=f"{self.folder_name}/**",
            local_dir=download_path,
            force_download=force_download,
            token=self.token,
        )

        local_dataset_path = download_path / self.folder_name

        if not local_dataset_path.exists():
            raise ValueError(
                f"Dataset folder '{self.folder_name}' not found"
                f" in repository '{self.hf_repo_name}'"
            )


class URLDownloader(DatasetDownloadStrategy):
    """Downloads and extracts dataset archives from provided direct URLs.

    Assumes that each URL points to a downloadable file (e.g., .zip).
    Downloads the first successful URL and extracts it into the target folder.

    Where It Looks
    --------------
    Downloads from provided URL(s). Must be direct links to the dataset files.

    Files and Side Effects
    ----------------------
    The downloaded archive is extracted into:
        {download_path}/{folder_name}/

    The extracted files are assumed to be flat or internally organized.

    Parameters
    ----------
    base_urls : str or list of str
        Direct URL(s) to zip archives containing the dataset. Must be downloadable
        via `urllib`.

    Examples
    --------
    >>> from sktime.datasets._dataset_downloader import (  # doctest: +SKIP
    ...     URLDownloader,
    ... )
    >>> urls = [
    ...     "https://timeseriesclassification.com/aeon-toolkit/Beef.zip"
    ... ]  # doctest: +SKIP
    >>> downloader = URLDownloader(base_urls=urls)  # doctest: +SKIP
    >>> downloader.download()  # doctest: +SKIP
    """

    def __init__(self, base_urls):
        super().__init__()
        self.base_urls = base_urls if isinstance(base_urls, list) else [base_urls]
        self.available = True

    def _get_dataset_folder_path(self, download_path):
        """Return the path where the dataset folder will be located.

        Since the folder name is determined from the URL during download,
        we need to check what folder would be created from the first URL.
        """
        first_url = self.base_urls[0]
        file_name = os.path.basename(first_url)
        basename = file_name.split(".")[0]
        return download_path / basename

    def _download(self, download_path, force_download):
        """
        Download and extract a dataset from a list of URLs.

        This method attempts to download a zip file from each URL in `self.base_urls`
        in order. The first successful download is extracted and used.

        Parameters
        ----------
        download_path : str or Path, optional
            Path where the dataset folder will be created. Defaults to
            `datasets/local_data`.
        force_download : bool, default=False
            If True, deletes and redownloads the dataset even if it already exists.

        Raises
        ------
        RuntimeError
            If downloading from all provided URLs fails.
        """
        last_error = None

        for url in self.base_urls:
            try:
                self._download_and_extract(url, download_path, force_download)
                return
            except Exception as e:
                last_error = e
                continue

        raise last_error or RuntimeError("Failed to download dataset from any URL")

    def _download_and_extract(self, url, root_path, force=False):
        """Download zip from `url` and extract it to `{root_path}/{dataset_name}/`.

        The target directory is created (or overwritten if `force=True`).
        The downloaded zip file is deleted after extraction.
        """
        file_name = os.path.basename(url)
        dl_dir = tempfile.mkdtemp()
        zip_file_name = os.path.join(dl_dir, file_name)

        basename = file_name.split(".")[0]
        extract_path = root_path / basename

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
    """Composite downloader that attempts multiple download strategies in order.

    Tries each strategy (e.g., HuggingFaceDownloader, URLDownloader) until
    one succeeds, using a retry mechanism.

    Where It Looks
    --------------
    1. Attempts to download from Hugging Face repository (`hf_repo_name`)
       by downloading only the subdirectory matching `folder_name`.

    2. Falls back to downloading zip files from `fallback_urls`.

    Files and Side Effects
    ----------------------
    Regardless of strategy, the dataset is saved at:
        {download_path}/{dataset_name}/

    Parameters
    ----------
    hf_repo_name : str
        Name of the Hugging Face repository (e.g., "sktime/tsc-datasets").
    fallback_urls : str or list of str
        Backup URLs to try if Hugging Face download fails.
    retries : int, default=1
        Number of retries per strategy before failing.

    Examples
    --------
    >>> from sktime.datasets._dataset_downloader import (  # doctest: +SKIP
    ...     DatasetDownloader,
    ... )
    >>> hf_repo_name = "sktime/tsc-datasets"  # doctest: +SKIP
    >>> urls = [
    ...     "https://timeseriesclassification.com/aeon-toolkit/Beef.zip",
    ... ]  # doctest: +SKIP
    >>> downloader = DatasetDownloader(  # doctest: +SKIP
    ...     hf_repo_name=hf_repo_name,
    ...     folder_name="Beef",
    ...     fallback_urls=urls
    ... )
    >>> downloader.download()  # doctest: +SKIP
    """

    def __init__(
        self, hf_repo_name=None, folder_name=None, fallback_urls=None, retries=1
    ):
        super().__init__()
        self.strategies = [
            HuggingFaceDownloader(hf_repo_name, folder_name),
            URLDownloader(fallback_urls),
        ]
        self.retries = retries

    def _get_dataset_folder_path(self, download_path):
        """Get the dataset folder path using the first available strategy."""
        for strategy in self.strategies:
            if strategy.available:
                return strategy._get_dataset_folder_path(download_path)

        return download_path

    def _download(self, download_path, force_download):
        """
        Download dataset using a sequence of strategies.

        This method iterates through the available download strategies
        (e.g., HuggingFaceDownloader, URLDownloader) and attempts to download
        the dataset using each one, with a specified number of retries.
        The first strategy to succeed will terminate the process.

        Parameters
        ----------
        download_path : str or Path, optional
            Path where the dataset folder will be created. Defaults to
            `datasets/local_data`.
        force_download : bool, default=False
            If True, deletes and redownloads the dataset even if it already exists.

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
                    strategy._download(download_path, force_download)
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
