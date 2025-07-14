import shutil

import pytest

from sktime.datasets._dataset_downloader import HuggingFaceDownloader, URLDownloader

HF_REPO_NAME = "sktime/tsc-datasets"
DATASET_NAME = "Beef"
URL = ["https://timeseriesclassification.com/aeon-toolkit/Beef.zip"]
EXPECTED_FILES = ["Beef_TRAIN.ts", "Beef_TEST.ts"]


@pytest.mark.parametrize(
    "strategy",
    [
        HuggingFaceDownloader(HF_REPO_NAME),
        URLDownloader(URL),
    ],
)
def test_downloader_strategy_behavior(tmp_path, strategy):
    """Test different dataset download strategies."""
    download_path = tmp_path / "test_data"

    dataset_path = download_path / DATASET_NAME
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    strategy.download(DATASET_NAME, download_path=download_path, force_download=False)
    assert dataset_path.exists()
    for fname in EXPECTED_FILES:
        assert (dataset_path / fname).exists()

    strategy.download(DATASET_NAME, download_path=download_path, force_download=False)
    assert dataset_path.exists()
    for fname in EXPECTED_FILES:
        assert (dataset_path / fname).exists()

    strategy.download(DATASET_NAME, download_path=download_path, force_download=True)
    assert dataset_path.exists()
    for fname in EXPECTED_FILES:
        assert (dataset_path / fname).exists()
