import shutil
import time

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

    if not strategy.available:
        with pytest.raises(ModuleNotFoundError):
            strategy.download(
                DATASET_NAME, download_path=download_path, force_download=False
            )
        return

    # download for the first time
    strategy.download(DATASET_NAME, download_path=download_path, force_download=False)
    assert dataset_path.exists()
    for fname in EXPECTED_FILES:
        assert (dataset_path / fname).exists()

    # download for the second time
    strategy.download(DATASET_NAME, download_path=download_path, force_download=False)
    assert dataset_path.exists()
    for fname in EXPECTED_FILES:
        assert (dataset_path / fname).exists()

    # timestamp after second download
    old_timestamps = {
        fname: (dataset_path / fname).stat().st_mtime for fname in EXPECTED_FILES
    }

    # 1 sec sleep to ensure time difference
    time.sleep(1)

    # download for the third time (force_download=True)
    strategy.download(DATASET_NAME, download_path=download_path, force_download=True)
    assert dataset_path.exists()

    for fname in EXPECTED_FILES:
        fpath = dataset_path / fname
        assert fpath.exists()
        # timestamp after the third download
        new_timestamp = fpath.stat().st_mtime
        assert new_timestamp > old_timestamps[fname], (
            f"Expected {fname} to be re-downloaded (newer timestamp), "
            f"but old={old_timestamps[fname]}, new={new_timestamp}"
        )
