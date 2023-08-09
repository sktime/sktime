"""Test single problem loaders with varying return types."""
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import (  # Univariate; Unequal length; Multivariate
    load_acsf1,
    load_arrow_head,
    load_basic_motions,
    load_forecastingdata,
    load_italy_power_demand,
    load_japanese_vowels,
    load_osuleaf,
    load_plaid,
    load_UCR_UEA_dataset,
    load_unit_test,
)
from sktime.datasets.tsf_dataset_names import tsf_all, tsf_all_datasets

UNIVARIATE_PROBLEMS = [
    load_acsf1,
    load_arrow_head,
    load_italy_power_demand,
    load_osuleaf,
    load_unit_test,
]
MULTIVARIATE_PROBLEMS = [
    load_basic_motions,
]
UNEQUAL_LENGTH_PROBLEMS = [
    load_plaid,
    load_japanese_vowels,
]


@pytest.mark.parametrize(
    "loader", UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS + UNEQUAL_LENGTH_PROBLEMS
)
def test_load_dataframe(loader):
    """Test that we can load all baked in TSC problems into nested pd.DataFrames."""
    # should work for all
    X, y = loader()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    X = loader(return_X_y=False)
    assert isinstance(X, pd.DataFrame)


@pytest.mark.parametrize("loader", UNIVARIATE_PROBLEMS + MULTIVARIATE_PROBLEMS)
@pytest.mark.parametrize("split", [None, "train", "test", "TRAIN", "TEST"])
def test_load_numpy3d(loader, split):
    """Test that we can load equal length TSC problems into numpy3d."""
    X, y = loader(split=split, return_type="numpy3d")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 3
    assert y.ndim == 1


@pytest.mark.parametrize("loader", UNIVARIATE_PROBLEMS)
def test_load_numpy2d_univariate(loader):
    """Test that we can load univariate equal length TSC problems into numpy2d."""
    X, y = loader(return_type="numpy2d")
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.ndim == 2
    assert y.ndim == 1


@pytest.mark.parametrize("loader", MULTIVARIATE_PROBLEMS)
def test_load_numpy2d_multivariate_raises(loader):
    """Test that multivariate and/or unequal length raise the correct error."""
    with pytest.raises(ValueError, match="attempting to load into a numpy2d"):
        X, y = loader(return_type="numpy2d")


@pytest.mark.xfail(
    reason="repeated upstream location failures, see 4754. xfail until fixed."
)
def test_load_UEA():
    """Test loading of a random subset of the UEA data, to check API."""
    from sktime.datasets.tsc_dataset_names import multivariate, univariate

    TOO_LARGE_DATA = ["InsectWingbeat"]

    univariate = list(set(univariate).difference(TOO_LARGE_DATA))
    multivariate = list(set(multivariate).difference(TOO_LARGE_DATA))

    n_univariate = 3
    n_multivariate = 2

    univ_names = np.random.choice(univariate, n_univariate)
    mult_names = np.random.choice(multivariate, n_multivariate)

    for univ_name in univ_names:
        load_UCR_UEA_dataset(univ_name)

    for mult_name in mult_names:
        load_UCR_UEA_dataset(mult_name)


def test_load_forecastingdata():
    """Test loading downloaded dataset from forecasting.org."""
    file = "UnitTest"
    loaded_datasets, metadata = load_forecastingdata(name=file)
    assert len(loaded_datasets) == 1
    assert metadata["frequency"] == "yearly"
    assert metadata["forecast_horizon"] == 4
    assert metadata["contain_missing_values"] is False
    assert metadata["contain_equal_length"] is False


@pytest.mark.parametrize("name", tsf_all_datasets)
def test_check_link_downloadable(name):
    """Test dataset URL from forecasting.org is downloadable and exits."""
    url = f"https://zenodo.org/record/{tsf_all[name]}/files/{name}.zip"

    # Send a GET request to check if the link exists without downloading the file
    # response = requests.get(url, stream=True)
    req = Request(url, method="HEAD")
    response = urlopen(req)

    # Check if the response status code is 200 (OK)
    assert (
        response.status == 200
    ), f"URL is not valid or does not exist. Error code {response.status}."

    # Check if the response headers indicate that the content is downloadable
    content_type = response.headers.get("Content-Type")
    content_disposition = response.headers.get("Content-Disposition")

    assert "application/octet-stream" in content_type, "URL is not downloadable."
    assert "attachment" in content_disposition, "URL is not downloadable."
