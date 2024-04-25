"""Test data loaders that download from external sources."""
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from sktime.datasets import load_forecastingdata, load_solar, load_UCR_UEA_dataset
from sktime.datasets.tsf_dataset_names import tsf_all, tsf_all_datasets

# test tsf download only on a random uniform subsample of datasets
N_TSF_SUBSAMPLE = 3
TSF_SUBSAMPLE = np.random.choice(tsf_all_datasets, N_TSF_SUBSAMPLE)


@pytest.mark.datadownload
def test_load_solar():
    """Test whether solar dataset can be downloaded."""
    solar = load_solar()

    assert isinstance(solar, pd.Series)
    assert len(solar) == 5905


@pytest.mark.xfail(reason="known sporadic failure of unknown cause, see #5460")
@pytest.mark.datadownload
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


@pytest.mark.datadownload
def test_load_forecastingdata():
    """Test loading downloaded dataset from forecasting.org."""
    file = "UnitTest"
    loaded_datasets, metadata = load_forecastingdata(name=file)
    assert len(loaded_datasets) == 1
    assert metadata["frequency"] == "yearly"
    assert metadata["forecast_horizon"] == 4
    assert metadata["contain_missing_values"] is False
    assert metadata["contain_equal_length"] is False


@pytest.mark.xfail(reason="known sporadic failure of unknown cause, see #5462")
@pytest.mark.datadownload
@pytest.mark.parametrize("name", TSF_SUBSAMPLE)
def test_check_link_downloadable(name):
    """Test dataset URL from forecasting.org is downloadable and exits."""
    url = f"https://zenodo.org/record/{tsf_all[name]}/files/{name}.zip"

    # Send a GET request to check if the link exists without downloading the file
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
