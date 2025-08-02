"""Solar dataset for time series forecasting."""

import pandas as pd

from sktime.datasets._single_problem_loaders import load_solar
from sktime.datasets.forecasting._base import _ForecastingDatasetFromLoader

__all__ = ["Solar"]


class Solar(_ForecastingDatasetFromLoader):
    """Load the GB National Solar Estimates dataset for time series forecasting.

    This class wraps the Sheffield Solar PV_Live API to extract national solar data
    for the GB electricity network. Note that these are estimates of the true solar
    generation, as the true values are "behind the meter" and unknown.

    Parameters
    ----------
    start : string, default="2021-05-01"
        The start date of the time series in "YYYY-MM-DD" format.
    end : string, default="2021-09-01"
        The end date of the time series in "YYYY-MM-DD" format.
    normalise : boolean, default=True
        Normalise the returned time series by installed capacity.
    return_full_df : boolean, default=False
        Return a pd.DataFrame with power, capacity, and normalised estimates.
    api_version : string or None, default="v4"
        API version to call. If None, a stored sample of the data is loaded.

    Examples
    --------
    >>> from sktime.datasets.forecasting import Solar
    >>> y = Solar().load("y")

    Notes
    -----
    The returned time series is half-hourly. For more information, refer to:

    References
    ----------
    .. [1] https://www.solar.sheffield.ac.uk/pvlive/
    .. [2] https://www.solar.sheffield.ac.uk/pvlive/api/
    """

    _tags = {
        "name": "solar",
        "n_splits": 0,  # No splits available
        "is_univariate": True,
        "is_one_series": True,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_empty": False,
        "has_nans": False,  # May depend on API data quality
        "has_exogenous": False,  # The series itself is standalone
        "n_instances": None,  # Only one series is returned
        "n_timepoints": None,  # Dynamic depending on the time range
        "frequency": "30min",  # Half-hourly data
        "n_dimensions": 1,
        "n_panels": 1,
        "n_hierarchy_levels": 0,
    }

    loader_func = load_solar

    def __init__(
        self,
        start="2021-05-01",
        end="2021-09-01",
        normalise=True,
        return_full_df=False,
        api_version="v4",
    ):
        self.start = start
        self.end = end
        self.normalise = normalise
        self.return_full_df = return_full_df
        self.api_version = api_version
        super().__init__()

        start = pd.to_datetime(self.start)
        end = pd.to_datetime(self.end)
        n_timepoints = (end - start).days * 24 * 2 + 1

        n_dimensions = 3 if return_full_df else 1

        self.set_tags(
            **{
                "n_timepoints": n_timepoints,
                "n_instances": n_timepoints,
                "n_dimensions": n_dimensions,
                "is_univariate": n_dimensions == 1,
            }
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameters."""
        return [
            {
                "start": "2021-05-01",
                "end": "2021-05-02",
                "normalise": True,
                "return_full_df": False,
                "api_version": "v4",
            },
            {
                "start": "2021-06-01",
                "end": "2021-06-02",
                "normalise": True,
                "return_full_df": True,
                "api_version": "v4",
            },
        ]
