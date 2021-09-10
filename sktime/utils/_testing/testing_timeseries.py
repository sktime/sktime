# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Union, List

from sktime.datatypes import convert_to, Mtype
from sktime.utils._testing.series import _make_series
from sktime.utils._testing.panel import _make_panel_X
from sktime.utils.validation.panel import from_nested_to_3d_numpy


class TestingTimeseries:
    """
    Class used to create time series used for testing purposes. An example use case
    is ensuring a function can work with all possible time series formats.

    Parameters
    ----------
    mtype: Mtype or str
        The machine type of the timeseries being passed

    timeseries: List or pd.Dataframe or np.ndarray or pd.Series
        Timeseries to run tests on
    """

    def __init__(
            self,
            mtype: Union[Mtype, str],
            timeseries: Union[List, pd.DataFrame, np.ndarray, pd.Series]
    ):
        if isinstance(mtype, str):
            mtype = Mtype(mtype.upper())
        self._mtype: Mtype = mtype
        self._timeseries = timeseries

    def iter_timeseries_variations(
            self,
            ignore_type: Union[List[str], List[Mtype]] = None
    ):
        """
        Generator that returns the timeseries in a different format every time it is
        called.

        Parameters
        ----------
        ignore_type: List[str] or List[Mtype], defaults = []
            List containing the types to ignore i.e not to yield
        """
        validated_ignore = []
        if ignore_type is not None:
            for curr_type in ignore_type:
                if isinstance(curr_type, str):
                    validated_ignore.append(Mtype[curr_type])
                else:
                    validated_ignore.append(curr_type)

        conversion_dict = self._mtype.conversion_dict
        for conversion in conversion_dict:
            if conversion in validated_ignore:
                continue
            yield_val = convert_to(self._timeseries, conversion)
            yield yield_val

    def as_type(self, to_type: Union[str, Mtype]):
        """
        Method used to get the timeseries as a specific type

        Parameters
        ----------
        to_type: str or Mtype
            Type to get timeseries as
        """
        return convert_to(self._timeseries, to_type=to_type)

    @staticmethod
    def create_univariate_series(
            n_timepoints: int = 10,
            random_state: int = 1
    ):
        """
        Static method used to create a univariate timeseries TestingTimeseries object

        Parameters
        ----------
        n_timepoints: int, defaults = 10
            Number of timepoints to generate
        random_state: int, defaults = 1
            Random state to generate from

        Returns
        -------
        TestingTimeseries
            Univariate TestingTimeseries
        """
        series: pd.Series = _make_series(
            n_timepoints=n_timepoints,
            n_columns=1,
            random_state=random_state
        )
        return TestingTimeseries(Mtype.PD_SERIES, series)

    @staticmethod
    def create_multivariate_series(
            n_timepoints: int = 10,
            n_columns: int = 10,
            random_state: int = 1
    ):
        """
        Static method used to create a multivariate timeseries TestingTimeseries object

        Parameters
        ----------
        n_timepoints: int, defaults = 10
            Number of timepoints to generate
        n_columns: int, defaults = 10
            Number of columns to generate
        random_state: int, defaults = 1
            Random state to generate from

        Returns
        -------
        TestingTimeseries
            Multivariate TestingTimeseries
        """
        series: pd.DataFrame = _make_series(
            n_timepoints=n_timepoints,
            n_columns=n_columns,
            random_state=random_state
        )
        return TestingTimeseries(Mtype.PD_DATAFRAME, series)

    @staticmethod
    def create_univariate_panel(
            n_timepoints: int = 10,
            n_instances: int = 10,
            random_state: int = 1
    ):
        """
        Static method used to create a univariate panel timeseries TestingTimeseries
        object

        Parameters
        ----------
        n_timepoints: int, defaults = 10
            Number of timepoints to generate
        n_instances: int, defaults = 10
            Number of timeseries in the panel to generate
        random_state: int, defaults = 1
            Random state to generate from

        Returns
        -------
        TestingTimeseries
            Univariate panel TestingTimeseries
        """
        panel = pd.DataFrame = _make_panel_X(
            n_instances=n_instances,
            n_columns=1,
            n_timepoints=n_timepoints,
            random_state=random_state
        )
        return TestingTimeseries(Mtype.NUMPY_3D, from_nested_to_3d_numpy(panel))

    @staticmethod
    def create_multivariate_panel(
            n_timepoints: int = 10,
            n_columns: int = 10,
            n_instances: int = 10,
            random_state: int = 1
    ):
        """
         Static method used to create a multivariate panel timeseries TestingTimeseries
         object

         Parameters
         ----------
         n_timepoints: int, defaults = 10
             Number of timepoints to generate
         n_columns: int, defaults = 10
            Number of columns to generate
         n_instances: int, defaults = 10
             Number of timeseries in the panel to generate
         random_state: int, defaults = 1
             Random state to generate from

         Returns
         -------
         TestingTimeseries
             Multivariate panel TestingTimeseries
         """
        panel: pd.DataFrame = _make_panel_X(
            n_instances=n_instances,
            n_columns=n_columns,
            n_timepoints=n_timepoints,
            random_state=random_state
        )
        return TestingTimeseries(Mtype.NUMPY_3D, from_nested_to_3d_numpy(panel))


# if __name__ == "__main__":
#     test_timeseries_panel = TestingTimeseries.create_multivariate_panel()
#     for val in test_timeseries_panel.iter_timeseries_variations():
#         print(type(val))
