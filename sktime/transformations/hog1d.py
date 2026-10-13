"""HOG1D transform."""

import numbers

import numpy as np
import pandas as pd

from sktime.datatypes._panel._convert import from_nested_to_2d_array
from sktime.transformations.base import BaseTransformer

"""
The HOG1D Transformer proposed by:

@article{zhao2015classifying,
  title={Classifying time series using local descriptors with hybrid sampling},
  author={Zhao, Jiaping and Itti, Laurent},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={28},
  number={3},
  pages={623--637},
  year={2015},
  publisher={IEEE}
}
"""


class HOG1DTransformer(BaseTransformer):
    """HOG1D transform.

    This class is to calculate the HOG1D transform of a
    dataframe of time series data. Works by splitting
    the time series num_intervals times, and calculate
    a histogram of gradients within each interval.

    Parameters
    ----------
    num_intervals : int, default=2
        length of interval.
    num_bins : int, default=8
        num bins in the histogram.
    scaling_factor : float, default=0.1
        a constant that is multiplied to modify the distribution.

    Examples
    --------
    >>> from sktime.transformations.hog1d import HOG1DTransformer
    >>> from sktime.datasets import load_arrow_head
    >>>
    >>> X, y = load_arrow_head(return_X_y=True)
    >>>
    >>> # Initialize the transformer
    >>> hog1d_transformer = HOG1DTransformer(
    ...     num_intervals=5, num_bins=8, scaling_factor=0.1
    ... )
    >>>
    >>> # Transform the data
    >>> Xt = hog1d_transformer.fit_transform(X)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["vnicholson1"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "nested_univ",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
        "capability:categorical_in_X": False,
    }

    def __init__(self, num_intervals=2, num_bins=8, scaling_factor=0.1):
        self.num_intervals = num_intervals
        self.num_bins = num_bins
        self.scaling_factor = scaling_factor
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of X must contain pandas.Series
            Data to fit transform to
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_features]
            each cell of Xt contains pandas.Series
            transformed version of X
        """
        # Get information about the dataframe
        col_names = X.columns
        num_atts = len(X.iloc[0, 0])

        # Check the parameters are appropriate
        self._check_parameters(num_atts)

        df = pd.DataFrame()

        for x in col_names:
            # Convert one of the columns in the dataframe to a numpy array
            arr = from_nested_to_2d_array(pd.DataFrame(X[x]), return_numpy=True)

            # Get the HOG1Ds of all time series at once
            transformedData = self._calculate_hog1ds(arr)

            # Add it to the dataframe
            colToAdd = []
            for i in range(len(transformedData)):
                inst = transformedData[i]
                colToAdd.append(pd.Series(inst))

            df[x] = colToAdd

        return df

    def _calculate_hog1ds(self, X):
        """Calculate the HOG1Ds given time series.

        Parameters
        ----------
        X : a numpy array of shape = [time_series_length] or
            [n_instances, time_series_length]

        Returns
        -------
        HOG1Ds : a numpy array of shape = [num_intervals*num_bins] or
                 [n_instances, num_intervals*num_bins].
                 It contains the histogram of each gradient within
                 each interval.
        """
        splitTimeSeries = self._split_time_series(X)
        HOG1Ds = [self._get_hog1d(interval) for interval in splitTimeSeries]

        return np.concatenate(HOG1Ds, axis=-1)

    def _get_hog1d(self, X):
        """Get the HOG1D given a portion of time series.

        X : a numpy array of shape = [interval_size] or
            [n_instances, interval_size]

        Returns
        -------
        histogram : a numpy array of shape = [num_bins] or
                    [n_instances, num_bins].
        """
        # First step is to pad the portion on both ends once.
        X = np.pad(X, [(0, 0)] * (X.ndim - 1) + [(1, 1)], mode="edge")
        gradients = self.scaling_factor * 0.5 * (X[..., 2:] - X[..., :-2])
        orients = np.degrees(np.arctan(gradients))

        bin_width = 180 / self.num_bins
        his_boundaries = -90 + bin_width + bin_width * np.arange(self.num_bins)

        # index of the first boundary that is >= the orientation
        bin_idx = np.searchsorted(his_boundaries, orients, side="left")

        # count the bin indices per series, ignoring orientations above all bins
        bin_idx = bin_idx.reshape(-1, bin_idx.shape[-1])
        n_series = bin_idx.shape[0]
        offsets = np.arange(n_series)[:, np.newaxis] * self.num_bins
        in_range = bin_idx < self.num_bins
        histogram = np.bincount(
            (bin_idx + offsets)[in_range], minlength=n_series * self.num_bins
        ).astype(float)

        return histogram.reshape(orients.shape[:-1] + (self.num_bins,))

    def _split_time_series(self, X):
        """Split a time series into approximately equal intervals.

        Adopted from = https://stackoverflow.com/questions/2130016/splitting
                       -a-list-into-n-parts-of-approximately-equal-length

        Parameters
        ----------
        X : a numpy array corresponding to the time series being split
            into approx equal length intervals of shape
            [num_intervals,interval_length].
        """
        length = X.shape[-1]
        avg = length / float(self.num_intervals)
        output = []
        beginning = 0.0

        while beginning < length:
            output.append(X[..., int(beginning) : int(beginning + avg)])
            beginning += avg

        return output

    def _check_parameters(self, num_atts):
        """Check the values of parameters inserted into HOG1D.

        Throws
        ------
        ValueError or TypeError if a parameters input is invalid.
        """
        if isinstance(self.num_intervals, int):
            if self.num_intervals <= 0:
                raise ValueError(
                    "num_intervals must have \
                                  the value of at least 1"
                )
            if self.num_intervals > num_atts:
                raise ValueError(
                    "num_intervals cannot be higher \
                                  than subsequence_length"
                )
        else:
            raise TypeError(
                "num_intervals must be an 'int'. \
                            Found '"
                + type(self.num_intervals).__name__
                + "' instead."
            )

        if isinstance(self.num_bins, int):
            if self.num_bins <= 0:
                raise ValueError(
                    "num_bins must have the value of \
                                  at least 1"
                )
        else:
            raise TypeError(
                "num_bins must be an 'int'. Found '"
                + type(self.num_bins).__name__
                + "' \
                            instead."
            )

        if not isinstance(self.scaling_factor, numbers.Number):
            raise TypeError(
                "scaling_factor must be a 'number'. \
                            Found '"
                + type(self.scaling_factor).__name__
                + "' instead."
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        param1 = {}
        param2 = {"num_intervals": 10, "num_bins": 16, "scaling_factor": 0.1}
        param3 = {"num_intervals": 5, "num_bins": 8, "scaling_factor": 1.0}
        return [param1, param2, param3]
