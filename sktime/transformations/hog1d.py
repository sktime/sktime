"""HOG1D transform."""

import math
import numbers

import numpy as np
import pandas as pd

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
        "authors": ["vnicholson1", "shubham5080"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd-multiindex",  # replace nested_univ (#10716)
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
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
        X : pd.DataFrame with MultiIndex (instance, time)
            Panel of equal-length series; columns are variables
        y : ignored argument for interface compatibility
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : nested pandas DataFrame of shape [n_instances, n_variables]
            each cell of Xt contains pandas.Series of HOG1D features
        """
        instances = X.index.get_level_values(0).unique()
        num_atts = len(X.xs(instances[0], level=0))

        # Check the parameters are appropriate
        self._check_parameters(num_atts)

        df = pd.DataFrame()

        for col in X.columns:
            # shape (n_instances, n_timepoints)
            arr = np.stack(
                [X.xs(i, level=0)[col].to_numpy(dtype=float) for i in instances]
            )

            # Get the HOG1Ds of each time series for this variable
            transformedData = []
            for i in range(len(instances)):
                inst = self._calculate_hog1ds(arr[i])
                transformedData.append(inst)

            # Convert to numpy array
            transformedData = np.asarray(transformedData)

            # Add it to the dataframe
            colToAdd = []
            for i in range(len(transformedData)):
                inst = transformedData[i]
                colToAdd.append(pd.Series(inst))

            df[col] = colToAdd

        return df

    def _calculate_hog1ds(self, X):
        """Calculate the HOG1Ds given a time series.

        Parameters
        ----------
        X : a numpy array of shape = [time_series_length]

        Returns
        -------
        HOG1Ds : a numpy array of shape = [num_intervals*num_bins].
                 It contains the histogram of each gradient within
                 each interval.
        """
        # Firstly, split the time series into approx equal
        # length intervals
        splitTimeSeries = self._split_time_series(X)
        HOG1Ds = []

        for x in range(len(splitTimeSeries)):
            HOG1Ds.extend(self._get_hog1d(splitTimeSeries[x]))

        return HOG1Ds

    def _get_hog1d(self, X):
        """Get the HOG1D given a portion of a time series.

        X : a numpy array of shape = [interval_size]

        Returns
        -------
        histogram : a numpy array of shape = [num_bins].
        """
        # First step is to pad the portion on both ends once.
        gradients = [0.0] * (len(X))
        X = np.pad(X, 1, mode="edge")
        histogram = [0.0] * self.num_bins

        # Calculate the gradients of each element
        for i in range(1, len(X) - 1):
            gradients[(i - 1)] = self.scaling_factor * 0.5 * (X[(i + 1)] - X[(i - 1)])

        # Calculate the orientations
        orients = [math.degrees(math.atan(x)) for x in gradients]

        # Calculate the boundaries of the histogram
        hisBoundaries = [
            -90 + (180 / self.num_bins) + ((180 / self.num_bins) * x)
            for x in range(self.num_bins)
        ]

        # Construct the histogram
        for x in range(len(orients)):
            orientToAdd = orients[x]
            for y in range(len(hisBoundaries)):
                if orientToAdd <= hisBoundaries[y]:
                    histogram[y] += 1.0
                    break

        return histogram

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
        avg = len(X) / float(self.num_intervals)
        output = []
        beginning = 0.0

        while beginning < len(X):
            output.append(X[int(beginning) : int(beginning + avg)])
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
