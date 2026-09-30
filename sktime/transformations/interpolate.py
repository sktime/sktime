"""Time series interpolator/re-sampler."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer

__author__ = ["mloning"]


class TSInterpolator(BaseTransformer):
    """Time series interpolator/re-sampler.

    Transformer that rescales series for another number of points.
    For each time series, the transformer fits a scipy linear interp1d
    and samples a user-defined number of points. Points are generated
    by numpy.linspace.

    After transformation each time series will have the given length.

    Parameters
    ----------
    length : integer, the length of time series to resize to.
    """

    _tags = {
        "authors": ["mloning"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "pd-multiindex",
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,
        "python_dependencies": "scipy",
    }

    def __init__(self, length):
        """Initialize estimator.

        Parameters
        ----------
        length : integer, the length of time series to resize to.
        """
        if length <= 0 or (not isinstance(length, int)):
            raise ValueError("resizing length must be integer and > 0")

        self.length = length
        super().__init__()

    def _resize_series(self, values):
        """Resize a single 1D array via linear interpolation.

        Fits a 1D linear interpolation on the original array and samples
        ``self.length`` evenly spaced points.

        Parameters
        ----------
        values : np.ndarray
            1D array of values to interpolate.

        Returns
        -------
        np.ndarray : interpolated array with ``self.length`` elements.
        """
        from scipy import interpolate

        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, self.length)
        f = interpolate.interp1d(x_old, values)
        return f(x_new)

    def _transform(self, X, y=None):
        """Interpolate each time series in the panel to a fixed length.

        Parameters
        ----------
        X : pd.DataFrame with pd.MultiIndex
            Panel data in pd-multiindex format. MultiIndex has two levels:
            first level is instance index, second level is time index.
        y : ignored argument for interface compatibility

        Returns
        -------
        Xt : pd.DataFrame with pd.MultiIndex, same format as X
            Transformed version of X where every instance has been
            resampled to ``self.length`` time points.
        """
        instances = X.index.get_level_values(0).unique()

        result_frames = []
        for inst_id in instances:
            inst_data = X.loc[inst_id]

            transformed = {}
            for col in X.columns:
                transformed[col] = self._resize_series(inst_data[col].values)

            inst_df = pd.DataFrame(transformed)
            inst_df.index = pd.MultiIndex.from_arrays(
                [[inst_id] * self.length, range(self.length)],
                names=X.index.names,
            )
            result_frames.append(inst_df)

        return pd.concat(result_frames)

    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        params1 = {"length": 10}
        params2 = {"length": 5}
        return [params1, params2]
