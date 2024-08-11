# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a merger for panel data."""

import numpy as np

__author__ = ["benHeid"]


from sktime.transformations.base import BaseTransformer


class Merger(BaseTransformer):
    """Aggregates Panel data containing overlapping windows of one time series.

    The input data contains multiple overlapping time series elements that could
    arranged as follows:
    xxxx.....
    .xxxx....
    ..xxxx...
    ...xxxx..
    ....xxxx.
    .....xxxx
    ......xxxx
    .......xxxx
    ........xxxx
    .........xxxx
    The merger aggregates the data by aligning the time series windows as shown above
    and applying a aggregation function to the overlapping data points.
    The aggregation function can be one of "mean" or "median". I.e., the ``mean`` or
    ``median`` of each column is calculated, resulting in a univariate time series.

    Parameters
    ----------
    method : {``median``, ``mean``}, default="median"
        The method to use for aggregation. Can be one of "mean" or "median".
    stride : int, default=0
        The stride to use for the aggregation. The stride determines the number of
        shifts between consecutive instances. A stride of 0 means no shift. A
        stride of 1 means that the time series is aggregated as above.

    Examples
    --------
    >>> from sktime.transformations.merger import Merger
    >>> from sktime.utils._testing.panel import _make_panel
    >>> y = _make_panel(n_instances=10, n_columns=3, n_timepoints=5)
    >>> result = Merger(method="median").fit_transform(y)
    >>> result.shape
    (3, 5)

    >>> from sktime.transformations.merger import Merger
    >>> from sktime.utils._testing.panel import _make_panel
    >>> y = _make_panel(n_instances=10, n_columns=3, n_timepoints=5)
    >>> result = Merger(method="median", stride=1).fit_transform(y)
    >>> result.shape
    (14, 3)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "benHeid",
        "maintainers": "benHeid",
        # estimator type
        # ---------------
        "scitype:transform-input": "Panel",
        "scitype:transform-output": "Series",
        "X_inner_mtype": "numpy3D",
        "fit_is_empty": True,
    }

    def __init__(self, method="median", stride=0):
        self.stride = stride
        if method not in ["median", "mean"]:
            raise ValueError(f"{method} must be 'mean' or 'median'.")
        self.method = method
        super().__init__()

    def _transform(self, X=None, y=None):
        """Merge the Panel data by aligning them temporally.

        Parameters
        ----------
        X : pd.DataFrame
            The input panel data.
        y : pd.Series
            ignored

        Returns
        -------
        returns a single time series
        """
        horizon = X.shape[-1]

        if self.method == "mean":
            result = np.nanmean(self._align_temporal(horizon, X), axis=0)
        elif self.method == "median":
            result = np.nanmedian(self._align_temporal(horizon, X), axis=0)
        else:
            raise ValueError(f"{self.method} must be 'mean' or 'median'.")
        return result

    def _align_temporal(self, horizon, x):
        x = x.astype(float)
        if self.stride > 0:
            x = np.insert(
                x, np.arange(1, x.shape[0]).repeat(self.stride - 1), np.nan, axis=0
            )
        elif self.stride == 0:
            return x
        r = []
        for i in range(horizon):
            _res = np.concatenate(
                [
                    np.full(fill_value=np.nan, shape=(i, x.shape[1])),
                    x[:, :, i],
                    np.full(fill_value=np.nan, shape=((horizon - 1 - i), x.shape[1])),
                ]
            )
            r.append(_res)
        return np.stack(r)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        return [{"method": "mean"}, {"method": "median"}]
