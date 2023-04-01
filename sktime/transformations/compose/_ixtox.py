# -*- coding: utf-8 -*-
"""Use index or hierarchy values as exogeneous features transformer."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["IxToX"]

from sktime.transformations.base import BaseTransformer


class IxToX(BaseTransformer):
    """Create exogeneous features based on time index or hierarchy values.

    Replaces exogeneous features (`X`) by endogeneous data (`y`).

    To *add* instead of *replace*, use `FeatureUnion` and/or the `+` dunder.

    Parameters
    ----------
    coerce_to_type : str, optional, default="float"
        type to coerce the index columns to when passed to `X`
    levels : None (default) or iterable of pandas index level name elements
        if passed, selects the hierarchy levels that will be turned into columns in `X`
        if None, will convert only the time index (last level) into features
    source : str, optional, default="auto"
        which object to take the index from
        default = "auto" = `X` if passed to `fit`/`predict`, otherwise `y` or `fh`
        "y" = `y` as passed to `fit`, from `fh` in `predict`-like methods
        "X" = `X` as passed to `fit`, `predict`, etc
    """

    _tags = {
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": False,
        "univariate-only": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "both",
        "fit_is_empty": True,
        "requires_y": True,
    }

    def __init__(self, subset_index=False):

        self.subset_index = subset_index

        super(IxToX, self).__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : time series or panel in one of the pd.DataFrame formats
            Data to be transformed
        y : time series or panel in one of the pd.DataFrame formats
            Additional data, e.g., labels for transformation

        Returns
        -------
        y, as a transformed version of X
        """
        if self.subset_index:
            return y.loc[X.index.intersection(y.index)]
        else:
            return y

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        Drops featurized column that was added in transform().

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be inverse transformed
        y : Series or Panel of mtype y_inner_mtype, optional (default=None)
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        if self.subset_index:
            return y.loc[X.index.intersection(y.index)]
        else:
            return y
