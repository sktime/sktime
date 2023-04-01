# -*- coding: utf-8 -*-
"""Use index or hierarchy values as exogeneous features transformer."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["IxToX"]

from sktime.transformations.base import BaseTransformer


class IxToX(BaseTransformer):
    """Create exogeneous features based on time index or hierarchy values.

    Returns index of `X` in `transform` as transformed features.
    By default, time features only.
    Can also be used to select hierarchy levels in case of hierarchical input,
    via the `levels` argument.

    Return columns of `transform` applied to `pandas` based containers
    have same name as level if levels have name in `transform` input,
    otherwise `index` (time) and `level_X.{N}` where N is the level index integer.

    To *add* instead of *replace*, use `FeatureUnion` and/or the `+` dunder.

    For more custom options or a direct `pandas` interface,
    an alternative is `PandasTransformAdaptor` with `method="reset_index"`.

    Parameters
    ----------
    coerce_to_type : str, optional, default="float"
        type to coerce the index columns to when passed to `X`
    level : None (default), int, str, or iterable of pandas index level name elements
        if passed, selects the hierarchy levels that will be turned into columns in `X`
        if passed, passed on as `level` to `reset_index` internally
        if None, will convert only the time index (last level) into features
        Note that this is different from the default of `reset_index`
    ix_source : str, optional, default="X"
        which object to take the index from
        default = "X" = `X` as passed to `transform`
            if used within `ForecastingPipeline`, this means `X` by default
        "y" = `y` as passed to `transform`, if passed (not `None`), otherwise `X`
    """

    _tags = {
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": False,
        "univariate-only": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "both",
        "fit_is_empty": True,
        "requires_y": False,
    }

    def __init__(self, coerce_to_type="float", level=None, ix_source="X"):

        self.coerce_to_type = coerce_to_type
        self.level = level
        self.ix_source = ix_source

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
        transformed version of X
        """
        if self.ix_source == "y" and y is not None:
            X = y

        level = self.level
        if level is None:
            level = -1

        X_only_ix = X.drop(columns=X.columns)
        X_ix_in_df = X_only_ix.reset_index(level=level)

        if X.index.names[-1] is None:
            newcols = X_ix_in_df.columns
            newcols[-1] = "index"
            X_ix_in_df.columns = newcols

        X_ix_in_df.index = X.index
        X_ix_in_df = X_ix_in_df.astype(self.coerce_to_type)

        return X_ix_in_df

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
        return X
