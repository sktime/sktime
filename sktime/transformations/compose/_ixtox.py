"""Use index or hierarchy values as features transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["IxToX"]

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from sktime.transformations.base import BaseTransformer


class IxToX(BaseTransformer):
    """Create features based on time index or hierarchy values.

    Returns index of ``X`` in ``transform`` as transformed features.
    By default, time features only.
    Can also be used to select hierarchy levels in case of hierarchical input,
    via the ``levels`` argument.

    Return columns of ``transform`` applied to ``pandas`` based containers
    have same name as level if levels have name in ``transform`` input,
    otherwise ``index`` (time) and ``level_{N}`` where N is the level index integer.

    To *add* instead of *replace*, use ``FeatureUnion`` and/or the ``+`` dunder.

    Under the default setting of ``coerce_to_type="auto"``:

    * date-like indices incl periods are coerced to float (via int64)
      this typically results in units of periods since start of 1970 (first = 0)
    * object, string, and category indices are coerced to integer (unique category ID)
      mapping onto integers is per category levels, after ``pandas`` category coercion

    For more custom options or a direct ``pandas`` interface,
    an alternative is ``PandasTransformAdaptor`` with ``method="reset_index"``.

    Parameters
    ----------
    coerce_to_type : str, or dict, optional, default="auto"
        how to coerce the index columns to when passed to ``X``
        default="auto" coerces:
        date-like indices to float (via int64)
        object, string, and category indices to integer
        values other than "auto" are passed to ``DataFrame.astype`` in ``transform``
    level : None (default), int, str, or iterable of pandas index level name elements
        if passed, selects the hierarchy levels that will be turned into columns in
        ``X``
        if passed, passed on as ``level`` to ``reset_index`` internally
        if None, will convert only the time index (last level) into features
        Note that this is different from the default of ``reset_index``
    ix_source : str, optional, default="X"
        which object to take the index from
        default = "X" = ``X`` as passed to ``transform``
            if used within ``ForecastingPipeline``, this means ``X`` by default
        "y" = ``y`` as passed to ``transform``, if passed (not ``None``), otherwise
        ``X``

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.compose import IxToX
    >>>
    >>> X = load_airline()
    >>> t = IxToX()
    >>> Xt = t.fit_transform(X)
    """

    _tags = {
        "authors": "fkiraly",
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": False,
        "univariate-only": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "both",
        "fit_is_empty": True,
        "requires_y": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(self, coerce_to_type="auto", level=None, ix_source="X"):
        self.coerce_to_type = coerce_to_type
        self.level = level
        self.ix_source = ix_source

        super().__init__()

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
        coerce_to_type = self.coerce_to_type
        ix_source = self.ix_source
        level = self.level

        def is_date_like(x):
            return is_datetime64_any_dtype(x) or isinstance(x, pd.PeriodDtype)

        if ix_source == "y" and y is not None:
            X = y

        if level is None:
            level = -1

        X_only_ix = X.drop(columns=X.columns)

        if X.index.names[-1] is None:
            newcols = list(X_only_ix.index.names)
            newcols[-1] = "index"
            X_only_ix.index.names = newcols

        X_ix_in_df = X_only_ix.reset_index(level=level)

        X_ix_in_df.index = X.index

        if coerce_to_type == "auto":
            cd = {col: X_ix_in_df.dtypes[col] for col in X_ix_in_df.columns}
            coerce1 = {d: "int64" for d in cd if is_date_like(cd[d])}
            coerce2 = {d: "float64" for d in cd if is_date_like(cd[d])}
            X_ix_in_df = X_ix_in_df.astype(coerce1)
            X_ix_in_df = X_ix_in_df.astype(coerce2)
        else:
            X_ix_in_df = X_ix_in_df.astype(coerce_to_type)

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        params1 = {}
        params2 = {"level": -1}

        return [params1, params2]
