# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements compositors for performing transformations by group."""

import pandas as pd
from sktime.base._meta import _HeterogenousMetaEstimator
from sktime.datatypes import ALL_TIME_SERIES_MTYPES, mtype_to_scitype
from sktime.registry import coerce_scitype
from sktime.transformations._delegate import _DelegatedTransformer
from sktime.transformations.base import BaseTransformer
from sktime.utils.warnings import warn

__author__ = ["fkiraly"]
__all__ = ["TransformByLevel", "GroupByCategoryTransformer"]


class TransformByLevel(_DelegatedTransformer):
    """Transform by instance or panel.

    Used to apply multiple copies of ``transformer`` by instance or by panel.

    If ``groupby="global"``, behaves like ``transformer``.
    If ``groupby="local"``, fits a clone of ``transformer`` per time series instance.
    If ``groupby="panel"``, fits a clone of ``transformer`` by panel (first non-time
    level).

    The fitted transformers can be accessed in the ``transformers_`` attribute,
    if more than one clone is fitted, otherwise in the ``transformer_`` attribute.

    Parameters
    ----------
    transformer : sktime transformer used in TransformByLevel
        A "blueprint" transformer, state does not change when ``fit`` is called.
    groupby : str, one of ["local", "global", "panel"], optional, default="local"
        level on which data are grouped to fit clones of ``transformer``
        "local" = unit/instance level, one reduced model per lowest hierarchy level
        "global" = top level, one reduced model overall, on pooled data ignoring levels
        "panel" = second lowest level, one reduced model per panel level (-2)
        if there are 2 or less levels, "global" and "panel" result in the same
        if there is only 1 level (single time series), all three settings agree
    raise_warnings : bool, optional, default=True
        whether to warn the user if ``transformer`` is instance-wise
        in this case wrapping the ``transformer`` om ``TransformByLevel`` does not
        change
        the estimator logic, compared to not wrapping it.
        Wrapping this way can make sense in some cases of tuning,
        in which case ``warn=False`` can be set to suppress the warning raised.

    Attributes
    ----------
    transformer_ : sktime transformer, present only if ``groupby`` is "global"
        clone of ``transformer`` used for fitting and transformation
    transformers_ : pd.DataFrame of sktime transformer, present otherwise
        entries are clones of ``transformer`` used for fitting and transformation

    Examples
    --------
    >>> from sktime.transformations.compose import TransformByLevel
    >>> from sktime.transformations.hierarchical.reconcile import Reconciler
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> X = _make_hierarchical()
    >>> f = TransformByLevel(Reconciler(), groupby="panel")
    >>> f.fit(X)
    TransformByLevel(...)
    """

    _tags = {
        "authors": ["fkiraly"],
        "requires-fh-in-fit": False,
        "capability:missing_values": True,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "fit_is_empty": False,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    def __init__(self, transformer, groupby="local", raise_warnings=True):
        self.transformer = transformer
        self.groupby = groupby
        self.raise_warnings = raise_warnings

        self.transformer_ = transformer.clone()

        super().__init__()

        if raise_warnings and self.transformer_.get_tag("scitype:instancewise"):
            warn(
                f"instance of {type(self.transformer_)} passed to TransformByLevel "
                "transforms by instance already, wrapping in TransformByLevel "
                "will not change the estimator logic, compared to not wrapping it.",
                stacklevel=2,
                obj=self,
            )

        self.clone_tags(self.transformer_)
        self.set_tags(**{"fit_is_empty": False})

        if groupby == "local":
            scitypes = ["Series"]
        elif groupby == "global":
            scitypes = ["Series", "Panel", "Hierarchical"]
        elif groupby == "panel":
            scitypes = ["Series", "Panel"]
        else:
            raise ValueError(
                "groupby in TransformByLevel must be one of"
                ' "local", "global", "panel", '
                f"but found {groupby}"
            )

        mtypes = [x for x in ALL_TIME_SERIES_MTYPES if mtype_to_scitype(x) in scitypes]

        # this ensures that we convert in the inner estimator
        # but vectorization/broadcasting happens at the level of groupby
        self.set_tags(**{"X_inner_mtype": mtypes})
        if self.transformer_.get_tag("y_inner_mtype") != "None":
            self.set_tags(**{"y_inner_mtype": mtypes})

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
        params : dict or list of dict
        """
        from sktime.transformations.series.time_since import TimeSince

        groupbys = ["local", "panel", "global"]

        t = TimeSince()

        params = [{"transformer": t, "groupby": g} for g in groupbys]

        return params


class GroupByCategoryTransformer(BaseTransformer, _HeterogenousMetaEstimator):

    def __init__(
        self,
        transformers,
        categorizer=None,
        fallback_transformer=None,
    ):
        self.transformers = transformers
        self.fallback_transformer = fallback_transformer

        if categorizer is not None:
            self.categorizer = categorizer
        else:
            from sktime.transformations.series.adi_cv import ADICVTransformer

            self.categorizer = ADICVTransformer(features=["class"])

        self.categorizer_ = coerce_scitype(self.categorizer, "transformer").clone()

        super().__init__()

        #validate inputs
        for key, transformer in transformers.items():
            if not isinstance(transformer, BaseTransformer):
                raise TypeError(
                    f"transformers[{key}] must be a BaseTransformer, "
                    f"got {type(transformer)}"
                )

        # capability tags from wrapped transformers
        self._set_capability_tags()

    def _set_capability_tags(self):
        """Set capability tags based on wrapped transformers."""
        all_transformers = list(self.transformers.values())
        if self.fallback_transformer is not None:
            all_transformers.append(self.fallback_transformer)

        if len(all_transformers) == 0:
            return

        can_inverse = all(
            t.get_tag("capability:inverse_transform", False) for t in all_transformers
        )
        self.set_tags(**{"capability:inverse_transform": can_inverse})

        handles_missing = all(
            t.get_tag("capability:missing_values", True) for t in all_transformers
        )
        self.set_tags(**{"capability:missing_values": handles_missing})

    def _fit(self, X, y=None):
        """Fit categorizer and category-specific transformers.

        Parameters
        ----------
        X : pd.DataFrame
            Panel data to fit transformers to.
        y : ignored

        Returns
        -------
        self
        """
        self.category_ = self.categorizer_.fit_transform(X=X, y=y).iloc[:, 0]
        self.transformers_ = {}

        if X.index.nlevels == 1:
            self.grouped_by_category_ = [(self.category_.values[0], None)]
        else:
            self.grouped_by_category_ = self.category_.groupby(self.category_)

        for category, group in self.grouped_by_category_:
            if category in self.transformers:
                transformer = self.transformers[category].clone()
            elif self.fallback_transformer is not None:
                transformer = self.fallback_transformer.clone()
            else:
                raise ValueError(
                    f"No transformer for category '{category}' and no "
                    "fallback_transformer given."
                )

            X_category = self._loc_group(X, group)
            transformer.fit(X=X_category, y=y)
            self.transformers_[category] = transformer

        return self

    def _transform(self, X, y=None):
        """Transform X using category-specific transformers.

        Parameters
        ----------
        X : pd.DataFrame
            Panel data to transform.
        y : ignored

        Returns
        -------
        Xt : pd.DataFrame
            Transformed data.
        """
        transformed_parts = []

        for category, group in self.grouped_by_category_:
            X_category = self._loc_group(X, group)
            transformer = self.transformers_[category]
            Xt_category = transformer.transform(X=X_category, y=y)
            transformed_parts.append(Xt_category)

        Xt = pd.concat(transformed_parts, axis=0).sort_index()
        return Xt

    def _inverse_transform(self, X, y=None):
        """transform X using category transformers.

        Only available if all wrapped transformers support inverse_transform.

        Parameters
        ----------
        X : pd.DataFrame
            Data to inverse transform.
        y : ignored

        Returns
        -------
        Xi : pd.DataFrame
            Inverse transformed data.
        """
        inverse_parts = []

        for category, group in self.grouped_by_category_:
            X_category = self._loc_group(X, group)
            transformer = self.transformers_[category]
            Xi_category = transformer.inverse_transform(X=X_category, y=y)
            inverse_parts.append(Xi_category)

        Xi = pd.concat(inverse_parts, axis=0).sort_index()
        return Xi

    def _loc_group(self, df, group):
        """Subset DataFrame to rows matching group.

        Parameters
        ----------
        df : pd.DataFrame
            Data to subset.
        group : pd.DataFrame or None
            Group indices. None means return full df.

        Returns
        -------
        pd.DataFrame
            Subset of df matching group.
        """
        if df is None or group is None:
            return df
        return df.loc[df.index.droplevel(-1).isin(group.index)]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : list of dict
            Test parameter configurations.
        """
        from sktime.transformations.series.adi_cv import ADICVTransformer
        from sktime.transformations.series.difference import Differencer

        param1 = {
            "transformers": {
                "smooth": Differencer(),
                "erratic": Differencer(lags=1),
                "intermittent": Differencer(),
                "lumpy": Differencer(),
            },
            "categorizer": ADICVTransformer(features=["class"]),
            "fallback_transformer": None,
        }

        param2 = {
            "transformers": {},
            "categorizer": ADICVTransformer(features=["class"]),
            "fallback_transformer": Differencer(),
        }

        return [param1, param2]
