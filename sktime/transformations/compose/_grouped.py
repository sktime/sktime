# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements compositors for performing transformations by group."""

from sktime.datatypes import ALL_TIME_SERIES_MTYPES, mtype_to_scitype
from sktime.transformations._delegate import _DelegatedTransformer

__author__ = ["fkiraly"]
__all__ = ["TransformByLevel"]


class TransformByLevel(_DelegatedTransformer):
    """Transform by instance or panel.

    Used to apply multiple copies of `transformer` by instance or by panel.

    If `groupby="global"`, behaves like `transformer`.
    If `groupby="local"`, fits a clone of `transformer` per time series instance.
    If `groupby="panel"`, fits a clone of `transformer` by panel (first non-time level).

    The fitted transformers can be accessed in the `transformers_` attribute,
    if more than one clone is fitted, otherwise in the `transformer_` attribute.

    Parameters
    ----------
    transformer : sktime transformer used in TransformByLevel
        A "blueprint" transformer, state does not change when `fit` is called.
    groupby : str, one of ["local", "global", "panel"], optional, default="local"
        level on which data are grouped to fit clones of `transformer`
        "local" = unit/instance level, one reduced model per lowest hierarchy level
        "global" = top level, one reduced model overall, on pooled data ignoring levels
        "panel" = second lowest level, one reduced model per panel level (-2)
        if there are 2 or less levels, "global" and "panel" result in the same
        if there is only 1 level (single time series), all three settings agree

    Attributes
    ----------
    transformer_ : sktime transformer, present only if `groupby` is "global"
        clone of `transformer` used for fitting and transformation
    transformers_ : pd.DataFrame of sktime transformer, present otherwise
        entries are clones of `transformer` used for fitting and transformation

    Examples
    --------
    >>> from sktime.transformations.hierarchical.reconcile import Reconciler
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> X = _make_hierarchical()
    >>> f = TransformByLevel(Reconciler(), groupby="panel")
    >>> f.fit(X)
    TransformByLevel(...)
    """

    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": True,
        "X_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "y_inner_mtype": ALL_TIME_SERIES_MTYPES,
        "fit_is_empty": False,
    }

    # attribute for _DelegatedTrafoer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTrafoer docstring
    _delegate_name = "transformer_"

    def __init__(self, transformer, groupby="local"):

        self.transformer = transformer
        self.groupby = groupby

        self.transformer_ = transformer.clone()

        super(TransformByLevel, self).__init__()

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
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.transformations.series.time_since import TimeSince

        groupbys = ["local", "panel", "global"]

        t = TimeSince()

        params = [{"transformer": t, "groupby": g} for g in groupbys]

        return params
