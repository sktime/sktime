# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements compositors for performing transformations by group."""

import pandas as pd

from sktime.base._meta import _HeterogenousMetaEstimator
from sktime.datatypes import ALL_TIME_SERIES_MTYPES, mtype_to_scitype
from sktime.registry import coerce_scitype
from sktime.transformations.base import BaseTransformer, _DelegatedTransformer
from sktime.utils.warnings import warn

__author__ = ["fkiraly", "aminmiral"]
__all__ = ["TransformByLevel", "GroupbyCategoryTransformer"]


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
        from sktime.transformations.time_since import TimeSince

        groupbys = ["local", "panel", "global"]

        t = TimeSince()

        params = [{"transformer": t, "groupby": g} for g in groupbys]

        return params


class GroupbyCategoryTransformer(BaseTransformer, _HeterogenousMetaEstimator):
    """Apply different transformers by time series category or cluster.

    Programmatic generalization of "categorize, then transform by category"
    approach. Applies a series-to-primitives transformer or clusterer
    (the ``categorizer``) to assign a category to each time series instance
    in a panel, then applies the transformer corresponding to that category.

    Different from ``TransformByLevel``, which groups instances by hierarchy
    or panel level, this compositor groups instances by a data-driven
    category label generated by ``categorizer``.

    Parameters
    ----------
    transformers : dict[str, sktime transformer]
        dict of transformers with the key corresponding to categories
        generated by ``categorizer``, and the value the transformer applied
        to instances of that category.

    categorizer : sktime transformer or clusterer, default=ADICVTransformer()
        A series-to-primitives sktime transformer that generates a value
        which is used to assign a category to each time series instance.

        If a clusterer is used, it must support cluster assignment, i.e.,
        have the ``capability:predict`` tag.

        Note: to ensure correct functionality, the categorizer must store
        the generated category in the first column of the value returned
        by ``transform()``/``fit_transform()``.

    fallback_transformer : sktime transformer or None, optional, default=None
        Transformer used for instances whose category, as generated by
        ``categorizer``, has no matching entry in ``transformers``.
        If ``None`` and such a category is encountered, raises ``ValueError``.

    Attributes
    ----------
    categorizer_ : sktime transformer
        fitted clone of ``categorizer``.
    transformers_ : dict[str, sktime transformer]
        fitted clones of the transformers in ``transformers`` (or of
        ``fallback_transformer``), keyed by category seen during ``fit``.
    fallback_transformer_ : sktime transformer or None
        fitted clone of ``fallback_transformer`` on the whole of the ``fit``
        data, regardless of category. Used at ``transform``/``inverse_transform``
        time for a category that was not encountered at all during ``fit``,
        since no category-specific fallback state exists to use for it.
        ``None`` if ``fallback_transformer`` was not provided.

    Examples
    --------
    >>> from sktime.transformations.compose import GroupbyCategoryTransformer
    >>> from sktime.transformations.adi_cv import ADICVTransformer
    >>> from sktime.transformations.difference import Differencer
    >>> from sktime.transformations.exponent import ExponentTransformer
    >>> from sktime.transformations.tests.test_adi_cv import (
    ...     _generate_erratic_series)

    >>> transformer = GroupbyCategoryTransformer(
    ...     transformers={
    ...         "smooth": Differencer(),
    ...         "erratic": ExponentTransformer(),
    ...     },
    ...     categorizer=ADICVTransformer(features=["class"]),
    ...     fallback_transformer=Differencer(),
    ... )

    >>> X = _generate_erratic_series()
    >>> transformer = transformer.fit(X)
    >>> Xt = transformer.transform(X)
    """

    _tags = {
        "authors": ["aminmiral"],
        "maintainers": ["aminmiral"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": "None",
        "capability:inverse_transform": False,  # overridden in __dynamic_tags__
        "fit_is_empty": False,
        "visual_block_kind": "parallel",
    }

    def __init__(self, transformers, categorizer=None, fallback_transformer=None):
        self.transformers = transformers
        self.categorizer = categorizer
        self.fallback_transformer = fallback_transformer

        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        all_transformers = list(self.transformers.values())
        if self.fallback_transformer is not None:
            all_transformers.append(self.fallback_transformer)

        can_inverse = len(all_transformers) > 0 and all(
            t.get_tag("capability:inverse_transform", False) for t in all_transformers
        )
        self.set_tags(**{"capability:inverse_transform": can_inverse})

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor

        IMPORTANT: no significant compute or memory use should happen in
        __post_init__, memory and compute intensive operations should be
        in _fit, not __post_init__.
        """
        categorizer = self.categorizer

        if categorizer is not None:
            _categorizer = categorizer.clone()
        else:
            from sktime.transformations.adi_cv import ADICVTransformer

            _categorizer = ADICVTransformer(features=["class"])

        self.categorizer_ = coerce_scitype(_categorizer, "transformer").clone()

        for key, transformer in self.transformers.items():
            if not isinstance(transformer, BaseTransformer):
                raise TypeError(
                    f"transformers['{key}'] must be an sktime BaseTransformer, "
                    f"found {type(transformer)}"
                )

        fallback = self.fallback_transformer
        if fallback is not None and not isinstance(fallback, BaseTransformer):
            raise TypeError(
                "fallback_transformer must be an sktime BaseTransformer, "
                f"found {type(fallback)}"
            )

    @property
    def _steps(self):
        return [self._coerce_estimator_tuple(self.categorizer)] + self._transformers

    @property
    def steps_(self):
        return [self._coerce_estimator_tuple(self.categorizer_)] + self._transformers

    @property
    def _transformers(self):
        """Provide an internal list of the transformers available.

        Each list item is a tuple of the format (category, transformer)
        where category is the category for which the respective transformer
        is chosen, and transformer the value for each tuple. Also includes
        the fallback transformer under the key "fallback_transformer".

        Returns
        -------
        transformers : list[tuple[str, sktime transformer]]
        """
        return list(self.transformers.items()) + [
            ("fallback_transformer", self.fallback_transformer)
        ]

    @_transformers.setter
    def _transformers(self, new_transformers):
        """Set new values for the transformers, e.g., via set_params.

        Parameters
        ----------
        new_transformers : list[tuple[str, sktime transformer]]
        """
        for category, transformer in new_transformers:
            if category != "fallback_transformer":
                self.transformers[category] = transformer
            else:
                self.fallback_transformer = transformer

    def _group_by_category(self, X, category):
        """Group the instance index of X by category label.

        Handles the case where X is a single series without a panel level,
        in which case there is exactly one group, spanning all of X.

        Parameters
        ----------
        X : pd.DataFrame
            Data whose instance index is to be grouped.
        category : pd.Series
            Category label per instance in X, as generated by ``categorizer_``.

        Returns
        -------
        list of tuple (category, group)
            group is a pd.Series that can be passed to ``_loc_group``,
            or None if X has no panel level.
        """
        if X.index.nlevels == 1:
            return [(category.values[0], None)]
        return list(category.groupby(category))

    def _get_transformer_for_category(self, category, fitted):
        """Look up the (fitted or blueprint) transformer for a category.

        Parameters
        ----------
        category : str
            The category to look up a transformer for.
        fitted : bool
            If True, look up in the fitted ``transformers_`` dict, raising if
            the category was not seen during ``fit``.
            If False, look up in the ``transformers`` blueprint dict,
            falling back to ``fallback_transformer`` if not found.

        Returns
        -------
        sktime transformer
        """
        if fitted:
            if category in self.transformers_:
                return self.transformers_[category]
            if self.fallback_transformer_ is not None:
                return self.fallback_transformer_
            raise ValueError(
                f"category '{category}' was encountered, but was not seen "
                "during fit, so there is no fitted transformer, fallback "
                "or otherwise, available for it."
            )

        if category in self.transformers:
            return self.transformers[category]
        if self.fallback_transformer is not None:
            return self.fallback_transformer
        raise ValueError(
            f"no transformer was provided for category '{category}', and "
            "no fallback_transformer was given to fall back on."
        )

    def _fit(self, X, y=None):
        """Fit categorizer and category-specific transformers to X.

        private _fit containing the core logic, called from fit

        For the _fit function to work as intended, the categorizer
        must generate and store the extrapolated category in the
        first column.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.DataFrame
            Data to fit the categorizer and transformers to.
        y : pd.DataFrame, optional (default=None)
            Additional data, e.g., labels for transformation.

        Returns
        -------
        self : reference to self

        Raises
        ------
        ValueError : if a category generated by the categorizer has no
            matching entry in ``transformers``, and no ``fallback_transformer``
            was provided.
        """
        self.category_ = self.categorizer_.fit_transform(X=X, y=y).iloc[:, 0]
        self.transformers_ = {}

        # fit a single generic fallback instance on the whole of X, regardless
        # of category. This is what is used at transform/inverse_transform time
        # for a category that was not seen at all during fit - there is no
        # data to have fit a fallback specifically for it, so this generic
        # instance is the closest available fitted state, and covers the
        # fallback_transformer contract without fitting anything at
        # transform time.
        if self.fallback_transformer is not None:
            self.fallback_transformer_ = self.fallback_transformer.clone()
            self.fallback_transformer_.fit(X=X, y=y)
        else:
            self.fallback_transformer_ = None

        for category, group in self._group_by_category(X, self.category_):
            transformer = self._get_transformer_for_category(
                category, fitted=False
            ).clone()

            X_category = self._loc_group(X, group)
            y_category = self._loc_group(y, group)
            transformer.fit(X=X_category, y=y_category)

            self.transformers_[category] = transformer

        return self

    def _transform(self, X, y=None):
        """Transform X using the category-specific fitted transformers.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            Data to be transformed.
        y : pd.DataFrame, optional (default=None)
            Additional data, e.g., labels for transformation.

        Returns
        -------
        Xt : pd.DataFrame
            transformed version of X
        """
        return self._iterate_over_categories("transform", X, y=y)

    def _inverse_transform(self, X, y=None):
        """Inverse transform X using the category-specific fitted transformers.

        private _inverse_transform containing core logic, called from
        inverse_transform. Only available if all wrapped transformers
        (including fallback_transformer, if given) support inverse_transform.

        Parameters
        ----------
        X : pd.DataFrame
            Data to be inverse transformed.
        y : pd.DataFrame, optional (default=None)
            Additional data, e.g., labels for transformation.

        Returns
        -------
        Xit : pd.DataFrame
            inverse transformed version of X
        """
        return self._iterate_over_categories("inverse_transform", X, y=y)

    def _iterate_over_categories(self, methodname, X, y=None):
        """Re-categorize X and dispatch methodname to the fitted transformers.

        Re-categorizes the passed X using the already-fitted ``categorizer_``,
        rather than reusing the grouping computed during fit. This is what
        allows transform/inverse_transform to be called correctly on data
        different from what was seen in fit (e.g., held-out or new data),
        instead of silently replaying fit-time instance groups against it.

        Parameters
        ----------
        methodname : str
            The name of the method to call on each fitted transformer, one
            of "transform", "inverse_transform".
        X : pd.DataFrame
            The data to categorize and transform.
        y : pd.DataFrame, optional (default=None)
            Additional data, e.g., labels for transformation.

        Returns
        -------
        pd.DataFrame
            The result of calling methodname on X, category by category.
        """
        category = self.categorizer_.transform(X=X, y=y).iloc[:, 0]

        results = []
        for cat, group in self._group_by_category(X, category):
            transformer = self._get_transformer_for_category(cat, fitted=True)

            X_category = self._loc_group(X, group)
            y_category = self._loc_group(y, group)
            results.append(getattr(transformer, methodname)(X=X_category, y=y_category))

        result = pd.concat(results, axis=0).sort_index()
        return result

    def _loc_group(self, df, group):
        """Return the rows of df belonging to group.

        Parameters
        ----------
        df : pd.DataFrame or None
            The dataframe to locate the group in.
        group : pd.Series or None
            The group to locate in the dataframe. If None, or if df is None,
            df is returned unchanged - this covers the case of X without a
            panel level, and the case of y not being provided.

        Returns
        -------
        pd.DataFrame or None
            The rows of df that match the given group.
        """
        if df is None or group is None:
            return df
        return df.loc[df.index.droplevel(-1).map(lambda x: x in group.index),]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.
            If no special parameters are defined for a value, will return
            the ``"default"`` set.

        Returns
        -------
        params : list of dict
        """
        from sktime.transformations.adi_cv import ADICVTransformer
        from sktime.transformations.exponent import ExponentTransformer

        # note: Differencer is deliberately not used here. It only supports
        # being transformed on the same instances it was fit on (by design,
        # see its own docstring), which the generic estimator test suite
        # violates by fitting and transforming on differently-sized panels.
        # ExponentTransformer has no such restriction. A fallback_transformer
        # is provided in both parameter sets, since the categories generated
        # by ADICVTransformer on the test suite's generated panels are not
        # guaranteed to be identical between the fit-time and transform-time
        # data.
        param1 = {
            "transformers": {
                "smooth": ExponentTransformer(power=2),
                "erratic": ExponentTransformer(power=0.5),
                "intermittent": ExponentTransformer(power=3),
                "lumpy": ExponentTransformer(power=1),
            },
            "categorizer": ADICVTransformer(features=["class"]),
            "fallback_transformer": ExponentTransformer(power=1),
        }

        # exercising the case where no categories are mapped up front, and
        # everything is routed through the fallback_transformer
        param2 = {
            "transformers": {},
            "categorizer": ADICVTransformer(features=["class"]),
            "fallback_transformer": ExponentTransformer(power=2),
        }

        return [param1, param2]
