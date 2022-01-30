# -*- coding: utf-8 -*-
"""Test scenarios for transformers.

Contains TestScenario concrete children to run in tests for transformers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_transformers"]

from copy import deepcopy
from inspect import isclass

import pandas as pd

from sktime.base import BaseObject
from sktime.datatypes import mtype_to_scitype
from sktime.transformations.base import (
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)
from sktime.utils._testing.estimator_checks import _make_primitives, _make_tabular_X
from sktime.utils._testing.forecasting import _make_series
from sktime.utils._testing.panel import _make_classification_y, _make_panel_X
from sktime.utils._testing.scenarios import TestScenario

OLD_MIXINS = (
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)

OLD_PANEL_MIXINS = (
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
)

OLD_SERIES_MIXINS = (
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


def _is_child_of(obj, class_or_tuple):
    """Shorthand for 'inherits from', obj can be class or object."""
    if isclass(obj):
        return issubclass(obj, class_or_tuple)
    else:
        return isinstance(obj, class_or_tuple)


def get_tag(obj, tag_name):
    """Shorthand for get_tag vs get_class_tag, obj can be class or object."""
    if isclass(obj):
        return obj.get_class_tag(tag_name)
    else:
        return obj.get_tag(tag_name)


class TransformerTestScenario(TestScenario, BaseObject):
    """Generic test scenario for transformers."""

    def is_applicable(self, obj):
        """Check whether scenario is applicable to obj.

        Parameters
        ----------
        obj : class or object to check against scenario

        Returns
        -------
        applicable: bool
            True if self is applicable to obj, False if not
        """
        # pre-refactor classes can't deal with Series *and* Panel both
        X_scitype = self.get_tag("X_scitype")

        if _is_child_of(obj, OLD_PANEL_MIXINS) and X_scitype != "Panel":
            return False

        if _is_child_of(obj, OLD_SERIES_MIXINS) and X_scitype != "Series":
            return False

        # applicable only if number of variables in y complies with scitype:y
        is_univariate = self.get_tag("X_univariate")

        if not is_univariate and get_tag(obj, "univariate-only"):
            return False

        # the case that we would need to vectorize with y, skip
        X_inner_mtype = get_tag(obj, "X_inner_mtype")
        X_inner_scitypes = mtype_to_scitype(X_inner_mtype, return_unique=True)
        if not isinstance(X_inner_scitypes, list):
            X_inner_scitypes = [X_inner_scitypes]
        # we require vectorization from of a Series trafo to Panel data ...
        if X_scitype == "Panel" and "Panel" not in X_inner_scitypes:
            # ... but y is passed and y is not ignored internally ...
            if self.get_tag("has_y") and get_tag(obj, "y_inner_mtype") != "None":
                # ... this would raise an error since vectorization is not defined
                return False

        # only applicable if X of supported index type
        X = self.args["fit"]["X"]
        supported_idx_types = get_tag(obj, "enforce_index_type")
        if isinstance(X, (pd.Series, pd.DataFrame)) and supported_idx_types is not None:
            if type(X.index) not in get_tag(obj, "enforce_index_type"):
                return False

        return True

    def get_args(self, key, obj=None, deepcopy_args=False):
        """Return args for key. Can be overridden for dynamic arg generation.

        If overridden, must not have any side effects on self.args
            e.g., avoid assignments args[key] = x without deepcopying self.args first

        Parameters
        ----------
        key : str, argument key to construct/retrieve args for
        obj : obj, optional, default=None. Object to construct args for.
        deepcopy_args : bool, optional, default=True. Whether to deepcopy return.

        Returns
        -------
        args : argument dict to be used for a method, keyed by `key`
            names for keys need not equal names of methods these are used in
                but scripted method will look at key with same name as default
        """
        if key == "inverse_transform":
            if obj is None:
                raise ValueError('if key="inverse_transform", obj must be provided')

            X_scitype = self.get_tag("X_scitype")

            X_out_scitype = get_tag(obj, "scitype:transform-output")
            X_panel = get_tag(obj, "scitype:instancewise")

            X_out_series = X_out_scitype == "Series"
            X_out_prim = X_out_scitype == "Primitives"

            # determine output by X_out_scitype
            #   until transformer refactor is complete, use the old classes, too
            if _is_child_of(obj, OLD_MIXINS):
                s2s = _is_child_of(obj, _SeriesToSeriesTransformer)
                s2p = _is_child_of(obj, _SeriesToPrimitivesTransformer)
                p2t = _is_child_of(obj, _PanelToTabularTransformer)
                p2p = _is_child_of(obj, _PanelToPanelTransformer)
            else:
                s2s = X_scitype == "Series" and X_out_series
                s2p = X_scitype == "Series" and X_out_prim
                p2t = X_scitype == "Panel" and X_out_prim
                p2p = X_scitype == "Panel" and X_out_series

            # expected input type of inverse_transform is expected output of transform
            if s2p:
                args = {"X": _make_primitives(random_state=RAND_SEED)}
            elif s2s:
                args = {"X": _make_series(n_timepoints=20, random_state=RAND_SEED)}
            elif p2t:
                args = {"X": _make_tabular_X(n_instances=7, random_state=RAND_SEED)}
            elif p2p:
                args = {
                    "X": _make_panel_X(
                        n_instances=7, n_timepoints=20, random_state=RAND_SEED
                    )
                }
            else:
                raise RuntimeError(
                    "transformer with unexpected combination of tags: "
                    f"X_out_scitype = {X_out_scitype}, scitype:instancewise = {X_panel}"
                )

        else:
            # default behaviour, happens except when key = "inverse_transform"
            args = self.args[key]

        if deepcopy_args:
            args = deepcopy(args)

        return args


class TransformerFitTransformSeriesUnivariate(TransformerTestScenario):
    """Fit/transform, univariate Series X."""

    _tags = {
        "X_scitype": "Series",
        "X_univariate": True,
        "has_y": False,
        "pre-refactor": True,
    }

    args = {
        "fit": {"X": _make_series(n_timepoints=20, random_state=RAND_SEED)},
        "transform": {"X": _make_series(n_timepoints=10, random_state=RAND_SEED)},
        # "inverse_transform": {"X": _make_series(n_timepoints=10)},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformSeriesMultivariate(TransformerTestScenario):
    """Fit/transform, multivariate Series X."""

    _tags = {"X_scitype": "Series", "X_univariate": False, "has_y": False}

    args = {
        "fit": {
            "X": _make_series(n_columns=2, n_timepoints=20, random_state=RAND_SEED)
        },
        "transform": {
            "X": _make_series(n_columns=2, n_timepoints=10, random_state=RAND_SEED)
        },
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelUnivariate(TransformerTestScenario):
    """Fit/transform, univariate Panel X."""

    _tags = {"X_scitype": "Panel", "X_univariate": True, "has_y": False}

    args = {
        "fit": {
            "X": _make_panel_X(
                n_instances=7, n_columns=1, n_timepoints=20, random_state=RAND_SEED
            )
        },
        "transform": {
            "X": _make_panel_X(
                n_instances=3, n_columns=1, n_timepoints=10, random_state=RAND_SEED
            )
        },
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelMultivariate(TransformerTestScenario):
    """Fit/transform, multivariate Panel X."""

    _tags = {"X_scitype": "Panel", "X_univariate": False, "has_y": False}

    args = {
        "fit": {"X": _make_panel_X(n_instances=7, n_columns=2, n_timepoints=20)},
        "transform": {"X": _make_panel_X(n_instances=7, n_columns=2, n_timepoints=20)},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerFitTransformPanelUnivariateWithClassY(TransformerTestScenario):
    """Fit/transform, multivariate Panel X."""

    _tags = {
        "X_scitype": "Panel",
        "X_univariate": True,
        "pre-refactor": True,
        "has_y": True,
        "y_scitype": "classes",
    }

    args = {
        "fit": {
            "X": _make_panel_X(n_instances=7, n_columns=1, n_timepoints=20),
            "y": _make_classification_y(n_instances=7, n_classes=2),
        },
        "transform": {"X": _make_panel_X(n_instances=7, n_columns=1, n_timepoints=20)},
    }
    default_method_sequence = ["fit", "transform"]


# todo: scenario for Panel X
#   where test and training set has different n_instances or n_timepoints
#   may need a tag that tells us whethe transformer can cope with this


scenarios_transformers = [
    TransformerFitTransformSeriesUnivariate,
    TransformerFitTransformSeriesMultivariate,
    TransformerFitTransformPanelUnivariate,
    TransformerFitTransformPanelMultivariate,
    TransformerFitTransformPanelUnivariateWithClassY,
]
