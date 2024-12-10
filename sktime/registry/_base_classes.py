"""Register of estimator base classes corresponding to sktime scitypes.

This module exports the following:

---

BASE_CLASS_REGISTER - list of tuples

each tuple corresponds to a base class, elements as follows:
    0 : string - scitype shorthand
    1 : type - the base class itself
    2 : string - plain English description of the scitype

---

TRANSFORMER_MIXIN_REGISTER - list of tuples

each tuple corresponds to a transformer mixin, elements as follows:
    0 : string - scitype shorthand
    1 : type - the transformer mixin itself
    2 : string - plain English description of the scitype

---

BASE_CLASS_SCITYPE_LIST - list of string
    elements are 0-th entries of BASE_CLASS_REGISTER, in same order

---

BASE_CLASS_LIST - list of classes
    elements are 1-st entries of BASE_CLASS_REGISTER, in same order

---

BASE_CLASS_LOOKUP - dictionary
    keys/entries are 0/1-th entries of BASE_CLASS_REGISTER

---

TRANSFORMER_MIXIN_SCITYPE_LIST - list of string
    elements are 0-th entries of TRANSFORMER_MIXIN_REGISTER, in same order

---

TRANSFORMER_MIXIN_LIST - list of string
    elements are 1-st entries of TRANSFORMER_MIXIN_REGISTER, in same order

---

TRANSFORMER_MIXIN_LOOKUP - dictionary
    keys/entries are 0/1-th entries of TRANSFORMER_MIXIN_REGISTER
"""

import inspect
import sys
from functools import lru_cache

from sktime.base import BaseObject


class _BaseScitypeOfObject(BaseObject):
    """Base class for all object scitypes."""

    _tags = {
        "object_type": "scitype:object",
        "scitype_name": "fill_this_in",  # value if used for object_type
        "parent_scitype": None,  # parent scitype, for scitype inheritance
        "short_descr": "describe the scitype here",  # short description, max 80 chars
        "mixin": False,  # whether this is a mixin, not full scitype
    }


class object(_BaseScitypeOfObject):
    """Universal type for all objects."""

    _tags = {
        "scitype_name": "object",
        "short_descr": "base scitype for all objects",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.base import BaseObject

        return BaseObject


class estimator(_BaseScitypeOfObject):
    """Estimator objects, i.e., objects with fit method."""

    _tags = {
        "scitype_name": "estimator",
        "short_descr": "estimator = object with fit",
        "parent_scitype": "object",  # parent scitype, for scitype inheritance
    }

    @classmethod
    def get_base_class(cls):
        from sktime.base import BaseEstimator

        return BaseEstimator


class aligner(_BaseScitypeOfObject):
    """Time series aligner or sequence aligner."""

    _tags = {
        "scitype_name": "aligner",
        "short_descr": "time series aligner or sequence aligner",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.alignment.base import BaseAligner

        return BaseAligner


class classifier(_BaseScitypeOfObject):
    """Time series classifier."""

    _tags = {
        "scitype_name": "classifier",
        "short_descr": "time series classifier",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.classification.base import BaseClassifier

        return BaseClassifier


class clusterer(_BaseScitypeOfObject):
    """Time series clusterer."""

    _tags = {
        "scitype_name": "clusterer",
        "short_descr": "time series clusterer",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.clustering.base import BaseClusterer

        return BaseClusterer


class early_classifier(_BaseScitypeOfObject):
    """Early time series classifier."""

    _tags = {
        "scitype_name": "early_classifier",
        "short_descr": "early time series classifier",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.classification.early_classification import BaseEarlyClassifier

        return BaseEarlyClassifier


class forecaster(_BaseScitypeOfObject):
    """Time series forecaster."""

    _tags = {
        "scitype_name": "forecaster",
        "short_descr": "time series forecaster",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.forecasting.base import BaseForecaster

        return BaseForecaster


class global_forecaster(_BaseScitypeOfObject):
    """Global time series forecaster."""

    _tags = {
        "scitype_name": "global_forecaster",
        "short_descr": "global time series forecaster",
        "parent_scitype": "forecaster",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.forecasting.base import _BaseGlobalForecaster

        return _BaseGlobalForecaster


class metric(_BaseScitypeOfObject):
    """Performance metric for time series."""

    _tags = {
        "scitype_name": "metric",
        "short_descr": "performance metric",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.performance_metrics.base import BaseMetric

        return BaseMetric


class network(_BaseScitypeOfObject):
    """Deep learning network for time series."""

    _tags = {
        "scitype_name": "network",
        "short_descr": "deep learning network",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.networks.base import BaseDeepNetwork

        return BaseDeepNetwork


class param_est(_BaseScitypeOfObject):
    """Parameter fitting estimator."""

    _tags = {
        "scitype_name": "param_est",
        "short_descr": "parameter fitting estimator",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.param_est.base import BaseParamFitter

        return BaseParamFitter


class regressor(_BaseScitypeOfObject):
    """Time series regressor."""

    _tags = {
        "scitype_name": "regressor",
        "short_descr": "time series regressor",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.regression.base import BaseRegressor

        return BaseRegressor


class detector(_BaseScitypeOfObject):
    """Detector of anomalies, outliers, or change points."""

    _tags = {
        "scitype_name": "detector",
        "short_descr": "detector - anomalies, outliers, change points",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.detection.base import BaseDetector

        return BaseDetector


class splitter(_BaseScitypeOfObject):
    """Time series splitter."""

    _tags = {
        "scitype_name": "splitter",
        "short_descr": "time series splitter",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.split.base import BaseSplitter

        return BaseSplitter


class transformer(_BaseScitypeOfObject):
    """Time series transformer."""

    _tags = {
        "scitype_name": "transformer",
        "short_descr": "time series transformer",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.transformations.base import BaseTransformer

        return BaseTransformer


class transformer_pairwise(_BaseScitypeOfObject):
    """Pairwise transformer for tabular data, distance or kernel."""

    _tags = {
        "scitype_name": "transformer-pairwise",
        "short_descr": "pairwise transformer for tabular data, distance or kernel",
        "parent_scitype": "transformer",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.dists_kernels.base import BasePairwiseTransformer

        return BasePairwiseTransformer


class transformer_pairwise_panel(_BaseScitypeOfObject):
    """Pairwise transformer for panel data, distance or kernel."""

    _tags = {
        "scitype_name": "transformer-pairwise-panel",
        "short_descr": "pairwise transformer for panel data, distance or kernel",
        "parent_scitype": "transformer",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.dists_kernels.base import BasePairwiseTransformerPanel

        return BasePairwiseTransformerPanel


class distribution(_BaseScitypeOfObject):
    """Pandas-like probability distribution."""

    _tags = {
        "scitype_name": "distribution",
        "short_descr": "pandas-like probability distribution",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.proba._base import BaseDistribution

        return BaseDistribution


@lru_cache
def _get_base_classes(mixin=False):
    """Get all object scitype classes in this module.

    Returns
    -------
    clss : tuple
        tuple of all object scitype classes in this module
    """
    clss = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    base_cls = _BaseScitypeOfObject
    base_cls_name = base_cls.__name__

    def is_base_class(cl):
        return cl.__name__ != base_cls_name and issubclass(cl, base_cls)

    clss = [cl for _, cl in clss if is_base_class(cl)]

    clss = [cl for cl in clss if cl.get_class_tags().get("mixin", False) == mixin]
    clss = tuple(clss)
    return clss


@lru_cache
def _construct_base_class_register(mixin=False):
    """Generate the register from the classes in this module."""
    clss = _get_base_classes(mixin=mixin)

    register = []
    for cl in clss:
        cl_tags = cl.get_class_tags()

        scitype_name = cl_tags["scitype_name"]
        short_descr = cl_tags["short_descr"]
        base_cls_ref = cl.get_base_class()

        register.append((scitype_name, base_cls_ref, short_descr))

    return register


def get_base_class_for_str(scitype_str):
    """Return base class for a given scitype string.

    Parameters
    ----------
    scitype_str : str, or list of str
        scitype shorthand, as in scitype_name field of scitype classes

    Returns
    -------
    base_cls : class or list of class
        base class corresponding to the scitype string,
        or list of base classes if input was a list
    """
    if isinstance(scitype_str, list):
        return [get_base_class_for_str(s) for s in scitype_str]

    base_classes = _get_base_classes()
    base_classes += _get_base_classes(mixin=True)
    base_class_lookup = {cl.get_class_tags()["scitype_name"]: cl for cl in base_classes}
    base_cls = base_class_lookup[scitype_str].get_base_class()
    return base_cls


def get_base_class_register(mixin=False, include_baseobjs=True):
    """Return register of object scitypes and base classes in sktime.

    Parameters
    ----------
    mixin : bool, optional (default=False)
        whether to return only full base classes (False) or only mixin classes (True)
    include_baseobjs : bool, optional (default=True)
          whether to include the BaseObject and BaseEstimator classes in the lookup

    Returns
    -------
    register : list of tuples
        each tuple corresponds to a base class, elements as follows:

        * 0 : string - scitype shorthand
        * 1 : type - the base class itself
        * 2 : string - plain English description of the scitype
    """
    raw_list = _construct_base_class_register(mixin=mixin)

    if not include_baseobjs:
        raw_list = [x for x in raw_list if x[0] not in ["object", "estimator"]]

    # for downwards scompatibility, move the "distributions" to the end of the list
    distr = [x for x in raw_list if x[0] == "distribution"]
    rest = [x for x in raw_list if x[0] != "distribution"]
    reordered_list = rest + distr

    return reordered_list.copy()


@lru_cache
def _construct_scitype_list(mixin=False):
    """Generate list of scitype strings from the register."""
    clss = _get_base_classes(mixin=mixin)

    scitype_list = []
    for cl in clss:
        tags = cl.get_class_tags()
        scitype_list.append((tags["scitype_name"], tags["short_descr"]))
    return scitype_list


def get_obj_scitype_list(mixin=False, include_baseobjs=True, return_descriptions=False):
    """Return list of object scitype shorthands in sktime.

    Parameters
    ----------
    mixin : bool, optional (default=False)
        whether to return only full base classes (False) or only mixin classes (True)
    include_baseobjs : bool, optional (default=True)
          whether to include the BaseObject and BaseEstimator classes in the lookup
    return_descriptions : bool, optional (default=False)
        whether to return descriptions along with scitype shorthands

    Returns
    -------
    scitype_list : list of string or list of tuple
        elements are scitype shorthands.
        If ``return_descriptions`` is False, elements are strings.
        If ``return_descriptions`` is True, elements are pairs of strings,
        where the first element is the scitype shorthand and the second is the
        description.
    """
    raw_list = _construct_scitype_list(mixin=mixin)

    if not include_baseobjs:
        raw_list = [x for x in raw_list if x[0] not in ["object", "estimator"]]

    # for downwards scompatibility, move the "distributions" to the end of the list
    distr = [x for x in raw_list if x[0] == "distribution"]
    rest = [x for x in raw_list if x[0] != "distribution"]
    reordered_list = rest + distr

    if return_descriptions:
        return reordered_list.copy()
    else:
        return [x[0] for x in reordered_list].copy()


def get_base_class_list(mixin=False, include_baseobjs=True):
    """Return list of base classes in sktime.

    Parameters
    ----------
    mixin : bool, optional (default=False)
        whether to return only full base classes (False) or only mixin classes (True)
    include_baseobjs : bool, optional (default=True)
          whether to include the BaseObject and BaseEstimator classes in the lookup

    Returns
    -------
    base_class_list : list of classes
        elements are base classes
    """
    register = get_base_class_register(mixin=mixin, include_baseobjs=include_baseobjs)
    return [x[1] for x in register]


def get_base_class_lookup(mixin=False, include_baseobjs=True):
    """Return lookup dictionary of scitype shorthands to base classes in sktime.

    Parameters
    ----------
    mixin : bool, optional (default=False)
        whether to return only full base classes (False) or only mixin classes (True)
    include_baseobjs : bool, optional (default=True)
          whether to include the BaseObject and BaseEstimator classes in the lookup

    Returns
    -------
    base_class_lookup : dict
        keys/entries are scitype shorthands/base classes
    """
    register = get_base_class_register(mixin=mixin, include_baseobjs=include_baseobjs)
    base_class_lookup = {x[0]: x[1] for x in register}
    return base_class_lookup


# LEGACY types - remove in 1.0.0
# ------------------------------


class series_annotator(_BaseScitypeOfObject):
    """Time series annotator."""

    _tags = {
        "scitype_name": "series-annotator",
        "short_descr": "detector - anomalies, outliers, change points",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        from sktime.detection.base import BaseDetector

        return BaseDetector


class transformer_series_to_primitives(_BaseScitypeOfObject):
    """LEGACY - time series to primitives transformer."""

    _tags = {
        "scitype_name": "series-to-primitives-trafo",
        "short_descr": "time series to primitives transformer",
        "parent_scitype": "transformer",
        "mixin": True,
    }

    @classmethod
    def get_base_class(cls):
        from sktime.transformations.base import _SeriesToPrimitivesTransformer

        return _SeriesToPrimitivesTransformer


class transformer_series_to_series(_BaseScitypeOfObject):
    """LEGACY - time series to time series transformer."""

    _tags = {
        "scitype_name": "series-to-series-trafo",
        "short_descr": "time series to time series transformer",
        "parent_scitype": "transformer",
        "mixin": True,
    }

    @classmethod
    def get_base_class(cls):
        from sktime.transformations.base import _SeriesToSeriesTransformer

        return _SeriesToSeriesTransformer


class transformer_panel_to_tabular(_BaseScitypeOfObject):
    """LEGACY - panel to tabular transformer."""

    _tags = {
        "scitype_name": "panel-to-tabular-trafo",
        "short_descr": "panel to tabular transformer",
        "parent_scitype": "transformer",
        "mixin": True,
    }

    @classmethod
    def get_base_class(cls):
        from sktime.transformations.base import _PanelToTabularTransformer

        return _PanelToTabularTransformer


class transformer_panel_to_panel(_BaseScitypeOfObject):
    """LEGACY - panel to panel transformer."""

    _tags = {
        "scitype_name": "panel-to-panel-trafo",
        "short_descr": "panel to panel transformer",
        "parent_scitype": "transformer",
        "mixin": True,
    }

    @classmethod
    def get_base_class(cls):
        from sktime.transformations.base import _PanelToPanelTransformer

        return _PanelToPanelTransformer
