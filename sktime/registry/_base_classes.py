# -*- coding: utf-8 -*-
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

BASE_CLASS_LIST - list of string
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

__author__ = ["fkiraly"]

import pandas as pd

from sktime.alignment.base import BaseAligner
from sktime.annotation.base import BaseSeriesAnnotator
from sktime.base import BaseEstimator, BaseObject
from sktime.classification.base import BaseClassifier
from sktime.classification.early_classification import BaseEarlyClassifier
from sktime.clustering.base import BaseClusterer
from sktime.dists_kernels._base import (
    BasePairwiseTransformer,
    BasePairwiseTransformerPanel,
)
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_selection._split import BaseSplitter
from sktime.networks.base import BaseDeepNetwork
from sktime.param_est.base import BaseParamFitter
from sktime.performance_metrics.base import BaseMetric
from sktime.regression.base import BaseRegressor
from sktime.transformations.base import (
    BaseTransformer,
    _PanelToPanelTransformer,
    _PanelToTabularTransformer,
    _SeriesToPrimitivesTransformer,
    _SeriesToSeriesTransformer,
)

BASE_CLASS_REGISTER = [
    ("object", BaseObject, "object"),
    ("estimator", BaseEstimator, "estimator = object with fit"),
    ("aligner", BaseAligner, "time series aligner or sequence aligner"),
    ("classifier", BaseClassifier, "time series classifier"),
    ("clusterer", BaseClusterer, "time series clusterer"),
    ("early_classifier", BaseEarlyClassifier, "early time series classifier"),
    ("forecaster", BaseForecaster, "forecaster"),
    ("metric", BaseMetric, "performance metric"),
    ("network", BaseDeepNetwork, "deep learning network"),
    ("param_est", BaseParamFitter, "parameter fitting estimator"),
    ("regressor", BaseRegressor, "time series regressor"),
    ("series-annotator", BaseSeriesAnnotator, "time series annotator"),
    ("splitter", BaseSplitter, "time series splitter"),
    ("transformer", BaseTransformer, "time series transformer"),
    (
        "transformer-pairwise",
        BasePairwiseTransformer,
        "pairwise transformer for tabular data, distance or kernel",
    ),
    (
        "transformer-pairwise-panel",
        BasePairwiseTransformerPanel,
        "pairwise transformer for panel data, distance or kernel",
    ),
]


BASE_CLASS_SCITYPE_LIST = pd.DataFrame(BASE_CLASS_REGISTER)[0].tolist()

BASE_CLASS_LIST = pd.DataFrame(BASE_CLASS_REGISTER)[1].tolist()

BASE_CLASS_LOOKUP = dict(zip(BASE_CLASS_SCITYPE_LIST, BASE_CLASS_LIST))


TRANSFORMER_MIXIN_REGISTER = [
    (
        "series-to-primitive-trafo",
        _SeriesToPrimitivesTransformer,
        "time-series-to-primitives transformer",
    ),
    (
        "series-to-series-trafo",
        _SeriesToSeriesTransformer,
        "time-series-to-time-series transformer",
    ),
    (
        "panel-to-tabular-trafo",
        _PanelToTabularTransformer,
        "panel-to-tabular transformer",
    ),
    ("panel-to-panel-trafo", _PanelToPanelTransformer, "panel-to-panel transformer"),
]

TRANSFORMER_MIXIN_SCITYPE_LIST = pd.DataFrame(TRANSFORMER_MIXIN_REGISTER)[0].tolist()

TRANSFORMER_MIXIN_LIST = pd.DataFrame(TRANSFORMER_MIXIN_REGISTER)[1].tolist()

TRANSFORMER_MIXIN_LOOKUP = dict(
    zip(TRANSFORMER_MIXIN_SCITYPE_LIST, TRANSFORMER_MIXIN_LIST)
)
