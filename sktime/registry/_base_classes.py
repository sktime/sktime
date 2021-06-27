# -*- coding: utf-8 -*-
"""
Register of estimator base classes corresponding to sktime scitypes.

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

import pandas as pd

from sktime.classification.base import BaseClassifier
from sktime.clustering.base import BaseCluster
from sktime.forecasting.base import BaseForecaster
from sktime.regression.base import BaseRegressor
from sktime.transformations.base import BaseTransformer

from sktime.transformations.base import _PanelToPanelTransformer
from sktime.transformations.base import _PanelToTabularTransformer
from sktime.transformations.base import _SeriesToPrimitivesTransformer
from sktime.transformations.base import _SeriesToSeriesTransformer


BASE_CLASS_REGISTER = [
    ("classifier", BaseClassifier, "time series classifier"),
    ("clusterer", BaseCluster, "time series clusterer"),
    ("regressor", BaseRegressor, "time series regressor"),
    ("forecaster", BaseForecaster, "forecaster"),
    ("transformer", BaseTransformer, "time series transformer"),
]

BASE_CLASS_SCITYPE_LIST = pd.DataFrame(BASE_CLASS_REGISTER)[0].values

BASE_CLASS_LIST = pd.DataFrame(BASE_CLASS_REGISTER)[1].values

BASE_CLASS_LOOKUP = dict(zip(BASE_CLASS_SCITYPE_LIST, BASE_CLASS_LIST))


TRANSFORMER_MIXIN_REGISTER = [
    (
        "s-to-prim-trafo",
        _SeriesToPrimitivesTransformer,
        "time-series-to-primitives transformer",
    ),
    (
        "s-to-s-trafo",
        _SeriesToSeriesTransformer,
        "time-series-to-time-series transformer",
    ),
    (
        "p-to-t-trafo", _PanelToTabularTransformer, "panel-to-tabular transformer"),
    (
        "p-to-p-trafo", _PanelToPanelTransformer, "panel-to-panel transformer"),
]

TRANSFORMER_MIXIN_SCITYPE_LIST = pd.DataFrame(TRANSFORMER_MIXIN_REGISTER)[0].values

TRANSFORMER_MIXIN_LIST = pd.DataFrame(TRANSFORMER_MIXIN_REGISTER)[1].values

TRANSFORMER_MIXIN_LOOKUP = dict(
    zip(TRANSFORMER_MIXIN_SCITYPE_LIST, TRANSFORMER_MIXIN_LIST)
)
