# -*- coding: utf-8 -*-

import pandas as pd

__all__ = [
    "MTYPE_REGISTER_PROBA",
    "MTYPE_LIST_PROBA",
]


MTYPE_REGISTER_PROBA = [
    ("pred_interval", "Proba", "predictive intervals"),
    ("pred_quantiles", "Proba", "quantile predictions"),
    ("pred_var", "Proba", "variance predictions"),
    # ("pred_dost", "Proba", "full distribution predictions, tensorflow-probability"),
]

MTYPE_LIST_PROBA = pd.DataFrame(MTYPE_REGISTER_PROBA)[0].values
