"""Registry of mtypes for Proba scitype.

See datatypes._registry for API.
"""

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

MTYPE_LIST_PROBA = [x[0] for x in MTYPE_REGISTER_PROBA]
