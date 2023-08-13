"""Registry of mtypes for Alignment scitype.

See datatypes._registry for API.
"""

__all__ = [
    "MTYPE_REGISTER_ALIGNMENT",
    "MTYPE_LIST_ALIGNMENT",
]


MTYPE_REGISTER_ALIGNMENT = [
    (
        "alignment",
        "Alignment",
        "pd.DataFrame in alignment format, values are iloc index references",
    ),
    (
        "alignment_loc",
        "Alignment",
        "pd.DataFrame in alignment format, values are loc index references",
    ),
]

MTYPE_LIST_ALIGNMENT = [x[0] for x in MTYPE_REGISTER_ALIGNMENT]
