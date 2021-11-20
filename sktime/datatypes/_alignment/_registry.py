# -*- coding: utf-8 -*-

import pandas as pd

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

MTYPE_LIST_ALIGNMENT = pd.DataFrame(MTYPE_REGISTER_ALIGNMENT)[0].values
