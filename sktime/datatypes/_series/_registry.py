# -*- coding: utf-8 -*-

import pandas as pd

__all__ = [
    "MTYPE_REGISTER_SERIES",
    "MTYPE_LIST_SERIES",
]


MTYPE_REGISTER_SERIES = [
    ("pd.Series", "Series", "pd.Series representation of a univariate series"),
    (
        "pd.DataFrame",
        "Series",
        "pd.DataFrame representation of a uni- or multivariate series",
    ),
    (
        "np.ndarray",
        "Series",
        "2D numpy.ndarray with rows=samples, cols=variables, index=integers",
    ),
]

MTYPE_LIST_SERIES = pd.DataFrame(MTYPE_REGISTER_SERIES)[0].values
