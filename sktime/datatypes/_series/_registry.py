# -*- coding: utf-8 -*-

__all__ = ["MTYPE_REGISTER_SERIES", "MTYPE_LIST_SERIES", "SeriesMtype"]

import numpy as np
import pandas as pd

from sktime.base._registry_enum import BaseRegistryEnum


class SeriesMtype(BaseRegistryEnum):
    """Enum class for series mtypes."""

    pd_series = (
        "pd.Series representation of a univariate series",
        "Series",
        pd.Series,
    )
    pd_dataframe = (
        "pd.DataFrame representation of a uni- or multivariate series",
        "Series",
        pd.DataFrame,
    )
    np_array = (
        "2D numpy.ndarray with rows=samples, cols=variables, index=integers",
        "Series",
        np.ndarray,
    )


MTYPE_REGISTER_SERIES = [tuple(mtype) for mtype in SeriesMtype]

MTYPE_LIST_SERIES = pd.DataFrame(MTYPE_REGISTER_SERIES)[0].values
