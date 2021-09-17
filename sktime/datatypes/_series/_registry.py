# -*- coding: utf-8 -*-

__all__ = ["MTYPE_REGISTER_SERIES", "MTYPE_LIST_SERIES", "SeriesMtype"]

import pandas as pd

from sktime.base._registry_enum import BaseRegistryEnum


class SeriesMtype(BaseRegistryEnum):
    """Enum class for series mtypes."""

    PD_SERIES = (
        "pd.Series",
        "pd.Series representation of a univariate series",
        "Series",
    )
    PD_DATAFRAME = (
        "pd.DataFrame",
        "pd.DataFrame representation of a uni- or multivariate series",
        "Series",
    )
    NP_NDARRAY = (
        "np.ndarray",
        "2D numpy.ndarray with rows=samples, cols=variables, index=integers",
        "Series",
    )


MTYPE_REGISTER_SERIES = [tuple(mtype) for mtype in SeriesMtype]

MTYPE_LIST_SERIES = pd.DataFrame(MTYPE_REGISTER_SERIES)[0].values
