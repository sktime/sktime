# -*- coding: utf-8 -*-

__all__ = [
    "MTYPE_REGISTER_SERIES",
    "MTYPE_LIST_SERIES",
    "SeriesMtype"
]
import pandas as pd
from sktime.registry._registry_enum import BaseRegistryEnum


class SeriesMtype(BaseRegistryEnum):
    NP_NDARRAY = (
        'np.ndarray',
        'Series',
        '2D numpy.ndarray with rows=samples, cols=variables, index=integers'
    )
    PD_DATAFRAME = (
        'pd.DataFrame',
        'Series',
        'pd.DataFrame representation of a uni- or multivariate series'
    )
    PD_SERIES = (
        'pd.Series',
        'Series'
        'pd.Series representation of a univariate series'
    )


MTYPE_REGISTER_SERIES = [tuple(mtype) for mtype in SeriesMtype]

MTYPE_LIST_SERIES = pd.DataFrame(MTYPE_REGISTER_SERIES)[0].values
