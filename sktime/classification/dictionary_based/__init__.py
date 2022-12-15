# -*- coding: utf-8 -*-
"""Dictionary based time series classifiers."""
__all__ = [
    "IndividualBOSS",
    "BOSSEnsemble",
    "ContractableBOSS",
    "TemporalDictionaryEnsemble",
    "IndividualTDE",
    "HYDRA",
    "MPDist",
    "WEASEL",
    "WEASEL_DILATION",
    "MUSE",
    "MUSE_DILATION",
]

from sktime.classification.dictionary_based._boss import BOSSEnsemble, IndividualBOSS
from sktime.classification.dictionary_based._cboss import ContractableBOSS
from sktime.classification.dictionary_based._hydra import HYDRA
from sktime.classification.dictionary_based._mpdist import MPDist
from sktime.classification.dictionary_based._muse import MUSE
from sktime.classification.dictionary_based._muse_dilation import MUSE_DILATION
from sktime.classification.dictionary_based._tde import (
    IndividualTDE,
    TemporalDictionaryEnsemble,
)
from sktime.classification.dictionary_based._weasel import WEASEL
from sktime.classification.dictionary_based._weasel_dilation import WEASEL_DILATION
