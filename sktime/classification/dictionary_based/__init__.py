# -*- coding: utf-8 -*-
"""Dictionary based time series classifiers."""
__all__ = [
    "IndividualBOSS",
    "BOSSEnsemble",
    "ContractableBOSS",
    "TemporalDictionaryEnsemble",
    "IndividualTDE",
    "Hydra",
    "WEASEL",
    "WEASEL_STEROIDS",
    "MUSE",
    "MUSE_NEW",
]

from sktime.classification.dictionary_based._boss import BOSSEnsemble, IndividualBOSS
from sktime.classification.dictionary_based._cboss import ContractableBOSS
from sktime.classification.dictionary_based._hydra import Hydra
from sktime.classification.dictionary_based._muse import MUSE
from sktime.classification.dictionary_based._muse_new import MUSE_NEW
from sktime.classification.dictionary_based._tde import (
    IndividualTDE,
    TemporalDictionaryEnsemble,
)
from sktime.classification.dictionary_based._weasel import WEASEL
from sktime.classification.dictionary_based._weasel_steroids import WEASEL_STEROIDS
