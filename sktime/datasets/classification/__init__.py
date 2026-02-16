"""Time series classification datasets."""

from sktime.datasets.classification.acsf1 import ACSF1
from sktime.datasets.classification.arrow_head import ArrowHead
from sktime.datasets.classification.basic_motions import BasicMotions
from sktime.datasets.classification.gunpoint import GunPoint
from sktime.datasets.classification.italy_power_demand import ItalyPowerDemand
from sktime.datasets.classification.japanese_vowels import JapaneseVowels
from sktime.datasets.classification.osuleaf import OSULeaf
from sktime.datasets.classification.plaid import PLAID
from sktime.datasets.classification.ucr_uea_archive import UCRUEADataset

__all__ = [
    "ACSF1",
    "ArrowHead",
    "BasicMotions",
    "GunPoint",
    "ItalyPowerDemand",
    "JapaneseVowels",
    "OSULeaf",
    "PLAID",
    "UCRUEADataset",
]
