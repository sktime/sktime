"""Time series classification datasets."""

from .arrow_head import ArrowHead
from .basic_motions import BasicMotions
from .gunpoint import GunPoint
from .italy_power_demand import ItalyPowerDemand
from .japanese_vowels import JapaneseVowels
from .osuleaf import OSULeaf
from .plaid import PLAID

__all__ = [
    "ArrowHead",
    "BasicMotions",
    "GunPoint",
    "ItalyPowerDemand",
    "JapaneseVowels",
    "OSULeaf",
    "PLAID",
]
