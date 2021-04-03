# -*- coding: utf-8 -*-
__all__ = [
    "load_airline",
    "load_arrow_head",
    "load_gunpoint",
    "load_basic_motions",
    "load_osuleaf",
    "load_italy_power_demand",
    "load_longley",
    "load_lynx",
    "load_shampoo_sales",
    "ArmaGenerator",
    "LinearGenerator",
    "NoiseGenerator",
    "load_uschange",
    "load_UCR_UEA_dataset",
]

from sktime.datasets.base import load_airline
from sktime.datasets.base import load_gunpoint
from sktime.datasets.base import load_arrow_head
from sktime.datasets.base import load_basic_motions
from sktime.datasets.base import load_osuleaf
from sktime.datasets.base import load_italy_power_demand
from sktime.datasets.base import load_longley
from sktime.datasets.base import load_lynx
from sktime.datasets.base import load_shampoo_sales
from sktime.datasets.base import load_uschange
from sktime.datasets.base import load_UCR_UEA_dataset
from sktime.datasets.generators import ArmaGenerator
from sktime.datasets.generators import LinearGenerator
from sktime.datasets.generators import NoiseGenerator
