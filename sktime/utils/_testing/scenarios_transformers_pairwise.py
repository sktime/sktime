# -*- coding: utf-8 -*-
"""Test scenarios for pairwise transformers.

Contains TestScenario concrete children to run in tests for pairwise transformers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_transformers_pairwise", "scenarios_transformers_pairwise_panel"]

from inspect import isclass

import pandas as pd

from sktime.base import BaseObject
from sktime.utils._testing.scenarios import TestScenario

# random seed for generating data to keep scenarios exactly reproducible
RAND_SEED = 42


def get_tag(obj, tag_name):
    """Shorthand for get_tag vs get_class_tag, obj can be class or object."""
    if isclass(obj):
        return obj.get_class_tag(tag_name)
    else:
        return obj.get_tag(tag_name)


# no logic in scenario classes, but placeholder and for pattern homogeneity
class TransformerPairwiseTestScenario(TestScenario, BaseObject):
    """Generic test scenario for pairwise transformers."""

    pass


class TransformerPairwisePanelTestScenario(TestScenario, BaseObject):
    """Generic test scenario for pairwise panel transformers."""

    pass


d = {"col1": [1, 2], "col2": [3, 4]}
d = pd.DataFrame(d)

d2 = {"col1": [2, 3, 4], "col2": [3, 4, 5]}
d2 = pd.DataFrame(d2)


class TransformerPairwiseTransformSymm(TransformerPairwiseTestScenario):
    """Empty fit, one argument in transform."""

    _tags = {"symmetric": True, "pre-refactor": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": d},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwiseTransformAsymm(TransformerPairwiseTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "pre-refactor": False}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": d, "X2": d2},
    }
    default_method_sequence = ["fit", "transform"]


scenarios_transformers_pairwise = [
    TransformerPairwiseTransformSymm,
    TransformerPairwiseTransformAsymm,
]

X = [d, d]
X2 = [d2, d, d2]


class TransformerPairwisePanelTransformSymm(TransformerPairwisePanelTestScenario):
    """Empty fit, one argument in transform."""

    _tags = {"symmetric": True, "pre-refactor": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwisePanelTransformAsymm(TransformerPairwisePanelTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "pre-refactor": False}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X, "X2": X2},
    }
    default_method_sequence = ["fit", "transform"]


scenarios_transformers_pairwise_panel = [
    TransformerPairwisePanelTransformSymm,
    TransformerPairwisePanelTransformAsymm,
]
