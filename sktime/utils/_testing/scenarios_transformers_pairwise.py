"""Test scenarios for pairwise transformers.

Contains TestScenario concrete children to run in tests for pairwise transformers.
"""

__author__ = ["fkiraly"]

__all__ = ["scenarios_transformers_pairwise", "scenarios_transformers_pairwise_panel"]

from inspect import isclass

import pandas as pd

from sktime.base import BaseObject
from sktime.datatypes import convert_to
from sktime.utils._testing.panel import make_transformer_problem
from sktime.utils._testing.scenarios import TestScenario
from sktime.utils._testing.series import _make_series

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

X1_np = _make_series(
    n_columns=4,
    n_timepoints=4,
    random_state=1,
    return_mtype="np.ndarray",
)
X2_np = _make_series(
    n_columns=4,
    n_timepoints=5,
    random_state=2,
    return_mtype="np.ndarray",
)


class TransformerPairwiseTransformSymm(TransformerPairwiseTestScenario):
    """Empty fit, one argument in transform."""

    _tags = {"symmetric": True, "is_enabled": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": d},
        "transform_diag": {"X": d},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwiseTransformAsymm(TransformerPairwiseTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "is_enabled": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": d, "X2": d2},
        "transform_diag": {"X": d},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwiseTransformNumpy(TransformerPairwiseTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "is_enabled": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X1_np, "X2": X2_np},
        "transform_diag": {"X": X1_np},
    }
    default_method_sequence = ["fit", "transform"]


scenarios_transformers_pairwise = [
    TransformerPairwiseTransformSymm,
    TransformerPairwiseTransformAsymm,
    TransformerPairwiseTransformNumpy,
]

X = [d, d]
X2 = [d, d, d]

X1_list_df = make_transformer_problem(
    n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=False
)
X2_list_df = make_transformer_problem(
    n_instances=5, n_columns=4, n_timepoints=5, random_state=2, return_numpy=False
)

X1_num_pan = convert_to(X1_list_df, to_type="numpy3D")
X2_num_pan = convert_to(X2_list_df, to_type="numpy3D")


class TransformerPairwisePanelTransformSymm(TransformerPairwisePanelTestScenario):
    """Empty fit, one argument in transform."""

    _tags = {"symmetric": True, "is_enabled": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X},
        "transform_diag": {"X": X},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwisePanelTransformAsymm(TransformerPairwisePanelTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "is_enabled": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X, "X2": X2},
        "transform_diag": {"X": X},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwisePanelTransformListdf(TransformerPairwisePanelTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "is_enabled": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X1_list_df, "X2": X2_list_df},
        "transform_diag": {"X": X1_list_df},
    }
    default_method_sequence = ["fit", "transform"]


class TransformerPairwisePanelTransformNumpy(TransformerPairwisePanelTestScenario):
    """Empty fit, two arguments of different length in transform."""

    _tags = {"symmetric": False, "is_enabled": True}

    args = {
        "fit": {"X": None, "X2": None},
        "transform": {"X": X1_num_pan, "X2": X2_num_pan},
        "transform_diag": {"X": X1_num_pan},
    }
    default_method_sequence = ["fit", "transform"]


scenarios_transformers_pairwise_panel = [
    TransformerPairwisePanelTransformSymm,
    TransformerPairwisePanelTransformAsymm,
    TransformerPairwisePanelTransformListdf,
    TransformerPairwisePanelTransformNumpy,
]
