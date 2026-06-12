"""Example generation for testing.

Exports examples of in-memory data containers, useful for testing as fixtures.

Examples come in clusters, tagged by scitype: str, index: int, and metadata: dict.

All examples with the same index are considered "content-wise the same", i.e.,
representing the same abstract data object. They differ by mtype, i.e.,
machine type, which is the specific in-memory representation.

If an example returns None, it indicates that representation
with that specific mtype is not possible.

If the tag "lossy" is True, the representation is considered incomplete,
e.g., metadata such as column names are missing.

Types of tests that can be performed with these examples:

* the mtype and scitype of the example should be correctly inferred by checkers.
* the metadata of the example should be correctly inferred by checkers.
* conversions from non-lossy representations to any other ones
  should yield the element exactly, identically, for examples of the same index.
"""

import numpy as np
import pandas as pd

from sktime.datatypes._base import BaseExample

###
# example 0: univariate


class _ProbaUniv(BaseExample):
    _tags = {
        "scitype": "Proba",
        "index": 0,
        "metadata": {
            "is_univariate": True,
            "is_empty": False,
            "has_nans": False,
        },
    }


class _ProbaUnivPredQ(_ProbaUniv):
    _tags = {
        "mtype": "pred_quantiles",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        pred_q = pd.DataFrame({0.2: [1, 2, 3], 0.6: [2, 3, 4]})
        pred_q.columns = pd.MultiIndex.from_product([["foo"], [0.2, 0.6]])

        return pred_q


class _ProbaUnivPredInt(_ProbaUniv):
    _tags = {
        "mtype": "pred_interval",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        # we need to use this due to numerical inaccuracies
        # from the binary based representation
        pseudo_0_2 = 2 * np.abs(0.6 - 0.5)

        pred_int = pd.DataFrame({0.2: [1, 2, 3], 0.6: [2, 3, 4]})
        pred_int.columns = pd.MultiIndex.from_tuples(
            [("foo", 0.6, "lower"), ("foo", pseudo_0_2, "upper")]
        )

        return pred_int


###
# example 1: multi


class _ProbaMulti(BaseExample):
    _tags = {
        "scitype": "Proba",
        "index": 1,
        "metadata": {
            "is_univariate": False,
            "is_empty": False,
            "has_nans": False,
        },
    }


class _ProbaMultiPredQ(_ProbaMulti):
    _tags = {
        "mtype": "pred_quantiles",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        pred_q = pd.DataFrame(
            {0.2: [1, 2, 3], 0.6: [2, 3, 4], 42: [5, 3, -1], 46: [5, 3, -1]}
        )
        pred_q.columns = pd.MultiIndex.from_product([["foo", "bar"], [0.2, 0.6]])

        return pred_q


class _ProbaMultiPredInt(_ProbaMulti):
    _tags = {
        "mtype": "pred_interval",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        # we need to use this due to numerical inaccuracies
        # from the binary based representation
        pseudo_0_2 = 2 * np.abs(0.6 - 0.5)

        pred_int = pd.DataFrame(
            {0.2: [1, 2, 3], 0.6: [2, 3, 4], 42: [5, 3, -1], 46: [5, 3, -1]}
        )
        pred_int.columns = pd.MultiIndex.from_tuples(
            [
                ("foo", 0.6, "lower"),
                ("foo", pseudo_0_2, "upper"),
                ("bar", 0.6, "lower"),
                ("bar", pseudo_0_2, "upper"),
            ]
        )

        return pred_int
