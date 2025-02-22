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
* the metadata of hte example should be correctly inferred by checkers.
* conversions from non-lossy representations to any other ones
  should yield the element exactly, identically, for examples of the same index.
"""

import pandas as pd

from sktime.datatypes._base import BaseExample

###


class _AlignmentSimple(BaseExample):
    _tags = {
        "scitype": "Alignment",
        "index": 0,
        "metadata": {},
    }


class _AlignmentSimpleAlignment(_AlignmentSimple):
    _tags = {
        "mtype": "alignment",
        "python_dependencies": None,
        "lossy": False,
        "metadata": {"is_multiple": False},
    }

    def build(self):
        return pd.DataFrame({"ind0": [1, 2, 2, 3], "ind1": [0, 0, 1, 1]})


class _AlignmentSimpleAlignmentLoc(_AlignmentSimple):
    _tags = {
        "mtype": "alignment_loc",
        "python_dependencies": None,
        "lossy": False,
        "metadata": {"is_multiple": False},
    }

    def build(self):
        return pd.DataFrame({"ind0": [2, 2.5, 2.5, 100], "ind1": [-1, -1, 2, 2]})
