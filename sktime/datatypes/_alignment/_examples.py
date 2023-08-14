"""Example generation for testing.

Exports dict of examples, useful for testing as fixtures.

example_dict: dict indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are data objects, considered examples for the mtype
    all examples with same index are considered "same" on scitype content
    if None, indicates that representation is not possible
"""

import pandas as pd

example_dict = dict()

###

align = pd.DataFrame({"ind0": [1, 2, 2, 3], "ind1": [0, 0, 1, 1]})

example_dict[("alignment", "Alignment", 0)] = align


align = pd.DataFrame({"ind0": [2, 2.5, 2.5, 100], "ind1": [-1, -1, 2, 2]})

example_dict[("alignment_loc", "Alignment", 0)] = align
