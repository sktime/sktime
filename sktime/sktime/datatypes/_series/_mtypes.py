"""Machine type inference for Series scitype.

Exports mtype dictionaries for Series scitype:

infer_mtype_dict: dict indexed by str
  key = considered as this scitype - str

Function signature of all elements
convert_dict[key]

Parameters
----------
obj : object to infer scitype of

Returns
-------
scitype : str - inferred scitype

Raises
------
TypeError, if scitype cannot be inferred
"""

__author__ = ["fkiraly"]

__all__ = ["infer_mtype_dict"]

import numpy as np
import pandas as pd

#########################################################
# methods to infer the machine type subject to a scitype
#########################################################

infer_mtype_dict = dict()


def infer_mtype_Series(obj):
    obj_type = type(obj)

    infer_dict = {
        pd.Series: "pd.Series",
        pd.DataFrame: "pd.DataFrame",
        np.ndarray: "np.ndarray",
    }

    if obj_type not in infer_dict.keys():
        raise TypeError("scitype cannot be inferred")

    return infer_dict[obj_type]


infer_mtype_dict["Series"] = infer_mtype_Series
