# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common utilities for polars based data containers."""
from sktime.datatypes._common import _req
from sktime.datatypes._common import _ret as ret


def check_polars_frame(obj, return_metadata=False, var_name="obj", lazy=False):
    """Check polars frame, generic format."""
    import polars as pl

    metadata = {}

    if lazy:
        exp_type = pl.LazyFrame
        exp_type_str = "LazyFrame"
    else:
        exp_type = pl.DataFrame
        exp_type_str = "DataFrame"

    if not isinstance(obj, exp_type):
        msg = f"{var_name} must be a polars {exp_type_str}, found {type(obj)}"
        return ret(False, msg, None, return_metadata)

    # we now know obj is a polars DataFrame or LazyFrame
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = obj.width < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.width == 1
    if _req("n_instances", return_metadata):
        if hasattr(obj, "height"):
            metadata["n_instances"] = obj.height
        else:
            metadata["n_instances"] = "NA"
    if _req("n_features", return_metadata):
        metadata["n_features"] = obj.width
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns

    # check if there are any nans
    #   compute only if needed
    if _req("has_nans", return_metadata):
        if isinstance(obj, pl.LazyFrame):
            metadata["has_nans"] = "NA"
        else:
            hasnan = obj.null_count().sum_horizontal().to_numpy()[0] > 0
            metadata["has_nans"] = hasnan

    return ret(True, None, metadata, return_metadata)
