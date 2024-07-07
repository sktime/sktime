"""Utilities to handle categorical features in data."""


from sktime.datatypes._dtypekind import DtypeKind

__author__ = ["Abhay-Lejith"]


def _need_to_encode(est, metadata, var_name):
    """Return whether encoding is required or not.

    Function to check whether categorical support is available and raise error or
    return the need to encode accordingly.
    Current state of categorical support:

    |  Module            |  X  |  y  |
    |--------------------|-----|-----|
    |  Forecasting       | Yes | No  |
    |  Classification    | No  | Yes |

    All other modules do not support categorical in X or y currently.
    """
    # if data doesn't contain categorical features, then no need to encode
    if DtypeKind.CATEGORICAL not in metadata["feature_kind"]:
        return False

    est_scitype = est.get_tag("object_type")
    if est_scitype == "classifier" and var_name == "y":
        return False
    elif est_scitype == "forecaster" and var_name == "X":
        if est.get_tag("capability:categorical_in_X", False, False):
            return False
        return True

    # in all other cases, raise not supported error
    raise TypeError(
        f"{est_scitype} {est} does not support categorical features in {var_name}."
        " Currently, sktime supports categorical values in exogeneous(X) data in "
        "forecasters and endogeneous(y) in classifiers."
    )


def _handle_categorical(est, df, metadata, var_name):
    if var_name == "X" and est.get_tag("ignores-exogeneous-X", False, False):
        return df

    if _need_to_encode(est, metadata, var_name):
        # initial step of raising error in yes/no case.
        # will replace this with encoding logic in next step.
        raise TypeError(
            f"""Forecaster {est} does not support categorical features in {var_name}."""
        )

    return df
