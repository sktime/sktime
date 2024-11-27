"""Plaid dataset."""

from sktime.datasets._single_problem_loaders import load_plaid
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class PLAID(_ClassificationDatasetFromLoader):
    """PLAID time series classification problem.

    Example of a univariate problem with unequal length series.

    Parameters
    ----------
    return_mtype: valid Panel mtype str or None, optional (default="pd-multiindex")
        Memory data format specification to return X in, None = "nested_univ" type.
        str can be any supported sktime Panel mtype,
            for list of mtypes, see datatypes.MTYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        commonly used specifications:
            "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
            "numpy3D"/"numpy3d"/"np3D": 3D np.ndarray (instance, variable, time index)
            "numpy2d"/"np2d"/"numpyflat": 2D np.ndarray (instance, time index)
            "nested_univ: nested pd.DataFrame, pd.Series in cells
        Exception is raised if the data cannot be stored in the requested type.

    Examples
    --------
    >>> from sktime.datasets.classification.plaid import PLAID
    >>> X, y = PLAID().load()
    """

    _tags = {
        "is_univariate": True,
        "is_one_series": False,
        "n_panels": 1,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_equal_length": False,
        "is_equal_index": False,
        "is_empty": False,
        "has_nans": False,
        "n_instances": 1074,
        "n_instances_train": 537,
        "n_instances_test": 537,
        "n_classes": 11,
    }

    loader_func = load_plaid
