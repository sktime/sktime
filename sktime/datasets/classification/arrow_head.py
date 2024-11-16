"""ArrowHead dataset."""

from sktime.datasets._single_problem_loaders import load_arrow_head
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class ArrowHead(_ClassificationDatasetFromLoader):
    """ArrowHead time series classification problem.

    Example of a univariate problem with equal length series.

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
            "nested_univ": nested pd.DataFrame, pd.Series in cells
        Exception is raised if the data cannot be stored in the requested type.

    Examples
    --------
    >>> from sktime.datasets.classification.arrow_head import ArrowHead
    >>> X, y = ArrowHead().load()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      251
    Train cases:        36
    Test cases:         175
    Number of classes:  3

    The ArrowHead dataset consists of outlines of images of arrowheads. The shapes of
    the projectile points are converted into a time series using the angle-based method.
    Classification of projectile points is an important topic in anthropology.
    The classes are based on shape distinctions such as the presence and location of a
    notch in the arrow. The problem in the repository is a length-normalized version of
    that used in Ye09shapelets. The three classes are called "Avonlea", "Clovis", and
    "Mix".

    Dataset details: http://timeseriesclassification.com/description.php?Dataset=ArrowHead
    """

    _tags = {
        "is_univariate": True,
        "n_panels": 1,
        "is_one_panel": True,
        "is_equally_spaced": True,
        "is_equal_length": True,
        "is_equal_index": False,
        "is_empty": False,
        "has_nans": False,
        "n_instances": 211,
        "n_instances_train": 36,
        "n_instances_test": 175,
        "n_classes": 3,
    }

    loader_func = load_arrow_head
