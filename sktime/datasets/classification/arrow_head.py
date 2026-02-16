"""ArrowHead dataset."""

from sktime.datasets._single_problem_loaders import load_arrow_head
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class ArrowHead(_ClassificationDatasetFromLoader):
    """ArrowHead time series classification problem.

    Example of a univariate problem with equal length series.

    Examples
    --------
    >>> from sktime.datasets.classification import ArrowHead
    >>> X, y = ArrowHead().load("X", "y")

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
        "name": "arrow_head",
        "n_splits": 1,
        "is_univariate": True,
        "n_instances": 211,
        "n_instances_train": 36,
        "n_instances_test": 175,
        "n_classes": 3,
    }

    loader_func = load_arrow_head
