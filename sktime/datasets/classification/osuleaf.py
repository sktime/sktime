"""OSULeaf dataset."""

from sktime.datasets._single_problem_loaders import load_osuleaf
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class OSULeaf(_ClassificationDatasetFromLoader):
    """OSULeaf time series classification problem.

    Example of a univariate problem with equal-length series.

    Examples
    --------
    >>> from sktime.datasets.classification import OSULeaf
    >>> X, y = OSULeaf().load("X", "y")

    Notes
    -----
    Dimensionality:     univariate
    Series length:      427
    Train cases:        200
    Test cases:         242
    Number of classes:  6

    The OSULeaf dataset consists of one-dimensional outlines of leaves.
    The series were obtained by color image segmentation and boundary
    extraction (in the anti-clockwise direction) from digitized leaf images
    of six classes: Acer Circinatum, Acer Glabrum, Acer Macrophyllum,
    Acer Negundo, Quercus Garryana, and Quercus Kelloggii for the MSc thesis
    "Content-Based Image Retrieval: Plant Species Identification" by A. Grandhi.

    Dataset details: http://timeseriesclassification.com/description.php?Dataset=OSULeaf
    """

    _tags = {
        "name": "osuleaf",
        "n_splits": 1,
        "is_univariate": True,
        "n_instances": 442,
        "n_instances_train": 200,
        "n_instances_test": 242,
        "n_classes": 6,
    }

    loader_func = load_osuleaf
