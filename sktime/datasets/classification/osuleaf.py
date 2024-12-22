"""OSULeaf dataset."""

from sktime.datasets._single_problem_loaders import load_osuleaf
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class OSULeaf(_ClassificationDatasetFromLoader):
    """OSULeaf time series classification problem.

    Example of a univariate problem with equal-length series.

    Parameters
    ----------
    return_mtype: valid Panel mtype str or None, optional (default="pd-multiindex")
        Memory data format specification to return X in, None = "nested_univ" type.
        str can be any supported sktime Panel mtype,
            for list of mtypes, see ``sktime.datatypes.MTYPE_REGISTER``
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        Commonly used specifications:
            - "nested_univ": nested pd.DataFrame, pd.Series in cells
            - "numpy3D"/"numpy3d"/"np3D": 3D np.ndarray (instance, variable, time index)
            - "numpy2d"/"np2d"/"numpyflat": 2D np.ndarray (instance, time index)
            - "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.

    Examples
    --------
    >>> from sktime.datasets.classification.osuleaf import OSULeaf
    >>> X, y = OSULeaf().load()

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
        "is_univariate": True,
        "n_instances": 442,
        "n_instances_train": 200,
        "n_instances_test": 242,
        "n_classes": 6,
    }

    loader_func = load_osuleaf
