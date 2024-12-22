"""GunPoint dataset."""

from sktime.datasets._single_problem_loaders import load_gunpoint
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class GunPoint(_ClassificationDatasetFromLoader):
    """GunPoint time series classification problem.

    Example of a univariate problem with equal-length series.

    Parameters
    ----------
    return_mtype: valid Panel mtype str or None, optional (default="pd-multiindex")
        Memory data format specification to return X in, None = "nested_univ" type.
        str can be any supported sktime Panel mtype,
            for list of mtypes, see datatypes.MTYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        Commonly used specifications:
            - "nested_univ": nested pd.DataFrame, pd.Series in cells
            - "numpy3D"/"numpy3d"/"np3D": 3D np.ndarray (instance, variable, time index)
            - "numpy2d"/"np2d"/"numpyflat": 2D np.ndarray (instance, time index)
            - "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.

    Examples
    --------
    >>> from sktime.datasets.classification.gunpoint import GunPoint
    >>> X, y = GunPoint().load()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      150
    Train cases:        50
    Test cases:         150
    Number of classes:  2

    This dataset involves one female actor and one male actor making a motion with their
    hand. The two classes are: Gun-Draw and Point. For Gun-Draw, the actors have their
    hands by their sides. They draw a replicate gun from a hip-mounted holster, point it
    at a target for approximately one second, then return the gun to the holster, and
    their hands to their sides. For Point, the actors have their hands by their sides.
    They point with their index fingers to a target for approximately one second, and
    then return their hands to their sides. For both classes, we tracked the centroid
    of the actor's right hands in both X- and Y-axes, which appear to be highly
    correlated. The data in the archive is just the X-axis.

    Dataset details: http://timeseriesclassification.com/description.php?Dataset=GunPoint
    """

    _tags = {
        "is_univariate": True,
        "n_instances": 200,
        "n_instances_train": 50,
        "n_instances_test": 150,
        "n_classes": 2,
    }

    loader_func = load_gunpoint
