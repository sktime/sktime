"""BasicMotions dataset."""

from sktime.datasets._single_problem_loaders import load_basic_motions
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class BasicMotions(_ClassificationDatasetFromLoader):
    """BasicMotions time series classification problem.

    Example of a multivariate problem with equal-length series.

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
            - "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.
        Note: For this multivariate dataset, return_type="numpy2d" or "numpyflat" is
        not supported.

    Examples
    --------
    >>> from sktime.datasets.classification.basic_motions import BasicMotions
    >>> X, y = BasicMotions().load()

    Notes
    -----
    Dimensionality:     multivariate, 6 variables
    Series length:      100
    Train cases:        40
    Test cases:         40
    Number of classes:  4

    The data was generated as part of a student project where four students performed
    four activities whilst wearing a smartwatch. The watch collects 3D accelerometer
    and 3D gyroscope data. It consists of four classes: walking, resting,
    running, and badminton. Participants were required to record motion a total of
    five times, and the data is sampled once every tenth of a second, for a ten-second
    period.

    Dataset details: http://timeseriesclassification.com/description.php?Dataset=BasicMotions
    """

    _tags = {
        "is_univariate": False,
        "n_instances": 80,
        "n_instances_train": 40,
        "n_instances_test": 40,
        "n_classes": 4,
    }

    loader_func = load_basic_motions
