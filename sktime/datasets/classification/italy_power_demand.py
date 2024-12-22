"""ItalyPowerDemand dataset."""

from sktime.datasets._single_problem_loaders import load_italy_power_demand
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class ItalyPowerDemand(_ClassificationDatasetFromLoader):
    """ItalyPowerDemand time series classification problem.

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
    >>> from sktime.datasets.classification.italy_power_demand import ItalyPowerDemand
    >>> X, y = ItalyPowerDemand().load()

    Notes
    -----
    Dimensionality:     univariate
    Series length:      24
    Train cases:        67
    Test cases:         1029
    Number of classes:  2

    The data was derived from twelve monthly electrical power demand time series from
    Italy and was first used in the paper "Intelligent Icons: Integrating Lite-Weight
    Data Mining and Visualization into GUI Operating Systems". The classification task
    is to distinguish days from October to March (inclusive) from April to September.

    Dataset details:
    http://timeseriesclassification.com/description.php?Dataset=ItalyPowerDemand
    """

    _tags = {
        "is_univariate": True,
        "n_instances": 1096,
        "n_instances_train": 67,
        "n_instances_test": 1029,
        "n_classes": 2,
    }

    loader_func = load_italy_power_demand
