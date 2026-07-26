"""Tecator dataset."""

from sktime.datasets._single_problem_loaders import load_tecator
from sktime.datasets.regression._base import _RegressionDatasetFromLoader

__all__ = ["Tecator"]


class Tecator(_RegressionDatasetFromLoader):
    """Load the Tecator time series regression problem.

    Returns
    -------
    X: sktime data container, following mtype specification ``return_type``
        The time series data for the problem, with n instances
    y: 1D numpy array of length n, only returned if return_X_y if True
        The target values for each time series instance in X
        If return_X_y is False, y is appended to X instead.

    Examples
    --------
    >>> from sktime.datasets.regression import Tecator
    >>> dataset = Tecator()
    >>> X, y = dataset.load("X", "y")

    Notes
    -----
    Dimensionality:     univariate
    Series length:      100
    Train cases:        172
    Test cases:         43

    The purpose of this dataset is to measure the fat content of meat based off its near
      infrared absorbance spectrum.
    The absorbance spectrum is measured in the wavelength range of 850 nm to 1050 nm.
    The fat content is measured by standard chemical analysis methods.
    The dataset contains 215 samples of meat, each with 100 spectral measurements.
    For more information see:
    https://www.openml.org/search?type=data&sort=runs&id=505&status=active

    References
    ----------
    [1] C.Borggaard and H.H.Thodberg, "Optimal Minimal Neural Interpretation of Spectra"
    , Analytical Chemistry 64 (1992), p 545-551.
    [2] H.H.Thodberg, "Ace of Bayes: Application of Neural Networks with Pruning"
    Manuscript 1132, Danish Meat Research Institute (1993), p 1-12.
    """

    _tags = {
        "name": "tecator",
        "n_splits": 1,
        "is_univariate": True,
        "is_equal_length": True,
        "has_nans": False,
        "n_instances": 215,
        "n_instances_train": 172,
        "n_instances_test": 43,
    }

    loader_func = load_tecator
