"""JapaneseVowels dataset."""

from sktime.datasets._single_problem_loaders import load_japanese_vowels
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class JapaneseVowels(_ClassificationDatasetFromLoader):
    """JapaneseVowels time series classification problem.

    Example of a multivariate problem with unequal-length series.

    Examples
    --------
    >>> from sktime.datasets.classification import JapaneseVowels
    >>> X, y = JapaneseVowels().load("X", "y")

    Notes
    -----
    Dimensionality:     multivariate, 12 variables
    Series length:      7-29 (variable length)
    Train cases:        270
    Test cases:         370
    Number of classes:  9

    A UCI Archive dataset. Nine Japanese male speakers were recorded saying the vowels
    'a' and 'e'. A 12-degree linear prediction analysis is applied to the raw recordings
    to obtain time series with 12 dimensions and varying lengths between 7 and 29.
    The classification task is to predict the speaker. Each instance is a transformed
    utterance with a single class label attached (labels 1 to 9).

    Reference:
    M. Kudo, J. Toyama, and M. Shimbo. (1999). "Multidimensional Curve Classification
    Using Passing-Through Regions". Pattern Recognition Letters, Vol. 20, No. 11-13,
    pages 1103-1111.

    Dataset details:
    http://timeseriesclassification.com/description.php?Dataset=JapaneseVowels
    """

    _tags = {
        "name": "japanese_vowels",
        "n_splits": 1,
        "is_univariate": False,
        "has_nans": False,
        "n_instances": 640,
        "n_instances_train": 270,
        "n_instances_test": 370,
        "n_classes": 9,
    }

    loader_func = load_japanese_vowels
