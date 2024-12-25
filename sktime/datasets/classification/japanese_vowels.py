"""JapaneseVowels dataset."""

from sktime.datasets._single_problem_loaders import load_japanese_vowels
from sktime.datasets.classification._base import _ClassificationDatasetFromLoader


class JapaneseVowels(_ClassificationDatasetFromLoader):
    """JapaneseVowels time series classification problem.

    Example of a multivariate problem with unequal-length series.

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
            - "pd-multiindex": pd.DataFrame with 2-level (instance, time) MultiIndex
        Exception is raised if the data cannot be stored in the requested type.

    Examples
    --------
    >>> from sktime.datasets.classification.japanese_vowels import JapaneseVowels
    >>> X, y = JapaneseVowels().load()

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
