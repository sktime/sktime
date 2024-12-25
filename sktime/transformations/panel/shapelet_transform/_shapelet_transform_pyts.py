"""Shapelet transformer, from pyts."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Abhay-Lejith"]
__all__ = ["ShapeletTransformPyts"]

from sktime.base.adapters._pyts import _PytsAdapter
from sktime.transformations.base import BaseTransformer


class ShapeletTransformPyts(_PytsAdapter, BaseTransformer):
    """Shapelet Transform, from ``pyts``.

    Direct interface to ``pyts.transformation.ShapeletTransform``.

    The Shapelet Transform algorithm extracts shapelets from a data set of time series
    and returns the distances between the shapelets and the time series. A shapelet is
    defined as a subset of a time series, that is a set of values from consecutive time
    points. The distance between a shapelet and a time series is defined as the minimum
    of the distances between this shapelet and all the shapelets of same length
    extracted from this time series. The most discriminative shapelets are selected.
    Two criteria are made available: mutual information and F-scores.

    Parameters
    ----------
    n_shapelets : int or 'auto' (default = 'auto')
        The number of shapelets to keep. If 'auto', `n_timestamps // 2`
        shapelets are considered, where `n_timestamps` is the number of
        time points in the dataset. Note that there might be a smaller
        number of shapelets if fewer than ``n_shapelets`` shapelets have been
        extracted during the search.

    criterion : 'mutual_info' or 'anova' (default = 'mutual_info')
        Criterion to perform the selection of the shapelets.
        'mutual_info' uses the mutual information, while 'anova' use
        the ANOVA F-value.

    window_sizes : array-like or 'auto '(default = 'auto')
        Size of the sliding windows. If 'auto', the range for the
        window sizes is determined automatically.
        Otherwise, all the elements must be either integers or floats.
        In the latter case, each element represents the percentage
        of the size of each time series and must be between 0 and 1; the size
        of the sliding windows will be computed as
        ``np.ceil(window_sizes * n_timestamps)``.

    window_steps : None or array-like (default = None)
        Step of the sliding windows. If None, each ``window_step`` is equal
        to 1. Otherwise, all the elements must be either integers or floats.
        In the latter case, each element represents the percentage of the size
        of each time series and must be between 0 and 1; the step of the
        sliding windows will be computed as
        ``np.ceil(window_steps * n_timestamps)``.
        Must be None if ``window_sizes='auto'``.

    remove_similar : bool (default = True)
        If True, self-similar shapelets are removed, keeping only
        the non-self-similar shapelets with the highest scores. Two
        shapelets are considered to be self-similar if they are
        taken from the the same time series and have at least one
        overlapping index.

    sort : bool (default = False)
        If True, shapelets are sorted in descending order according to
        their associated scores. If False, the order is undefined.

    verbose : int (default = 0)
        Verbosity level when fitting: if non zero, progress messages are
        printed. Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.

    random_state : int, RandomState instance or None (default = None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``. Only used if ``window_sizes='auto'`` in order to
        subsample the dataset to find the best range or if
        ``criterion=='mutual_info'`` to add small noise to the data.

    n_jobs : None or int (default = None)
        The number of jobs to run in parallel for ``fit``.
        If -1, then the number of jobs is set to the number of cores.

    Attributes
    ----------
    shapelets_ : array, shape = (n_shapelets,)
        The array with the selected shapelets.

    indices_ : array, shape = (n_shapelets, 3)
        The indices for the corresponding shapelets in the training set.
        The first column consists of the indices of the samples.
        The second column consists of the starting indices (included)
        of the shapelets. The third column consists of the ending indices
        (excluded) of the shapelets.

    scores_ : array, shape = (n_shapelets,)
        The scores associated to the shapelets. The higher, the more
        discriminant.
        If ``criterion='mutual_info'``, mutual information scores are reported.
        If ``criterion='anova'``, F-scores are reported.

    window_range_ : None or tuple
        Range of the window sizes if ``window_sizes='auto'``. None otherwise.

    References
    ----------
    .. [1] J. Lines, L. M. Davis, J. Hills and A. Bagnall, "A Shapelet
           Transform for Time Series Classification". Data Mining and Knowledge
           Discovery, 289-297 (2012).

    Examples
    --------
    >>> from sktime.transformations.panel.shapelet_transform import (
    ...     ShapeletTransformPyts
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train") # doctest: +SKIP
    >>> stp = ShapeletTransformPyts() # doctest: +SKIP
    >>> stp.fit(X_train,y_train) # doctest: +SKIP
    >>> stp.transform(X_train) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["johannfaouzi", "Abhay-Lejith"],
        # johannfaouzi is author of upstream pyts code
        "python_dependencies": "pyts",
        # univariate-only controls whether internal X can be univariate/multivariate
        # if True (only univariate), always applies vectorization over variables
        "univariate-only": True,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "fit_is_empty": False,
        "y_inner_mtype": "numpy1D",
        "requires_y": True,
    }

    _estimator_attr = "_pyts_shapelet_transform"

    def _get_pyts_class(self):
        """Get pyts class.

        should import and return pyts class
        """
        from pyts.transformation.shapelet_transform import ShapeletTransform

        return ShapeletTransform

    def __init__(
        self,
        n_shapelets="auto",
        criterion="mutual_info",
        window_sizes="auto",
        window_steps=None,
        remove_similar=True,
        sort=False,
        verbose=0,
        random_state=None,
        n_jobs=None,
    ):
        self.n_shapelets = n_shapelets
        self.criterion = criterion
        self.window_sizes = window_sizes
        self.window_steps = window_steps
        self.remove_similar = remove_similar
        self.sort = sort
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

        super().__init__()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params1 = {"criterion": "anova", "n_shapelets": 10}
        params2 = {"window_sizes": [3]}
        return [params1, params2]
