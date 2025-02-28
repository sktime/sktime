"""Bootstrapping method based on any sktime splitter."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

from sktime.transformations.base import BaseTransformer


class SplitterBootstrapTransformer(BaseTransformer):
    """Splitter based Bootstrapping method for synthetic time series generation.

    A generalized form of bootstrap based on an sktime splitter.

    Any sktime splitter can be passed as a component to this transformer,
    which will then produce for each series in the input of ``transform``
    a panel of time series with the train and/or test sub-series.

    The output of ``transform`` will have additional levels:

    * all levels of the ``transform`` input
    * an additional integer indexed top level, indicating the number of the sample
      note: this is in general the number of the *sample* and corresponds to the
      number of the *fold* only in the deterministic, exhaustive case
    * if ``split="train"`` or ``split="test"``, no further levels
    * if ``split="both"``, the second top level contains strings ``"train"`` and
    ``"test"``
      to indicate train or test fold from the split

    For instance, if ``split="train"``, and there is a single original series ``X``,
    the output of ``transform`` will have a top level (level 0) with integer index
    ranging from 0 to ``splitter.get_n_splits(X)-1``.

    The splitter can be exhaustive and deterministic, or random.
    By default, exhaustive ordered samples are returned (deterministic).
    Randomness is controlled by the following parameters:

    * ``shuffle`` (by default off) applies random uniform shuffling to the instances
    * ``subsample`` (by default off) applies sub-sampling with or without replacement
    * ``replace`` (by default ``False``) selects sub-sampling with or without
    replacement

    Caution: the instance index of the ``transform`` output will correspond to
    the split index only if ``shuffle=False`` and ``subsample=None`` (unless by
    coincidence)

    Parameters
    ----------
    splitter : optional, sktime splitter, BaseSplitter descendant
        default = SlidingWindowSplitter(window_length=3, step_length=1)
        The splitter used for the bootstrap splitting.
    fold : str, one of "train" (default), "test", and "both"
        Determines which fold is returned as new instances in the panel.
        "train" - the training folds; "test" - the test folds;
        "both" - both training and test folds, and an additional string level with
        possible values ``"train"`` and ``"test"`` is present
    shuffle : bool, default=False
        whether to shuffle the order of folds uniformly at random before returning
        if not, folds will be returned in the ordering defined by the splitter
    subsample : optional, int or float, default = None
        if provided, subsamples the folds returned uniformly at random
        ``int`` = subsample of that size will be returned (or full sample if smaller)
        ``float``, must be between 0 and 1 = subsample of that fraction is returned
        Note: integer 1 selects *one* series; float 1 selects number in ``splitter``
        many
    replace : bool, default=True; only used if ``subsample=True``
        whether sampling, if ``subsample`` is provided is with or without replacement
        ``True`` = with replacement, ``False`` = without replacement
    random_state : int, np.random.RandomState or None (default)
        Random seed for the estimator
        if ``None``, ``numpy`` environment random seed is used
        if ``int``, passed to ``numpy`` ``RandomState`` as seed
        if ``RandomState``, will be used as random generator

    See Also
    --------
    sktime.transformations.bootstrap.MovingBlockBootstrapTransformer :
        Similar logic to sliding window splitter, with bootstrap random windows.

    Examples
    --------
    >>> from sktime.transformations.bootstrap import SplitterBootstrapTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SplitterBootstrapTransformer(fold="both")
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        # what is the scitype of X: Series, or Panel
        "scitype:transform-input": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-output": "Panel",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # X_inner_mtype can be Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def __init__(
        self,
        splitter=None,
        fold="train",
        shuffle=False,
        subsample=None,
        replace=True,
        random_state=None,
    ):
        self.splitter = splitter
        self.fold = fold
        self.shuffle = shuffle
        self.subsample = subsample
        self.replace = replace
        self.random_state = random_state

        super().__init__()

        self._rng = check_random_state(random_state)

        # if split == "both":
        #     self.set_tags(**{"scitype:transform-output": "Hierarchical"})

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        splitter = self.splitter
        fold = self.fold
        shuffle = self.shuffle
        subsample = self.subsample
        replace = self.replace
        rng = self._rng

        if splitter is None:
            from sktime.split import SlidingWindowSplitter

            splitter = SlidingWindowSplitter(fh=[1], window_length=3, step_length=1)

        if fold == "train":
            splits = [x[0] for x in splitter.split_series(X)]
        elif fold == "test":
            splits = [x[1] for x in splitter.split_series(X)]
        elif fold == "both":
            s = splitter.split_series(X)
            splits = [pd.concat(x, keys=["train", "test"]) for x in s]
        else:
            raise ValueError(
                "fold in SplitterBootstrapTransformer must be one of"
                'the strings "train", "test", or "both", '
                f"but found {fold}"
            )

        if subsample is None:
            size = len(splits)
        elif isinstance(subsample, float):
            size = int(len(splits) * subsample)
        else:
            size = subsample

        if subsample is not None or shuffle:
            subs = rng.choice(range(len(splits)), size=size, replace=replace)
            if not shuffle:
                subs = np.sort(subs)
            splits = [splits[x] for x in subs]

        pd_splits = pd.concat(splits, axis=0, keys=pd.RangeIndex(len(splits)))
        return pd_splits

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.split import ExpandingWindowSplitter

        params = [
            {},
            {
                "splitter": ExpandingWindowSplitter(initial_window=1),
                "fold": "test",
                "shuffle": True,
                "subsample": 0.5,
                "replace": True,
            },
            {
                "splitter": ExpandingWindowSplitter(initial_window=2),
                "fold": "train",
                "shuffle": False,
                "subsample": 3,
                "replace": False,
            },
        ]

        return params
