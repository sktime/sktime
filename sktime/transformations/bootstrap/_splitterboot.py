# -*- coding: utf-8 -*-
"""Bootstrapping method based on any sktime splitter."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class SplitterBootstrapTransformer(BaseTransformer):
    """Splitter based Bootstrapping method for synthetic time series generation.

    A generalized form of bootstrap based on an sktime splitter.

    Any sktime splitter can be passed as a component to this transformer,
    which will then produce for each series in the input of `transform`
    a panel of time series with the train and/or test sub-series.

    The output of `transform` will have additional levels:

    * all levels of the `transform` input
    * an additional integer indexed top level, indicating the number of the fold
    * if `split="train"` or `split="test"`, no further levels
    * if `split="both"`, the second top level contains strings `"train"` and `"test"`
      to indicate train or test fold from the split

    For instance, if `split="train"`, and there is a single original series `X`,
    the output of `transform` will have a top level )level 0) with integer index
    ranging from 0 to `splitter.get_n_splits(X)-1`.

    Parameters
    ----------
    splitter : optional, sktime splitter, BaseSplitter descendant
        default = SlidingWindowSplitter(window_length=3, step_length=1)
        The splitter used for the bootstrap splitting.
    split : str, one of "train" (default), "test", and "both"
        Determines which fold is returned as new instances in the panel.
        "train" - the training folds; "test" - the test folds;
        "both" - both training and test folds, and an additional string level with
        possible values `"train"` and `"test"` is present

    See Also
    --------
    sktime.transformations.bootstrap.MovingBlockBootstrapTransformer :
        Similar logic to sliding window splitter, but with random windows.

    Examples
    --------
    >>> from sktime.transformations.bootstrap import SplitterBootstrapTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = SplitterBootstrapTransformer(split="both")
    >>> y_hat = transformer.fit_transform(y)
    """

    _tags = {
        # todo: what is the scitype of X: Series, or Panel
        "scitype:transform-input": "Series",
        # todo: what scitype is returned: Primitives, Series, Panel
        "scitype:transform-output": "Panel",
        # todo: what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        # X_inner_mtype can be Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": True,  # can the transformer handle multivariate X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
    }

    def __init__(self, splitter=None, split="train"):
        self.splitter = splitter
        self.split = split

        super(SplitterBootstrapTransformer, self).__init__()

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
        split = self.split

        if splitter is None:
            from sktime.forecasting.model_selection import SlidingWindowSplitter

            splitter = SlidingWindowSplitter(fh=[1], window_length=3, step_length=1)

        if split == "train":
            splits = [x[0] for x in splitter.split_series(X)]
        elif split == "test":
            splits = [x[1] for x in splitter.split_series(X)]
        elif split == "both":
            s = splitter.split_series(X)
            splits = [pd.concat(x, keys=["train", "test"]) for x in s]
        else:
            raise ValueError(
                "split in SplitterBootstrapTransformer must be one of"
                'the strings "train", "test", or "both", '
                f"but found {split}"
            )

        return pd.concat(splits, axis=0, keys=pd.RangeIndex(len(splits)))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

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
        from sktime.forecasting.model_selection import ExpandingWindowSplitter

        params = [
            {},
            {
                "splitter": ExpandingWindowSplitter(initial_window=1),
                "split": "test",
            },
        ]

        return params
