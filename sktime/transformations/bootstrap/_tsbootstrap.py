"""Bootstrapping adapter for tsbootstrap."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["benheid"]

import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation._dependencies import _check_soft_dependencies


class TSBootstrapAdapter(BaseTransformer):
    """Adapter for TSBBootstrap.

    Parameters
    ----------
    bootstrapper : Bootstrapper from tsbootstrap
        default = SlidingWindowSplitter(window_length=3, step_length=1)
        The splitter used for the bootstrap splitting.
    include_actual : bool, default=True
        Whether to include the actual data in the output.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.bootstrap import TSBootstrapAdapter
    >>> from tsbootstrap import MovingBlockBootstrap, MovingBlockBootstrapConfig
    >>> y = load_airline()
    >>> config = MovingBlockBootstrapConfig(10, n_bootstraps=10)
    >>> bootstrap = TSBootstrapAdapter(MovingBlockBootstrap(config))
    >>> result = bootstrap.fit_transform(y)
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Panel",
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "pd.DataFrame",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "capability:inverse_transform": False,
        "skip-inverse-transform": True,  # is inverse-transform skipped when called?
        "univariate-only": False,  # can the transformer handle multivariate X?
        "handles-missing-data": True,  # can estimator handle missing data?
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "transform-returns-same-time-index": False,
        "python_dependencies": ["tsbootstrap>=0.0.3"],
    }

    def __init__(
        self,
        bootstrapper,
        include_actual=True,
    ):
        super().__init__()
        self.bootstrapper = bootstrapper
        self.include_actual = include_actual

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
        bootstrapped_samples = self.bootstrapper.bootstrap(X, test_ratio=0)

        bootstrapped_samples = [pd.DataFrame(sample) for sample in bootstrapped_samples]

        boostrapped_df = pd.concat(
            bootstrapped_samples,
            keys=[f"synthetic_{i}" for i in range(len(bootstrapped_samples))],
        )

        if self.include_actual:
            X.index = pd.MultiIndex.from_product([["actual"], X.index])
            return pd.concat([X, boostrapped_df], axis=0)
        else:
            return boostrapped_df

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
        if not _check_soft_dependencies("tsbootstrap", severity="none"):
            return []

        from tsbootstrap.block_bootstrap import (
            BaseSieveBootstrapConfig,
            MovingBlockBootstrap,
            MovingBlockBootstrapConfig,
            WholeSieveBootstrap,
        )

        params = [
            {
                "tsbootstrapper": WholeSieveBootstrap(
                    BaseSieveBootstrapConfig(
                        10,
                        n_bootstraps=10,
                    )
                )
            },
            {
                "tsbootstrapper": MovingBlockBootstrap(MovingBlockBootstrapConfig()),
            },
        ]

        return params
