"""Bootstrapping adapter for tsbootstrap."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["benheid"]

import pandas as pd

from sktime.transformations.base import BaseTransformer


class TSBootstrapAdapter(BaseTransformer):
    """Adapter for TSBootstrap.

    The bootstrap samples will be returned as a Panel with the first level
    being integer index of the synthetic sample.

    For hierarchical data, the bootstrap index will be added as the index
    at integer position -2.

    Parameters
    ----------
    bootstrap : bootstrap from tsbootstrap
        The splitter used for the bootstrap splitting.
    include_actual : bool, default=False
        Whether to include the actual data in the output.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.bootstrap import TSBootstrapAdapter
    >>> from tsbootstrap import (
    ...    MovingBlockBootstrap,
    ...    MovingBlockBootstrapConfig
    ... ) # doctest: +SKIP
    >>> y = load_airline()
    >>> config = MovingBlockBootstrapConfig(10, n_bootstraps=10)  # doctest: +SKIP
    >>> bootstrap = TSBootstrapAdapter(MovingBlockBootstrap(config))  # doctest: +SKIP
    >>> result = bootstrap.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "benheid",
        "python_dependencies": ["tsbootstrap>=0.1.0"],
        # estimator type
        # --------------
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
        "transform-returns-same-time-index": True,
    }

    def __init__(
        self,
        bootstrap,
        include_actual=False,
    ):
        self.bootstrap = bootstrap
        self.include_actual = include_actual
        super().__init__()

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
        bootstrapped_samples = self.bootstrap.bootstrap(X, test_ratio=0)

        def wrap_df(spl):
            return pd.DataFrame(spl, index=X.index, columns=X.columns)

        bootstrapped_samples = [wrap_df(sample) for sample in bootstrapped_samples]
        spl_keys = [f"synthetic_{i}" for i in range(len(bootstrapped_samples))]

        if self.include_actual:
            X_actual = X.copy()
            bootstrapped_samples = [X_actual] + bootstrapped_samples
            spl_keys = ["actual"] + spl_keys

        return pd.concat(bootstrapped_samples, keys=spl_keys, axis=0)

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
        from sktime.utils.dependencies import _check_soft_dependencies

        deps = cls.get_class_tag("python_dependencies")

        if _check_soft_dependencies(deps, severity="none"):
            from tsbootstrap import BlockBootstrap, MovingBlockBootstrap

            params = [
                {"bootstrap": BlockBootstrap(n_bootstraps=10)},
                {
                    "bootstrap": MovingBlockBootstrap(n_bootstraps=10, block_length=4),
                    "include_actual": True,
                },
            ]
        else:
            params = {"bootstrap": "dummy"}

        return params
