import numpy as np
import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.base import BaseTransformer
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks


class _TransformChangeNInstances(BaseTransformer):
    _tags = {
        "object_type": "transformer",  # type of object
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:transform-labels": "None",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "capability:inverse_transform": False,  # can the transformer inverse transform?
        "capability:inverse_transform:range": None,
        "capability:inverse_transform:exact": True,
        # inverting range of inverse transform = domain of invertibility of transform
        "univariate-only": False,  # can the transformer handle multivariate X?
        "X_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],  # which mtypes do _fit/_predict support for X?
        # this can be a Panel mtype even if transform-input is Series, vectorized
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "requires_X": True,  # does X need to be passed in fit?
        "requires_y": False,  # does y need to be passed in fit?
        "enforce_index_type": None,  # index type that needs to be enforced in X/y
        "fit_is_empty": False,  # is fit empty and can be skipped? Yes = True
        "X-y-must-have-same-index": False,  # can estimator handle different X/y index?
        "transform-returns-same-time-index": False,
        # does transform return have the same time index as input X
        "skip-inverse-transform": False,  # is inverse-transform skipped when called?
        "capability:unequal_length": True,
        # can the transformer handle unequal length time series (if passed Panel)?
        "capability:unequal_length:removes": False,
        # is transform result always guaranteed to be equal length (and series)?
        "capability:missing_values": False,  # can estimator handle missing data?
        # todo: rename to capability:missing_values
        "capability:missing_values:removes": False,
        # is transform result always guaranteed to contain no missing values?
        "capability:categorical_in_X": False,
        # does the transformer natively support categorical in exogeneous X?
        "remember_data": False,  # whether all data seen is remembered as self._X
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
    }

    def __init__(self, n=1, random_state=None):
        self.n = n
        self.random_state = random_state
        super().__init__()

        self._should_skip_transform = False

    def _fit(self, X, y=None):
        if X.index.nlevels < 2:
            self._should_skip_transform = True

        return self

    def _transform(self, X, y=None):
        if self._should_skip_transform:
            return X

        rng = np.random.default_rng(self.random_state)
        # series names
        instances_idx = X.index.droplevel(-1).unique()
        # Sample self.n_instances at random

        n = min(self.n, len(instances_idx))
        selected_instances_idx = rng.choice(
            list(range(len(instances_idx))), n, replace=False
        )
        instances = instances_idx[selected_instances_idx]
        return X.loc[X.index.droplevel(-1).isin(instances)].sort_index()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        return [
            {"n": 1},
            {"n": 2},
        ]


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations.base"),
    reason="run test only if transformations base class has changed",
)
@parametrize_with_checks([_TransformChangeNInstances])
def test_transformation_can_return_new_instances(obj, test_name):
    """
    Test if transformation can change the number of instances.
    """
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)
