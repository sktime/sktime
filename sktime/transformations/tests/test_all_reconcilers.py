"""Unit tests common to all transformers."""

__author__ = ["felipeangelimvieira"]
__all__ = []

import pytest

from sktime.tests.test_all_estimators import BaseFixtureGenerator, QuickTester
from sktime.transformations.hierarchical.aggregate import Aggregator
from sktime.utils._testing.hierarchical import _bottom_hier_datagen
from sktime.utils._testing.panel import _make_panel


class ReconciliationTransformerFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for reconciler tests.


    Fixtures parameterized
    ----------------------
    estimator_class: estimator inheriting from BaseTransformer and
        reconciler object type.
    estimator_instance: instance of estimator inheriting from BaseTransformer
        and reconciler object type.
    scenario: instance of TestScenario
    ranges over all scenarios returned by retrieve_scenarios
    """

    # note: this should be separate from TestAllTransformers
    #   additional fixtures, parameters, etc should be added here
    #   TestAllTransformers should contain the tests only

    estimator_type_filter = "reconciler"


class TestAllReconciliationTransformers(
    ReconciliationTransformerFixtureGenerator, QuickTester
):
    """Module level tests for all sktime transformers."""

    @pytest.mark.parametrize("no_levels", [1, 2, 3, 4])
    @pytest.mark.parametrize("flatten_single_levels", [True, False])
    @pytest.mark.parametrize("unnamed_levels", [True, False])
    @pytest.mark.parametrize("aggregate", [True, False])
    def test_hierarchical_reconcilers(
        self,
        estimator_instance,
        no_levels,
        flatten_single_levels,
        unnamed_levels,
        aggregate,
    ):
        """Test that hierarchical transformers can handle hierarchical data.

        * Test different number of hierarchical levels. The methods should work
            for any number of levels.
        * Test with and without flattening single levels. The methods should
        return the same original number of series.

        """
        import numpy as np
        from pandas.testing import assert_frame_equal

        # If aggregate = False and no_levels=1, we have a single level
        # without __total, which is not a hierarchy. We skip this case.
        if not aggregate and no_levels == 1:
            pytest.skip("No hierarchy with no_levels=1 and aggregate=False.")

        X = _bottom_hier_datagen(
            no_bottom_nodes=5,
            no_levels=no_levels,
            random_seed=123,
        )
        # add aggregate levels

        if aggregate:
            agg = Aggregator(flatten_single_levels=flatten_single_levels)
            X = agg.fit_transform(X)

        if unnamed_levels:
            X.index.names = [None] * X.index.nlevels

        X = X + np.random.normal(0, 10, (X.shape[0], 1))

        # reconcile forecasts
        reconciler = estimator_instance
        Xt = reconciler.fit_transform(X)
        prds = Xt + np.random.normal(0, 10, (Xt.shape[0], 1))
        prds_recon = reconciler._inverse_transform_reconciler(prds)

        # Assert hierarchy detected
        assert not reconciler._no_hierarchy
        # Assert not empty
        assert not prds_recon.empty
        # Assert no Nans
        assert not prds_recon.isnull().values.any()

        # check if we now remove aggregate levels and use Aggregator it is equal
        prds_recon_bottomlevel = Aggregator(False).fit_transform(prds_recon)
        prds_recon_bottomlevel = prds_recon_bottomlevel.loc[prds_recon.index]
        assert_frame_equal(prds_recon, prds_recon_bottomlevel)

    def test_implement_inverse_transform(self, estimator_instance):
        """Test that the reconciler has implemented the inverse_transform method."""
        methods_to_implement = [
            "_inverse_transform_reconciler",
        ]

        for method in methods_to_implement:
            assert method in estimator_instance.__class__.__dict__

    @pytest.mark.parametrize("n_instances", [1, 10])
    def test_behaves_as_identity_if_input_not_hierarchical(
        self, estimator_instance, n_instances
    ):
        """Test that the reconciler behaves as identity when required."""
        X = _make_panel(n_instances=20)

        # reconcile forecasts
        reconciler = estimator_instance
        Xt = reconciler.fit_transform(X)

        assert reconciler._no_hierarchy
        assert Xt.equals(X)

        Xinv = reconciler.inverse_transform(Xt)
        assert Xinv.equals(X)
