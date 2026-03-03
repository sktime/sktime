"""Tests for update capability tags and behaviour in transformers."""

from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose import ColumnwiseTransformer, TransformerPipeline
from sktime.transformations.hierarchical.reconcile import Reconciler
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.transformations.series.lag import Lag, ReducerTransform


def test_base_transformer_update_tag_default_false():
    """BaseTransformer should default to capability:update == False."""

    class DummyTransformer(BaseTransformer):
        pass

    est = DummyTransformer()
    assert est.get_tag("capability:update") is False


def test_transformers_with_custom_update_have_update_tag_true():
    """Transformers with custom _update should advertise capability:update."""
    for cls in [Detrender, Deseasonalizer, Lag, ReducerTransform]:
        est = cls.get_test_params()[0] if hasattr(cls, "get_test_params") else {}
        est = cls(**(est if isinstance(est, dict) else {}))
        assert est.get_tag("capability:update") is True


def test_meta_transformers_advertise_update_capability():
    """Meta-transformers that implement update should advertise capability:update."""
    base = Detrender()
    pipe = TransformerPipeline([base])
    colwise = ColumnwiseTransformer(Detrender())

    assert pipe.get_tag("capability:update") is True
    assert colwise.get_tag("capability:update") is True


def test_fit_is_empty_implies_update_is_noop():
    """If fit_is_empty is True and remember_data is False, update should be a no-op."""
    # Lag with remember_data=False sets fit_is_empty tag to True
    est = Lag(lags=1, remember_data=False)
    assert est.get_tag("fit_is_empty") is True

    est.fit([[1], [2], [3]])
    est_before = est.__dict__.copy()
    est.update([[4], [5], [6]])
    # no new attributes should be introduced by update in this configuration
    assert est_before.keys() == est.__dict__.keys()


def test_reconciler_has_update_tag_false():
    """Reconciler does not implement an update beyond base behaviour."""
    est = Reconciler()
    assert est.get_tag("capability:update") is False
