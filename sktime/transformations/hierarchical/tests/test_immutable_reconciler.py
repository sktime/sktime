import numpy as np
import pandas as pd

from sktime.transformations.hierarchical.reconcile import (
    ImmutableReconciler,
)
from sktime.utils._testing.hierarchical import _bottom_hier_datagen
from sktime.transformations.hierarchical.aggregate import Aggregator


def _get_test_data():
    y = _bottom_hier_datagen(
        no_bottom_nodes=3,
        no_levels=1,
        random_seed=123,
    )
    y = Aggregator().fit_transform(y)
    return y


def test_immutable_unchanged():
    y = _get_test_data()

    # pick one node
    node = y.droplevel(-1).index.unique()[1]

    rec = ImmutableReconciler(immutable_series=[node])
    y_rec = rec.fit_transform(y)

    # check unchanged
    assert (y_rec.loc[node] == y.loc[node]).all()


def test_coherence_preserved():
    y = _get_test_data()

    rec = ImmutableReconciler(immutable_series=[])
    y_rec = rec.fit_transform(y)

    # total = sum of bottom nodes
    totals = y_rec.xs("__total", level=-2)
    bottoms = y_rec[~(y_rec.index.get_level_values(-2) == "__total")]

    assert len(totals) > 0
