"""Tests for checking composites with categorical data."""

import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.compose import ColumnEnsembleTransformer
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.boxcox import BoxCoxTransformer
from sktime.transformations.series.subset import ColumnSelect
from sktime.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations")
    or _check_soft_dependencies("scikit-learn<1.6", severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_pipeline_with_categorical():
    """Test pipeline which can handle categorical inputs.

    Estimators used are ColumnSelect and OneHotEncoder which can both handle
    categorical so no error is expected to be raised.
    """

    y = pd.DataFrame({"var_0": [1, 2, 3, 4, 5, 6]})
    X = pd.DataFrame(
        {"var_1": ["a", "b", "c", "a", "b", "c"], "var_2": [1, 2, 3, 4, 5, 6]}
    )

    enc = TabularToSeriesAdaptor(OneHotEncoder(), pooling="global")
    pipeline = ColumnSelect(columns=["var_1"]) * enc

    pipeline.fit_transform(X, y)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.transformations")
    or _check_soft_dependencies("scikit-learn<1.6", severity="none"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ColumnEnsemble_with_categorical():
    df = pd.DataFrame(
        {"categorical": ["a", "b", "c", "a", "b", "c"], "num_col": [1, 2, 3, 4, 5, 6]}
    )

    encoder = TabularToSeriesAdaptor(LabelEncoder(), pooling="global")
    trafo = ColumnEnsembleTransformer(
        transformers=[
            ("trafo", BoxCoxTransformer(), "num_col"),
            ("encoder", encoder, "categorical"),
        ]
    )
    trafo.fit_transform(df)
