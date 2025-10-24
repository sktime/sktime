"""Test benchmarking using dummy catalogues."""

import pandas as pd
import pytest

from sktime.benchmarking.classification import ClassificationBenchmark
from sktime.benchmarking.forecasting import ForecastingBenchmark
from sktime.catalogues import (
    DummyClassificationCatalogue,
    DummyForecastingCatalogue,
)
from sktime.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("sktime.catalogues"),
    reason="run test only if catalogues module has changed",
)
def test_benchmarking_dummy_forecasting_catalogue(tmp_path):
    "Test benchmarking with a dummy forecasting catalogue."
    benchmark = ForecastingBenchmark()
    catalogue = DummyForecastingCatalogue()

    benchmark.add_catalogue(catalogue)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(
            [
                "[dataset=cif_2016_dataset]_[cv_splitter=ExpandingWindowSplitter]",
                "[dataset=hospital_dataset]_[cv_splitter=ExpandingWindowSplitter]_2",
            ],
            name="validation_id",
        ),
        results_df["validation_id"],
    )


@pytest.mark.skipif(
    not run_test_module_changed("sktime.catalogues"),
    reason="run test only if catalogues module has changed",
)
def test_benchmarking_dummy_classification_catalogue(tmp_path):
    "Test benchmarking with a dummy classification catalogue."
    benchmark = ClassificationBenchmark()
    catalogue = DummyClassificationCatalogue()

    benchmark.add_catalogue(catalogue)

    results_file = tmp_path / "results.csv"
    results_df = benchmark.run(results_file)

    pd.testing.assert_series_equal(
        pd.Series(
            [
                "[dataset=Beef]_[cv_splitter=KFold]",
                "[dataset=ArrowHead]_[cv_splitter=KFold]_2",
            ],
            name="validation_id",
        ),
        results_df["validation_id"],
    )
