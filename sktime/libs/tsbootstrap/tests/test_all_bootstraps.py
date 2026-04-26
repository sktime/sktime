"""Automated tests based on the skbase test suite template."""

import numpy as np
import pytest
from skbase.testing import QuickTester

from tsbootstrap.tests.test_all_estimators import (
    BaseFixtureGenerator,
    PackageConfig,
)


class TestAllBootstraps(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all bootstrap algorithms in tsbootstrap."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # which object types are generated; None=all, or class (passed to all_objects)
    object_type_filter = "bootstrap"

    def test_n_bootstraps(self, object_instance):
        """Tests handling of n_bootstraps parameter."""
        cls_name = object_instance.__class__.__name__

        params = object_instance.get_params()

        if "n_bootstraps" not in params:
            raise ValueError(
                f"{cls_name} is a bootstrap algorithm and must have "
                "n_bootstraps parameter, but it does not."
            )

        n_bootstraps = params["n_bootstraps"]

        get_n_bootstraps = object_instance.get_n_bootstraps()

        if not get_n_bootstraps == n_bootstraps:
            raise ValueError(
                f"{cls_name}.get_n_bootstraps() returned {get_n_bootstraps}, "
                f"but n_bootstraps parameter is {n_bootstraps}. "
                "These should be equal."
            )

    def test_bootstrap_input_output_contract(self, object_instance, scenario):
        """Tests that output of bootstrap method is as specified."""
        import types

        cls_name = object_instance.__class__.__name__

        result = scenario.run(object_instance, method_sequence=["bootstrap"])

        if not isinstance(result, types.GeneratorType):
            raise TypeError(
                f"{cls_name}.bootstrap did not return a generator, "
                f"but instead returned {type(result)}."
            )
        result = list(result)

        n_timepoints, n_vars = scenario.args["bootstrap"]["X"].shape
        n_bs_expected = object_instance.get_params()["n_bootstraps"]

        # if return_index=True, result is a tuple of (dataframe, index)
        # results are generators, so we need to convert to list
        if scenario.get_tag("return_index", False):
            if not all(isinstance(x, tuple) for x in result):
                raise TypeError(
                    f"{cls_name}.bootstrap did not return a generator of tuples, "
                    f"but instead returned {[type(x) for x in result]}."
                )
            if not all(len(x) == 2 for x in result):
                raise ValueError(
                    f"{cls_name}.bootstrap did not return a generator of tuples, "
                    f"but instead returned {[len(x) for x in result]}."
                )

            bss = [x[0] for x in result]
            index = [x[1] for x in result]

        else:
            bss = result

        if not len(bss) == n_bs_expected:
            raise ValueError(
                f"{cls_name}.bootstrap did not yield the expected number of "
                f"bootstrap samples. Expected {n_bs_expected}, but got {len(bss)}."
            )

        if not all(isinstance(bs, np.ndarray) for bs in bss):
            raise ValueError(
                f"{cls_name}.bootstrap must yield numpy.ndarray, "
                f"but yielded {[type(bs) for bs in bss]} instead."
                "Not all bootstraps are numpy arrays."
            )

        if not all(bs.ndim == 2 for bs in bss):
            raise ValueError(
                f"{cls_name}.bootstrap yielded arrays with unexpected number of "
                "dimensions. All bootstrap samples should have 2 dimensions."
            )

        if not all(bs.shape[0] == n_timepoints for bs in bss):
            raise ValueError(
                f"{cls_name}.bootstrap yielded arrays unexpected length,"
                f" {[bs.shape[0] for bs in bss]}. "
                f"All bootstrap samples should have the same, "
                f"expected length: {n_timepoints}."
            )
        if not all(bs.shape[1] == n_vars for bs in bss):
            raise ValueError(
                f"{cls_name}.bootstrap yielded arrays with unexpected number of "
                f"variables, {[bs.shape[1] for bs in bss]}. "
                "All bootstrap samples should have the same, "
                f"expected number of variables: {n_vars}."
            )

        if scenario.get_tag("return_index", False):
            if not all(isinstance(ix, np.ndarray) for ix in index):
                raise ValueError(
                    f"{cls_name}.bootstrap did not return a generator of tuples, "
                    f"but instead returned {[type(ix) for ix in index]}."
                )
            if not all(ix.ndim == 1 for ix in index):
                raise ValueError(
                    f"{cls_name}.bootstrap yielded arrays with unexpected number of "
                    "dimensions. All bootstrap samples should have 1 dimension."
                )
            if not all(ix.shape[0] == n_timepoints for ix in index):
                raise ValueError(
                    f"{cls_name}.bootstrap yielded arrays unexpected length,"
                    f" {[ix.shape[0] for ix in index]}. "
                    f"All bootstrap samples should have the same, "
                    f"expected length: {n_timepoints}."
                )

    @pytest.mark.parametrize("test_ratio", [0.2, 0.0, 0.314, 0])
    def test_bootstrap_test_ratio(self, object_instance, scenario, test_ratio):
        """Tests that the passing bootstrap test ratio has specified effect."""
        cls_name = object_instance.__class__.__name__

        bs_kwargs = scenario.args["bootstrap"]
        result = object_instance.bootstrap(test_ratio=test_ratio, **bs_kwargs)
        result = list(result)

        n_timepoints, n_vars = bs_kwargs["X"].shape
        n_bs_expected = object_instance.get_params()["n_bootstraps"]

        expected_length = np.floor(n_timepoints * (1 - test_ratio)).astype(int)

        # if return_index=True, result is a tuple of (dataframe, index)
        # results are generators, so we need to convert to list
        if scenario.get_tag("return_index", False):
            if not all(isinstance(x, tuple) for x in result):
                raise TypeError(
                    f"{cls_name}.bootstrap did not return a generator of tuples, "
                    f"but instead returned {[type(x) for x in result]}."
                )
            if not all(len(x) == 2 for x in result):
                raise ValueError(
                    f"{cls_name}.bootstrap did not return a generator of tuples, "
                    f"but instead returned {[len(x) for x in result]}."
                )

            bss = [x[0] for x in result]
            index = [x[1] for x in result]

        else:
            bss = list(result)

        if not len(bss) == n_bs_expected:
            raise ValueError(
                f"{cls_name}.bootstrap did not yield the expected number of "
                f"bootstrap samples. Expected {n_bs_expected}, but got {len(bss)}."
            )

        if not all(isinstance(bs, np.ndarray) for bs in bss):
            raise ValueError(
                f"{cls_name}.bootstrap must yield numpy.ndarray, "
                f"but yielded {[type(bs) for bs in bss]} instead."
                "Not all bootstraps are numpy arrays."
            )

        if not all(bs.ndim == 2 for bs in bss):
            raise ValueError(
                f"{cls_name}.bootstrap yielded arrays with unexpected number of "
                "dimensions. All bootstrap samples should have 2 dimensions."
            )

        if not all(bs.shape[0] == expected_length for bs in bss):
            raise ValueError(
                f"{cls_name}.bootstrap yielded arrays unexpected length,"
                f" {[bs.shape[0] for bs in bss]}. "
                f"All bootstrap samples should have the same, "
                f"expected length: {expected_length}."
            )
        if not all(bs.shape[1] == n_vars for bs in bss):
            raise ValueError(
                f"{cls_name}.bootstrap yielded arrays with unexpected number of "
                f"variables, {[bs.shape[1] for bs in bss]}. "
                "All bootstrap samples should have the same, "
                f"expected number of variables: {n_vars}."
            )

        if scenario.get_tag("return_index", False):
            if not all(isinstance(ix, np.ndarray) for ix in index):
                raise ValueError(
                    f"{cls_name}.bootstrap did not return a generator of tuples, "
                    f"but instead returned {[type(ix) for ix in index]}."
                )
            if not all(ix.ndim == 1 for ix in index):
                raise ValueError(
                    f"{cls_name}.bootstrap yielded arrays with unexpected number of "
                    "dimensions. All bootstrap samples should have 1 dimension."
                )
            if not all(ix.shape[0] == expected_length for ix in index):
                raise ValueError(
                    f"{cls_name}.bootstrap yielded arrays unexpected length,"
                    f" {[ix.shape[0] for ix in index]}. "
                    f"All bootstrap samples should have the same, "
                    f"expected length: {expected_length}."
                )
