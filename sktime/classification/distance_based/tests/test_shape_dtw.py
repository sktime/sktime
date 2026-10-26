"""Tests for ShapeDTW."""

import numpy as np
import pytest

from sktime.classification.distance_based._shape_dtw import ShapeDTW
from sktime.datasets import load_unit_test
from sktime.tests.test_switch import run_test_for_class


def _compound_descriptors(shape_descriptor_functions):
    """Fit ShapeDTW in compound mode, return combined descriptor of instance 0."""
    X, y = load_unit_test(split="train", return_type="nested_univ")
    X, y = X.iloc[:6], y[:6]

    clf = ShapeDTW(
        n_neighbors=1,
        shape_descriptor_function="compound",
        shape_descriptor_functions=shape_descriptor_functions,
        metric_params={"weighting_factor": 1.0},
    )
    clf.fit(X, y)
    # _generate_shape_descriptors reads the fitted weighting_factor
    clf.weighting_factor = 1.0
    combined = clf._generate_shape_descriptors(X)
    return np.asarray(list(combined.iloc[0, 0]), dtype=float)


@pytest.mark.skipif(
    not run_test_for_class(ShapeDTW),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_compound_uses_both_descriptors():
    """Test that compound mode combines two distinct descriptors, see issue #9434.

    ``_combine_data_frames`` iterated over ``second_desc.columns`` but converted
    ``first_desc[x]``, so the second descriptor was discarded and the first was
    concatenated with a weighted copy of itself.
    """
    raw_deriv = _compound_descriptors(["raw", "derivative"])
    raw_raw = _compound_descriptors(["raw", "raw"])

    half = len(raw_deriv) // 2

    # the two halves must differ, since "raw" and "derivative" differ
    assert not np.allclose(raw_deriv[:half], raw_deriv[half:])

    # and a genuinely different second descriptor must change the result
    assert not np.allclose(raw_deriv, raw_raw)

    # sanity checks: the first half is the raw descriptor in both cases, and
    # ("raw", "raw") legitimately does produce two identical halves
    assert np.allclose(raw_deriv[:half], raw_raw[:half])
    assert np.allclose(raw_raw[:half], raw_raw[half:])
