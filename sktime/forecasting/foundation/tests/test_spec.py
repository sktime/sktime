"""Tests for ``FoundationModelSpec``."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from dataclasses import FrozenInstanceError

import pytest

from sktime.forecasting.foundation import FoundationModelSpec


def test_extra_kwargs_are_copied_from_caller():
    """Nested extra dictionaries are isolated from subsequent caller mutation."""
    load_kwargs = {"options": {"layers": [1, 2]}}
    predict_kwargs = {"sampling": {"count": 10}}

    spec = FoundationModelSpec(
        load_extra_kwargs=load_kwargs,
        predict_extra_kwargs=predict_kwargs,
    )
    load_kwargs["options"]["layers"].append(3)
    predict_kwargs["sampling"]["count"] = 20

    assert spec.load_extra_kwargs == {"options": {"layers": [1, 2]}}
    assert spec.predict_extra_kwargs == {"sampling": {"count": 10}}


@pytest.mark.parametrize("extra_field", ["load_extra_kwargs", "predict_extra_kwargs"])
def test_extra_kwargs_reject_standard_fields(extra_field):
    """Standard settings cannot be duplicated in model-specific extras."""
    with pytest.raises(ValueError, match="use the explicit FoundationModelSpec field"):
        FoundationModelSpec(**{extra_field: {"device": "cpu"}})


def test_spec_is_frozen():
    """Top-level runtime settings cannot be reassigned after construction."""
    spec = FoundationModelSpec(model_path="dummy")

    with pytest.raises(FrozenInstanceError):
        spec.model_path = "other"
