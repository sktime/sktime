# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Hyper-Trees forecasters' frequency alias handling."""

__author__ = ["webzuweb"]

import warnings

import pandas as pd
import pytest

from sktime.forecasting.hypertrees import _period_to_offset_alias


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("M", "MS"),
        ("Q", "QS"),
        ("Y", "YS"),
        ("A", "YS"),
        # compound aliases keep their anchor, only the base alias is mapped
        ("Q-DEC", "QS-DEC"),
        ("A-JAN", "YS-JAN"),
        # already-offset aliases and non-period aliases pass through unchanged
        ("MS", "MS"),
        ("QS", "QS"),
        ("YS", "YS"),
        ("D", "D"),
        ("W-SUN", "W-SUN"),
    ],
)
def test_period_to_offset_alias(freq, expected):
    """``_period_to_offset_alias`` maps period aliases to offset aliases."""
    assert _period_to_offset_alias(freq) == expected


@pytest.mark.parametrize("freq", ["M", "Q", "Y", "A", "Q-DEC", "A-JAN"])
def test_period_alias_mapping_is_date_range_safe(freq):
    """Mapped aliases are accepted by ``pd.date_range`` without a FutureWarning.

    The raw period aliases (``"M"``, ``"Q"``, ``"Y"``, ``"A"``) are deprecated as
    ``pd.date_range`` frequencies on pandas >= 2.2 and raise a ``FutureWarning``.
    The mapped offset aliases must not.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = pd.date_range(
            "2000-01-01", periods=3, freq=_period_to_offset_alias(freq)
        )
    assert len(result) == 3
