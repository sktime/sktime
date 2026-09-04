import pytest

from sktime.transformations.augmenter import RandomSamplesAugmenter


@pytest.mark.parametrize(
    "invalid_n, expected_type_regex",
    [
        ("spam", r"<class 'str'>"),
        (None, r"<class 'NoneType'>"),
        ([], r"<class 'list'>"),
    ],
)
def test_random_samples_augmenter_invalid_n_raises_value_error(
    invalid_n, expected_type_regex
):
    """RandomSamplesAugmenter should raise ValueError
    for non-int/float n with correct message."""
    with pytest.raises(
        ValueError, match=rf"n must be int or float, not {expected_type_regex}\."
    ):
        RandomSamplesAugmenter(n=invalid_n)
