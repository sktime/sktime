"""Tests for tag register and tag functionality."""

from sktime.registry._tags import ESTIMATOR_TAG_REGISTER


def test_tag_register_type():
    """Test the specification of the tag register. See _tags for specs."""
    assert isinstance(ESTIMATOR_TAG_REGISTER, list)
    assert all(isinstance(tag, tuple) for tag in ESTIMATOR_TAG_REGISTER)

    for tag in ESTIMATOR_TAG_REGISTER:
        assert len(tag) == 4
        assert isinstance(tag[0], str)
        assert isinstance(tag[1], (str, list))
        if isinstance(tag[1], list):
            assert all(isinstance(x, str) for x in tag[1])
        assert isinstance(tag[2], (str, tuple))
        if isinstance(tag[2], tuple):
            assert len(tag[2]) == 2
            assert isinstance(tag[2][0], str)
            assert isinstance(tag[2][1], (list, str))
            if isinstance(tag[2][1], list):
                assert all(isinstance(x, str) for x in tag[2][1])
        assert isinstance(tag[3], str)
