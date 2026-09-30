"""Tests for tag register and tag functionality."""

import re

import pytest

from sktime.registry._tags import ESTIMATOR_TAG_REGISTER, check_tag_is_valid


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


def _example_value(tag_type):
    """Return a valid example value for a tag type spec, as used in the register.

    Parameters
    ----------
    tag_type : str, or tuple of (str, list of str), or tuple of (str, str)
        tag type specification, 2nd entry of an ``ESTIMATOR_TAG_REGISTER`` tuple

    Returns
    -------
    example value that is valid for ``tag_type``
    """
    if isinstance(tag_type, tuple):
        kind, allowed = tag_type
        if kind == "str":
            return allowed[0]
        if kind == "list" and allowed == "str":
            return ["foo"]
        if kind == "list":
            return [allowed[0]]
    return {
        "bool": True,
        "int": 1,
        "str": "foo",
        "list": [],
        "type": int,
    }[tag_type]


def _first_tag_of_type(predicate):
    """Return name and type of first registered tag whose type satisfies predicate."""
    for tag in ESTIMATOR_TAG_REGISTER:
        if predicate(tag[2]):
            return tag[0], tag[2]
    pytest.skip("no tag of the required type present in the register")


@pytest.mark.parametrize("tag", ESTIMATOR_TAG_REGISTER, ids=lambda tag: tag[0])
def test_check_tag_is_valid_accepts_valid_value(tag):
    """Check that a valid value is accepted, for every tag in the register.

    Regression test: ``check_tag_is_valid`` looked the tag type up by the
    literal string ``"tag_name"`` rather than by the ``tag_name`` argument,
    so the lookup matched no row and every call raised
    ``ValueError: The truth value of a Series is ambiguous``.
    """
    tag_name, tag_type = tag[0], tag[2]
    check_tag_is_valid(tag_name, _example_value(tag_type))


def test_check_tag_is_valid_unknown_tag():
    """Check that an unknown tag name raises KeyError."""
    with pytest.raises(KeyError):
        check_tag_is_valid("this_is_not_a_valid_tag", True)


def test_check_tag_is_valid_bool():
    """Check that a bool tag rejects non-bool values."""
    tag_name, _ = _first_tag_of_type(lambda t: t == "bool")
    check_tag_is_valid(tag_name, True)
    check_tag_is_valid(tag_name, False)
    with pytest.raises(ValueError, match=re.escape(tag_name)):
        check_tag_is_valid(tag_name, "True")


def test_check_tag_is_valid_str_from_list():
    """Check that a ("str", list) tag accepts listed values and rejects others."""
    tag_name, tag_type = _first_tag_of_type(
        lambda t: isinstance(t, tuple) and t[0] == "str"
    )
    check_tag_is_valid(tag_name, tag_type[1][0])
    with pytest.raises(ValueError, match=re.escape(tag_name)):
        check_tag_is_valid(tag_name, "definitely_not_an_allowed_value")


def test_check_tag_is_valid_list_of_str():
    """Check that a ("list", "str") tag accepts str or list of str.

    This case must be handled before the subset check for ("list", list) tags,
    since ``set(tag_value).issubset("str")`` is not a meaningful test.
    """
    tag_name, _ = _first_tag_of_type(lambda t: isinstance(t, tuple) and t[1] == "str")
    check_tag_is_valid(tag_name, "foo")
    check_tag_is_valid(tag_name, ["foo", "bar"])
    with pytest.raises(ValueError, match=re.escape(tag_name)):
        check_tag_is_valid(tag_name, [1, 2])


def test_check_tag_is_valid_error_is_value_error():
    """Check that an invalid non-str value raises ValueError, not TypeError.

    The error messages concatenated ``tag_value`` onto a str, which raised
    ``TypeError`` instead of the documented ``ValueError`` whenever the
    offending value was not itself a str.
    """
    tag_name, _ = _first_tag_of_type(lambda t: t == "bool")
    with pytest.raises(ValueError, match=re.escape(tag_name)):
        check_tag_is_valid(tag_name, 3)


def test_check_tag_is_valid_subset_unhashable():
    """Check that unhashable entries raise ValueError, not TypeError.

    The subset check converts ``tag_value`` to a ``set``, which raises
    ``TypeError`` for unhashable entries. The documented contract is
    ``ValueError``.
    """
    tag_name, tag_type = _first_tag_of_type(
        lambda t: isinstance(t, tuple) and t[0] == "list" and t[1] != "str"
    )
    check_tag_is_valid(tag_name, [tag_type[1][0]])
    with pytest.raises(ValueError, match=re.escape(tag_name)):
        check_tag_is_valid(tag_name, [{"unhashable": 1}])


@pytest.mark.parametrize(
    "plain_type, invalid_value",
    [
        ("bool", "not_a_bool"),
        ("int", "not_an_int"),
        ("str", 42),
        ("list", "not_a_list"),
    ],
)
def test_check_tag_is_valid_plain_types_reject(plain_type, invalid_value):
    """Check that the non-tuple tag types reject values of the wrong type."""
    tag_name, _ = _first_tag_of_type(lambda t: t == plain_type)
    with pytest.raises(ValueError, match=re.escape(tag_name)):
        check_tag_is_valid(tag_name, invalid_value)
