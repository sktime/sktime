# Deprecation recipes

Replace `1.4.0` with the version from step 2 of the skill. The sentinels `"changing_value"`,
`"deprecated"` and `"position_change"` are literal strings, used so that "the user passed
nothing" is distinguishable from any real value.

## Contents

- [Changing a default](#changing-a-default)
- [Renaming a parameter](#renaming-a-parameter)
- [Removing a parameter](#removing-a-parameter)
- [Adding a parameter that changes behaviour](#adding-a-parameter-that-changes-behaviour)
- [Changing parameter order](#changing-parameter-order)
- [Moving a class or function](#moving-a-class-or-function)
- [Removing a class or function](#removing-a-class-or-function)
- [Tags](#tags)
- [Tests](#tests)

## Changing a default

Copy "Step 1: during deprecation period" under "Changing the default value of a parameter",
in the "Examples to illustrate recipes" section of
`docs/source/developer_guide/deprecation.rst`. Then make the changes in "Things that often
go wrong" in the skill:

- call `warn` without `obj=self`
- write the markers as lowercase `# todo 1.4.0: ...`
- for an estimator, call `super().__init__()` after storing the parameters, before the
  warning block

Use `self._parameter` in all other methods.

## Renaming a parameter

Copy the guide's "Renaming a parameter" example in the same section, with the same changes.
Put the new parameter at the end, so positional calls keep working. The old one keeps its
position and gets the default `"deprecated"`. Both are written to `self` during the change
period, and the docstring describes only the new name.

## Removing a parameter

If it isn't last in the signature, move it to the end first, which is itself an order change.
Then do as for a default change, using `"deprecated"` as the default, and say whether
prior behaviour can be retained, and how. When several classes share a parameter, use one
helper, as `_check_n_features_per_kernel` in
`sktime/transformations/rocket/_multirocket.py` does:

```python
# todo 1.4.0: remove this function and its calls in MultiRocket and RocketClassifier,
# and remove the parameter from their signatures and docstrings
def _check_old_param(obj):
    """Warn if the deprecated old_param of obj is set."""
    if obj.old_param == "deprecated":
        return

    warn(
        f"Parameter old_param of {type(obj).__name__} is deprecated and will be removed "
        "in sktime 1.4.0. The value passed is ignored. No code change is needed to keep "
        "current results: remove the parameter from the call.",
        DeprecationWarning,
    )
```

## Adding a parameter that changes behaviour

Two steps, which can be in one PR:

1. Add it with a default that keeps prior behaviour. That alone needs no deprecation.
2. Follow [Changing a default](#changing-a-default) for the new value.

## Changing parameter order

Avoid this if any other recipe works. If it's needed, see "Changing the sequence of
parameters" in the dev guide. Parameters from the first moved position onward get the
default `"position_change"`, and the warning tells users to pass them by keyword. The todo
has to contain the complete final signature.

## Moving a class or function

Keep a forwarding definition at the old location, as `sktime/forecasting/model_selection/__init__.py`
does for splitters:

```python
# todo 1.4.0: remove this legacy export
def OldName(a=1, b=2):
    """Legacy export of OldName.

    DEPRECATED - use sktime.newplace.NewName instead.
    """
    from sktime.newplace import NewName

    warn(
        "The location of OldName in sktime.oldplace is deprecated and will be removed in "
        "sktime 1.4.0. Update the import to: from sktime.newplace import NewName.",
        DeprecationWarning,
    )

    return NewName(a=a, b=b)
```

Keep the import of the new location inside the function if the two modules could form an
import cycle. Update the API reference to the new location.

## Removing a class or function

Warn on use: in `__init__` or the function body, or in `__post_init__`. Name the replacement
if there is one, and if not, say so. Add the todo above the class.

## Tags

Use `TagAliaserMixin` from `sktime.base`, whose docstring in `sktime/base/_base.py` explains
it. Add it first in the bases of the youngest class that covers the tag, then:

- `alias_dict`: `{"old_tag": "new_tag"}` to rename, `{"old_tag": ""}` to remove
- `deprecate_dict`: `{"old_tag": "1.4.0"}`, with the same keys as `alias_dict`

When the period ends, remove the entries, and remove the mixin if none are left.

## Tests

Mark them for removal too:

```python
# todo 1.4.0: remove this test together with old_parameter
def test_old_parameter_warns():
    """Using old_parameter raises a DeprecationWarning and still works."""
    with pytest.warns(DeprecationWarning, match="old_parameter"):
        est = EstimatorName(old_parameter="x")

    assert est.get_params()["old_parameter"] == "x"


def test_new_parameter_does_not_warn():
    """Using the new parameter raises no DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        EstimatorName(new_parameter="x")
```

Also check that the old and new spelling give the same result where behaviour should be
retained.
