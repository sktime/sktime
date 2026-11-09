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

```python
from sktime.utils.warnings import warn


# todo 1.4.0: change the default of parameter to <new_default>, update the docstring,
# and remove the warning in __init__
class EstimatorName(BaseForecaster):
    """...

    Parameters
    ----------
    parameter : str, default=<old_default>
        The parameter description.
        The default will change to <new_default> in version 1.4.0.
    """

    def __init__(self, parameter="changing_value"):
        self.parameter = parameter  # stored unchanged, for get_params and clone

        super().__init__()

        # todo 1.4.0: replace this block with self._parameter = parameter
        if parameter == "changing_value":
            warn(
                "In EstimatorName, the default of parameter will change from "
                "<old_default> to <new_default> in sktime 1.4.0. To keep the current "
                "behaviour and silence this warning, set parameter=<old_default> "
                "explicitly.",
                DeprecationWarning,
            )
            self._parameter = <old_default>
        else:
            self._parameter = parameter
```

Use `self._parameter` in all other methods.

## Renaming a parameter

Put the new parameter at the end, so positional calls keep working. The old one keeps its
position and gets the default `"deprecated"`. The docstring describes only the new name.

```python
# todo 1.4.0: remove old_parameter from the signature, move new_parameter to its position,
# and remove the warning in __init__
class EstimatorName(BaseForecaster):
    """...

    Parameters
    ----------
    new_parameter : str, default=<default>
        The parameter description.
    """

    def __init__(self, old_parameter="deprecated", new_parameter=<default>):
        # both are written to self during the change period
        self.old_parameter = old_parameter
        self.new_parameter = new_parameter

        super().__init__()

        # todo 1.4.0: replace this block with self._new_parameter = new_parameter
        if old_parameter != "deprecated":
            warn(
                "In EstimatorName, parameter old_parameter was renamed to new_parameter, "
                "and will be removed in sktime 1.4.0. To keep the current behaviour and "
                "silence this warning, use new_parameter instead of old_parameter.",
                DeprecationWarning,
            )
            self._new_parameter = old_parameter
        else:
            self._new_parameter = new_parameter
```

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
