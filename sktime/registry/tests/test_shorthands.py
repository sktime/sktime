"""Tests for shorthand functionality."""

from sktime.registry._namespace import _namespace_shorthands


def test_shorthands_unique():
    """Test that all shorthands in the namespace are unique."""
    # this call automatically raises if the shorthands do not have unique names
    _namespace_shorthands(raise_if_not_unique=True)


def test_shorthands_craftable():
    """Test that all shorthands in the namespace are safe-craftable."""
    from sktime.registry import craft

    shorthands = _namespace_shorthands()
    for sh, spec in shorthands.items():
        if not isinstance(spec, str):
            raise ValueError(
                f"Shorthand spec for {sh} is not a string. Found type: {type(spec)}. "
                f"Found value: {spec}"
            )
        # this would raise an exception if there is unsafe code in the shorthand
        try:
            craft(spec, safe=True)
            craft(sh, safe=True)
        except Exception as e:
            raise AssertionError(
                f"Shorthand {sh} is not safe-craftable. Shorthand spec: {spec}"
            ) from e


def test_shorthand_simple_craft():
    """Test that a simple shorthand can be crafted safely."""
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.registry import craft

    naive_spec = "f_naive_last"
    naive_inst = craft(naive_spec, safe=True)
    assert isinstance(naive_inst, NaiveForecaster)
    assert naive_inst == NaiveForecaster(strategy="last")


def test_shorthands_pipe_craft():
    """Test that a pipeline involving a shorthand can be crafted safely."""
    from sktime.forecasting.compose import TransformedTargetForecaster as T
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.registry import craft
    from sktime.transformations.series.detrend import Deseasonalizer

    pipe_spec = "Deseasonalizer(sp=12) * f_naive_last"
    pipe_inst = craft(pipe_spec, safe=True)

    assert isinstance(pipe_inst, T)
    steps = pipe_inst.get_params().get("steps", None)
    assert isinstance(steps, list)
    assert steps[0] == Deseasonalizer(sp=12)
    assert steps[1] == NaiveForecaster(strategy="last")
