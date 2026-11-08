"""Subprocess worker that executes a serialized callable."""

import sys


def main():
    """Execute a serialized callable and write its return value to stdout.

    Reads a cloudpickled payload from stdin with the following keys:

    * ``func`` — callable to invoke
    * ``args`` — positional arguments, default ``()``
    * ``kwargs`` — keyword arguments, default ``{}``
    * ``input`` — optional first positional argument, used only when the
      key is present, kept to unify signature of `run` for scripts, modules,
      and callables.

    The callable is invoked as ``func(input, *args, **kwargs)`` when
    ``input`` is in the payload, otherwise as ``func(*args, **kwargs)``.
    The return value is cloudpickled to stdout.

    Raises
    ------
    KeyError
        If the payload does not contain ``func``.
    """
    import cloudpickle

    payload = cloudpickle.load(sys.stdin.buffer)
    func = payload["func"]
    args = tuple(payload.get("args") or ())
    kwargs = dict(payload.get("kwargs") or {})

    if "input" in payload:
        result = func(payload["input"], *args, **kwargs)
    else:
        result = func(*args, **kwargs)

    cloudpickle.dump(result, sys.stdout.buffer)


if __name__ == "__main__":
    main()
