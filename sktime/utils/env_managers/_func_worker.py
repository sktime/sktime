"""Subprocess worker that executes a serialized callable."""

from __future__ import annotations

import sys


def main():
    """Read a cloudpickled callable payload from stdin and write the result."""
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
