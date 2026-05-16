"""Run each import statement in all_import_statements.txt and report failures.

Each line is exec'd with a fresh globals dict in this same interpreter.
Lines that touch one of the flattened legacy folders (``panel``, ``series``)
must additionally raise a DeprecationWarning; the legacy module cache is
evicted before each such line so the shim ``__init__.py`` body re-runs.
"""

import sys
import traceback
import warnings
from pathlib import Path


FILE = Path(__file__).parent / "all_import_statements.txt"

# Subpackages flattened in the migration. Imports under these paths are
# served by deprecation-warning shims; any other path is non-deprecated.
LEGACY_FOLDERS = ("panel", "series")
LEGACY_PREFIXES = tuple(f"sktime.transformations.{f}" for f in LEGACY_FOLDERS)


def is_legacy(code):
    return any(p in code for p in LEGACY_PREFIXES)


def clear_legacy_cache():
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in LEGACY_PREFIXES):
            del sys.modules[name]


def main():
    lines = [
        (i, raw)
        for i, raw in enumerate(FILE.read_text().splitlines(), start=1)
        if raw.strip() and not raw.strip().startswith("#")
    ]
    print(f"running {len(lines)} import statements from {FILE.name}")

    failures = []
    for lineno, code in lines:
        legacy = is_legacy(code)
        if legacy:
            clear_legacy_cache()

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                exec(compile(code, f"<line {lineno}>", "exec"), {})
        except BaseException:
            failures.append((lineno, code, traceback.format_exc()))
            print(f"  [{lineno:>4}] FAIL (import): {code[:100]}")
            continue

        if legacy:
            dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
            if not dep:
                failures.append((lineno, code, "expected DeprecationWarning, none raised\n"))
                print(f"  [{lineno:>4}] FAIL (no warn): {code[:100]}")
                continue
            print(f"  [{lineno:>4}] ok (warned)")
        else:
            print(f"  [{lineno:>4}] ok")

    print()
    if failures:
        print(f"FAILED: {len(failures)} / {len(lines)}")
        for lineno, code, err in failures:
            print(f"\n--- line {lineno} ---")
            print(code)
            print(err.rstrip())
        sys.exit(1)
    print(f"OK: all {len(lines)} import statements succeeded")


if __name__ == "__main__":
    main()
