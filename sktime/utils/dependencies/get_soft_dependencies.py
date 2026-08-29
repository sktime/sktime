"""Util to collect and install soft dependencies from estimators."""

import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

from sktime.registry import all_estimators

__all__ = ["get_soft_dependencies"]


def _normalize_dependencies(deps: str | Iterable | None) -> set[str]:
    """Return a normalized set of dependency strings.

    Parameters
    ----------
    deps : str or iterable of str or iterable of iterable of str or None
        Dependency specification. Supported formats include a single string,
        a flat iterable of strings, nested iterables of strings, or None.

    Returns
    -------
    set of str
        Flattened set of dependency strings. Invalid or non-string entries
        are ignored.
    """
    normalized: set[str] = set()

    if deps is None:
        return normalized

    if isinstance(deps, str):
        normalized.add(deps)
        return normalized

    if isinstance(deps, (list, tuple, set)):
        for item in deps:
            if isinstance(item, str):
                normalized.add(item)
            elif isinstance(item, (list, tuple, set)):
                for sub in item:
                    if isinstance(sub, str):
                        normalized.add(sub)
            # silently ignore anything else

    return normalized


def get_soft_dependencies(
    install: bool = False,
    verbose: bool = False,
) -> set[str]:
    """Collect soft dependencies declared by all sktime estimators.

    Optionally installs the collected dependencies using pip.

    Parameters
    ----------
    install : bool, default=False
        If True, install the collected dependencies.
    verbose : bool, default=False
        If True, print progress information and warnings.

    Returns
    -------
    set of str
        Unique dependency specifications collected from estimator tags.
    """
    dependencies: set[str] = set()

    estimators = all_estimators()

    for name, Estimator in estimators:
        try:
            deps = Estimator.get_class_tags().get("python_dependencies", None)
            normalized = _normalize_dependencies(deps)
            dependencies.update(normalized)

        except Exception as e:
            if verbose:
                print(f"[WARN] Skipping {name}: {e}")

    if verbose:
        print(f"\nCollected {len(dependencies)} unique dependencies.")

    if install and dependencies:
        _install_packages(dependencies, verbose=verbose)

    return dependencies


def _merge_dependencies(packages, verbose=True, strategy="strict"):
    """Merge multiple specifications of the same package.

    This implementation uses proper parsing via ``packaging`` and attempts
    to compute the intersection of version constraints.

    Parameters
    ----------
    packages : iterable of str
        Collection of package specification strings.
    verbose : bool, default=True
        If True, print warnings for conflicts.
    strategy : {"strict", "permissive"}, default="strict"
        Strategy to handle conflicting constraints:
        - "strict": skip packages with incompatible constraints
        - "permissive": fall back to a heuristic (longest specifier)

    Returns
    -------
    list of str
        Sorted list of merged package specifications.
    """
    grouped = defaultdict(list)

    # group by package name
    for pkg in packages:
        try:
            req = Requirement(pkg)
            grouped[req.name].append(req.specifier)
        except Exception:
            # fallback for non-standard strings
            grouped[pkg].append(SpecifierSet(""))

    merged = []

    for name, specifiers in grouped.items():
        # compute intersection
        combined = SpecifierSet()
        for spec in specifiers:
            combined &= spec

        combined_str = str(combined)

        # detect potential conflict (empty intersection heuristic)
        if not combined_str:
            if len(specifiers) > 1:
                if verbose:
                    print(
                        f"[WARN] Conflicting requirements for {name}: "
                        f"{[str(s) for s in specifiers]}"
                    )

                if strategy == "strict":
                    # skip conflicting package
                    continue

                elif strategy == "permissive":
                    # fallback: pick longest original spec
                    fallback = max(
                        (str(s) for s in specifiers),
                        key=len,
                        default="",
                    )
                    merged.append(name + fallback)
                    continue

        merged.append(name + combined_str)

    return sorted(merged)


def _install_packages(packages, verbose=True):
    """Install packages using pip.

    Parameters
    ----------
    packages : iterable of str
        Package specification strings to install.
    verbose : bool, default=True
        If True, print installation progress and errors.

    Returns
    -------
    None
        This function installs packages as a side effect.
    """
    merged_packages = _merge_dependencies(packages)

    if verbose:
        print(f"\nInstalling {len(merged_packages)} merged packages...\n")

    for pkg in merged_packages:
        try:
            if verbose:
                print(f"[INSTALL] {pkg}")

            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])  # noqa: S603

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install {pkg}: {e}")
