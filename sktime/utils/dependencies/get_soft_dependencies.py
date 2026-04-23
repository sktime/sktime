"""Util to collect and install soft dependencies from estimators."""

import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable

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


def _parse_package_spec(spec: str):
    """Split a package specification into name and version specifier.

    Parameters
    ----------
    spec : str
        Package specification string, e.g., ``"numpy>=1.20"`` or ``"pandas"``.

    Returns
    -------
    tuple of str
        A tuple ``(name, version_specifier)`` where the version specifier
        may be an empty string if not provided.
    """
    match = re.match(r"^([a-zA-Z0-9_\-]+)(.*)$", spec)
    if match:
        return match.group(1), match.group(2)
    return spec, ""


def _merge_dependencies(packages):
    """Merge multiple specifications of the same package.

    For duplicate packages, a heuristic is used to keep the most restrictive
    version constraint by selecting the longest specifier string.

    Parameters
    ----------
    packages : iterable of str
        Collection of package specification strings.

    Returns
    -------
    list of str
        Sorted list of merged package specifications.
    """
    merged = defaultdict(list)

    for pkg in packages:
        name, spec = _parse_package_spec(pkg)
        merged[name].append(spec)

    final = []

    for name, specs in merged.items():
        # pick the "most restrictive" spec (heuristic: longest string)
        best_spec = max(specs, key=len) if specs else ""
        final.append(name + best_spec)

    return sorted(final)


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
