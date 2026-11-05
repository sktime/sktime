.. _dependencies:

Dependencies
============

.. contents::
   :local:

Types of dependencies
---------------------

There are three types of dependencies in ``sktime``: **core**, **soft**, or **developer**.

.. note::

   * **Core** dependencies are required to install and run ``sktime`` and are automatically installed with ``sktime``, *e.g.*  ``pandas``;
   * **Soft** dependencies are only required to import certain modules, but not necessary to use most functionalities. A soft dependency is not installed automatically with the package. Instead, users need to install it manually if they want to use a module that requires a soft dependency, *e.g.* ``pmdarima``;
   * **Developer** dependencies are required for ``sktime`` developers, but not for typical users of ``sktime``, *e.g.* ``pytest``.


We try to keep the number of core dependencies to a minimum and rely on other packages as soft dependencies when feasible.

This chapter covers two related topics: **maintaining** optional dependencies for the ``sktime`` package (extras, ``pyproject.toml``),
and **using** them in code (estimators, modules, tests). The sections below move from estimator-level isolation to module-level patterns,
then to packaging maintenance.

Handling soft dependencies
--------------------------

This section explains how to handle existing soft dependencies in code.
For adding a new soft dependency to the distribution, see the section "Adding and maintaining soft dependencies".

**Best practice:**

(a) Soft dependencies should be restricted to estimators whenever possible, see the section "Isolating soft dependencies to estimators".

(b) If restricting to estimators is not possible, follow the section "Isolating soft dependencies at module level".

Isolating soft dependencies to estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Soft dependencies in ``sktime`` should usually be isolated to estimators.

This means, importing only in methods of the estimator, such as ``_fit``, ``_predict``, or ``__init__``, and not at the module level.
This ensures that the soft dependency is only loaded when the estimator is used, and does not affect ``sktime`` as a whole.

Estimators with a soft dependency need to ensure the following:

*  imports of the soft dependency only happen inside the estimator,
   e.g., in ``_fit`` or ``__init__`` methods of the estimator.
   During constructor calls, imports should happen only in ``__post_init__``, never in ``__init__``.
*  the packaging tags of the estimator are populated, i.e., ``python_dependencies``
   with PEP 440 compliant dependency specifier strings such as ``pandas>=2.0.1``, and optionally
   ``python_version`` and ``env_marker`` if specific markers are needed.
   Exceptions will automatically be raised when constructing the estimator
   in an environment where the requirements are not met.
   For further details, see the tag API reference, :ref:`packaging_tags`.
*  the tag ``tests:vm`` of the estimator is set to ``True``. This automatically ensures
   that the estimator is tested in a virtual machine with all dependencies installed.
   If the tag is not set, the estimator will automatically fail compliance.
*  Decorate all ``pytest`` tests that import soft dependencies with a ``@pytest.mark.skipif(...)`` conditional on a soft dependency check.
   If the test is specific to a single estimator or object, use ``run_test_for_class`` from ``sktime.tests.test_switch``
   to mediate the condition through the class tags.
   Otherwise, use ``_check_soft_dependencies`` for your new soft dependency, with ``severity="none"``.
   Be sure that all soft dependencies imported for testing are imported within the test function itself,
   rather than at root level (at the top) of the module.
   This decorator will then skip your test, including imports,
   unless the system has the required packages installed.
   This prevents crashes for any users running ``check_estimator`` on all estimators,
   or a full local ``pytest`` run without the required soft dependency.
   See the tests in ``forecasting.tests.test_pmdarima`` for a concrete example of
   ``run_test_for_class`` usage to decorate a test. See ``utils.tests.test_plotting``
   for an example of ``_check_soft_dependencies`` usage.

For estimators, no additional warnings or error messages should be raised for missing soft dependencies,
as the base framework will automatically raise default exceptions, based on populated tags (see above),
when the estimator is used in an environment where the requirements are not met.
Adding additional warnings or error messages is not necessary, and will usually cause test failures.

To manage soft dependencies that are contingent on specific parameter settings of the estimator,
use the ``__dynamic_tags__`` dunder of the estimator, as outlined in the extension templates.

In case a step-out is needed from the above rules, it should be clearly justified
in the pull request description. In such a case, the ``_check_soft_dependencies`` utility
`here <https://github.com/sktime/sktime/blob/main/sktime/utils/dependencies/_dependencies.py>`__ can be used.

Isolating soft dependencies at module level
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In certain scenarios, it is hard to avoid soft dependency import at the module level, for example:

* class inheritance, where the base class is defined in a different package, e.g., ``torch.nn.Module`` in ``sktime`` deep learning estimators;
* module-level decorators, where the decorator is defined in a different package, e.g., ``numba.jit`` in ``sktime`` estimators that use JIT compilation;

Where such scenarios can be avoided, they should be avoided, and soft dependencies should be isolated to estimators as described above.

However, if a soft dependency must be imported at the module level,
the ``_safe_import`` utility should be used (see below).

The ``_safe_import`` utility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it does.** ``_safe_import`` attempts to import a module or object by dotted path.
If the dependency is installed and import succeeds, the real object is returned.
If the dependency is missing or import fails, a stand-in is returned so that **importing**
``sktime`` still succeeds. Runtime use of that dependency still requires the package to be
installed; keep estimator tags and ``_check_soft_dependencies`` checks as described above.

**Where it lives.** The implementation is maintained in ``skbase``; ``sktime`` re-exports it for developers as
``from sktime.utils.dependencies import _safe_import``. The upstream source is
`skbase/utils/dependencies/_import.py <https://github.com/sktime/skbase/blob/main/skbase/utils/dependencies/_import.py>`__.

**Basic usage.** Assign the return value at module level using a dotted import path:

.. code-block:: python

    from sktime.utils.dependencies import _safe_import

    nn = _safe_import("torch.nn")
    Linear = _safe_import("torch.nn.Linear")

**How paths are resolved.** The string is split on ``.`` and interpreted as follows:

* No dots (e.g. ``"torch"``): import the top-level module.
* One or more dots: import the parent module with ``importlib``, then ``getattr`` for the final name
  (e.g. ``"torch.nn"`` → submodule ``nn``; ``"torch.nn.Linear"`` → class ``Linear``).

**Optional arguments.**

* ``pkg_name``: PyPI / distribution name when it differs from the first segment of ``import_path``, e.g.
  ``_safe_import("sklearn.clone", pkg_name="scikit-learn")``. If omitted, the first segment of ``import_path`` is used to
  detect whether the package is installed.
* ``condition``: if ``False``, the import is skipped and the fallback is returned (same as a failed import).
* ``return_object``: ``"MagicMock"`` (default) or ``"None"`` for the fallback when the package is missing or the import fails.

**Fallback behaviour.** When the dependency is not available, ``_safe_import`` returns a unique mock type per import path
that accepts arbitrary attribute access, calls, and subclassing, so patterns such as ``class MyNet(nn.Module): ...`` parse without
raising ``ImportError``. Code that actually runs model logic will still need the real package installed.

**Limitations and caveats.**

* Do not use ``_safe_import`` return values with ``@dataclass`` or as bases for dataclasses; that combination is unsupported.
* Prefer lazy imports inside estimator methods when there is no need for a module-level base class or decorator from the optional package.
* Mock objects are not a substitute for runtime dependency checks: keep ``python_dependencies`` tags accurate and use
  ``_check_soft_dependencies`` where users need an explicit error message.

**Example (submodule as base class):**

.. code-block:: python

    from sktime.utils.dependencies import _safe_import

    nn = _safe_import("torch.nn")


    class MyNetwork(nn.Module):
        ...

**Example (distribution name differs from import name):**

.. code-block:: python

    from sktime.utils.dependencies import _safe_import

    clone = _safe_import("sklearn.clone", pkg_name="scikit-learn")

Concluding by repeating the important note at the top:

use of ``_safe_import`` should be avoided whenever possible,
in favour of isolating soft dependencies to estimators.

Adding and maintaining soft dependencies
----------------------------------------

When adding a new soft dependency or changing the version of an existing one,
the following need to be updated:

*  in `pyproject.toml <https://github.com/sktime/sktime/blob/main/pyproject.toml>`__,
   add the dependency or update version bounds in the ``all_extras`` dependency set.
   Following the `PEP 621 <https://www.python.org/dev/peps/pep-0621/>`_ convention, all dependencies
   including build time dependencies and optional dependencies are specified in ``pyproject.toml``.
*  Important: only the most important soft dependencies should be added to the ``all_extras``
   dependency set. Soft dependencies required only be one estimator or a small number of estimators
   should not be added to ``all_extras``, to avoid dependency bloat.
   For testing purposes, the ``tests:vm``
   tag of the estimator should be set, to ensure a VM with the specific soft dependencies
   is spun up regularly.

It shhould be checked that new soft dependencies added to ``all_extras`` do not imply
upper bounds on ``sktime`` core dependencies, or severe limitations to the user
installation workflow.

For maintenance purposes, it has been decided that all soft-dependencies will have lower
and upper bounds specified mandatorily. The soft-dependencies will be specified in
separate extras per each component of ``sktime``, for example ``forecasting``,
``classification``, ``regression``, etc. It is possible to have different upper and
lower bounds for a single package when present in different extras, and can be modified in one without affecting the others.

Upper bounds will be preferred to be set up as the next ``minor`` release of the
packages, as ``patch`` updates should never contain breaking changes by convention of
semantic versioning. For stable packages, next ``major`` version can be used as well.

Upper bounds will be automatically updated using ``dependabot``, which has been set up
to run daily based on releases on ``PyPI``. The CI introducing newer upper bound will be
merged into ``main`` branch only if all unit tests for the affected component(s) pass.

Lower bounds maintenance planning is in progress and will be updated here soon.

Adding a core or developer dependency
-------------------------------------

Core or developer dependencies can be added only by core developers after discussion in the core developer meeting.

When adding a new core dependency or changing the version of an existing one,
the following files need to be updated:

*  `pyproject.toml <https://github.com/sktime/sktime/blob/main/pyproject.toml>`__,
   adding the dependency or version bounds in the ``dependencies`` dependency set.

When adding a new developer dependency or changing the version of an existing one,
the following files need to be updated:

*  `pyproject.toml <https://github.com/sktime/sktime/blob/main/pyproject.toml>`__,
   adding the dependency or version bounds in the ``dev`` dependency set.
