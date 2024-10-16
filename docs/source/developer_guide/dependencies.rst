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

Handling soft dependencies
--------------------------

This section explains how to handle existing soft dependencies.
For adding a new soft dependency, see the section "adding a new soft dependency".

Soft dependencies in ``sktime`` should usually be isolated to estimators.

Informative warnings or error messages for missing soft dependencies should be raised, in a situation where a user would need them.
This is handled through our ``_check_soft_dependencies`` utility
`here <https://github.com/sktime/sktime/blob/main/sktime/utils/dependencies/_dependencies.py>`__.
There are specific conventions to add such warnings in estimators, as below.

Estimators with a soft dependency need to ensure the following:

*  imports of the soft dependency only happen inside the estimator,
   e.g., in ``_fit`` or ``__init__`` methods of the estimator.
   In ``__init__``, imports should happen only after calls to ``super(cls).__init__``.
*  the packaging tags of the estimator are populated, i.e., ``python_dependencies``
   with PEP 440 compliant dependency specifier strings such as ``pandas>=2.0.1``, and optionally
   ``python_version`` and ``env_marker`` if specific markers are needed.
   Exceptions will automatically be raised when constructing the estimator
   in an environment where the requirements are not met.
   For further details, see the tag API reference, :ref:`packaging_tags`.
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

Adding and maintaining soft dependencies
----------------------------------------

When adding a new soft dependency or changing the version of an existing one,
the following need to be updated:

*  in `pyproject.toml <https://github.com/sktime/sktime/blob/main/pyproject.toml>`__,
   add the dependency or update version bounds in the ``all_extras`` dependency set.
   Following the `PEP 621 <https://www.python.org/dev/peps/pep-0621/>`_ convention, all dependencies
   including build time dependencies and optional dependencies are specified in ``pyproject.toml``.
*  Soft dependencies compatible with ``pandas 2`` should also be added/updated in the
   ``all_extras_pandas2`` dependency set in ``pyproject.toml``. This dependency set
   is used only in testing.

It should be checked that new soft dependencies do not imply
upper bounds on ``sktime`` core dependencies, or severe limitations to the user
installation workflow.
In such a case, it is strongly suggested not to add the soft dependency.

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
