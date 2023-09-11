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

This section explains how to handle existing soft depencies.
For adding a new soft dependency, see the section "adding a new soft dependency".

Soft dependencies in ``sktime`` should usually be isolated to estimators.

Informative warnings or error messages for missing soft dependencies should be raised, in a situation where a user would need them.
This is handled through our ``_check_soft_dependencies`` utility
`here <https://github.com/sktime/sktime/blob/main/sktime/utils/validation/_dependencies.py>`__.
There are specific conventions to add such warnings in estimators, as below.

Estimators with a soft dependency need to ensure the following:

*  imports of the soft dependency only happen inside the estimator,
   e.g., in ``_fit`` or ``__init__`` methods of the estimator.
   In ``__init__``, imports should happen only after calls to ``super(cls).__init__``.
*  the ``python_dependencies`` tag of the estimator is populated with a ``str``,
   or a ``list`` of ``str``, of dependency requirements, where ``str`` are PEP 440 compliant version specification ``str``
   such as ``pandas>=2.0.1``. Exceptions will automatically be raised when constructing the estimator
   in an environment where the requirements are not met.
*  In a case where the package import differs from the package name, i.e., ``import package_string`` is different from
   ``pip install different-package-string`` (usually the case for packages containing a dash in the name), the ``python_dependencies_alias`` tag
   should be populated to pass the information on package and import strings as ``dict`` such as ``{"scikit-learn": "sklearn"}``.
*  If the soft dependencies require specific python versions, the ``python_version``
   tag should also be populated, with a PEP 440 compliant version specification ``str`` such as ``"<3.10"`` or ``">3.6,~=3.8"``.
*  If including docstring examples that use soft dependencies, ensure to skip the corresponding doctest,
   in order to avoid that ``doctest`` attempts to import the soft dependency when it is not present.
   To do this, add a ``# doctest: +SKIP`` to the end of each line in the doctest to skip it entirely.
   See ``forecasting.arima.ARIMA`` as as an example. If concerned that skipping the test will reduce test coverage,
   consider exposing the doctest example as a pytest test function instead, see below how to handle soft dependencies in pytest functions.
*  Decorate all ``pytest`` tests that import soft dependencies with a ``@pytest.mark.skipif(...)`` conditional on a check to ``_check_soft_dependencies``
   for your new soft depenency, with ``severity="none"``.  Be sure that all soft dependencies imported for testing are imported within the test function itself,
   rather than for the whole module!  This decorator will then skip your test, including imports, unless the system has the required packages installed.
   This prevents crashes for any users running ``check_estimator`` on all estimators, or a full local ``pytest`` run without the required soft dependency.
   See the tests in ``forecasting.tests.test_pmdarima`` for a concrete example.

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
semantic versioning. For stable packages, next ``major`` verion can be used as well.

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
