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


Adding a soft dependency
------------------------

Soft dependencies in ``sktime`` should usually be restricted to estimators.

When adding a new soft dependency or changing the version of an existing one, the following files need to be updated:

*  `pyproject.toml <https://github.com/sktime/sktime/blob/main/pyproject.toml>`__,
   adding the dependency or version bounds in the ``all_extras`` dependency set.
   Following the `PEP 621 <https://www.python.org/dev/peps/pep-0621/>`_ convention, all dependencies
   including build time dependencies and optional dependencies are specified in this file.

Informative warnings or error messages for missing soft dependencies should be raised, in a situation where a user would need them.
This is handled through our ``_check_soft_dependencies`` utility
`here <https://github.com/sktime/sktime/blob/main/sktime/utils/validation/_dependencies.py>`__.

There are specific conventions to add such warnings in estimators, as below.
To add an estimator with a soft dependency, ensure the following:

*  imports of the soft dependency only happen inside the estimator,
   e.g., in ``_fit`` or ``__init__`` methods of the estimator.
   In ``__init__``, imports should happen only after calls to ``super(cls).__init__``.
*  the ``python_dependencies`` tag of the estimator is populated with a ``str``,
   or a ``list`` of ``str``, of import dependencies. Exceptions will automatically raised when constructing the estimator
   in an environment without the required packages.
*  in the python module containing the estimator, the ``_check_soft_dependencies`` utility is called
   at the top of the module, with ``severity="warning"``. This will raise an informative warning message already at module import.
   See `here <https://github.com/sktime/sktime/blob/main/sktime/utils/validation/_dependencies.py>`__
*  In a case where the package import differs from the package name, i.e., ``import package_string`` is different from
   ``pip install different-package-string`` (usually the case for packages containing a dash in the name), the ``_check_soft_dependencies``
   utility should be used in ``__init__``. Both the warning and constructor call should use the ``package_import_alias`` argument for this.
*  If the soft dependencies require specific python versions, the ``python_version``
   tag should also be populated, with a PEP 440 compliant version specification ``str`` such as ``"<3.10"`` or ``">3.6,~=3.8"``.
*  If including docstring examples that use soft dependencies, ensure to skip doctest. To do this add a ``# doctest: +SKIP`` to the end of each
   line in the doctest to skip. Check out the arima estimator as as an example. If concerned that skipping the test will reduce test coverage,
   consider exposing the doctest example as a pytest test function instead, see below how to handle soft dependencies in pytest functions.".
*  Decorate all pytest tests that import soft dependencies with a ``@pytest.mark.skipif(...)`` conditional on a check to ``_check_soft_dependencies``
   for your new soft depenency.  Be sure that all soft dependencies which are imported for testing are imported within the test funciton itself,
   rather than for the whole module!  This decorator will then skip your test unless the system has the required packages installed.  Doing this is
   helpful for any users running ``check_estimator`` on all estimators, or a full local `pytest` run without the required soft dependency.
   Again, see the tests for pydarima (in forecasting) for a concrete example.

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
