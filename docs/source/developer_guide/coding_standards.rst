.. _coding_standards:

Coding standards
================

.. contents::
   :local:

Coding style
------------

We follow the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__
coding guidelines. A good example can be found
`here <https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01>`__.

We use the `pre-commit <#Code-quality-checks>`_ workflow together with
`black <https://black.readthedocs.io/en/stable/>`__ and
`flake8 <https://flake8.pycqa.org/en/latest/>`__ to automatically apply
consistent formatting and check whether your contribution complies with
the PEP8 style.

For docstrings, we use the `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_, along with sktime specific conventions described in our :ref:`developer_guide`'s :ref:`documentation section <developer_guide_documentation>`.

In addition, we add the following guidelines:

-  Please check out our :ref:`glossary`.
-  Use underscores to separate words in non-class names: ``n_instances``
   rather than ``ninstances``.
-  Avoid multiple statements on one line. Prefer a line return after a
   control flow statement (``if``/``for``).
-  Use absolute imports for references inside sktime.
-  Please don’t use ``import *`` in the source code. It is considered
   harmful by the official Python recommendations. It makes the code
   harder to read as the origin of symbols is no longer explicitly
   referenced, but most important, it prevents using a static analysis
   tool like pyflakes to automatically find bugs.

.. _infrastructure::

Dependencies
------------

We try to keep the number of core dependencies to a minimum and rely on
other packages as soft dependencies when feasible.

.. note::

   A soft dependency is a dependency that is only required to import
   certain modules, but not necessary to use most functionality. A soft
   dependency is not installed automatically when the package is
   installed. Instead, users need to install it manually if they want to
   use a module that requires a soft dependency.

If you add a new dependency or change the version of an existing one,
you need to update the following file:

-  `pyproject.toml <https://github.com/alan-turing-institute/sktime/blob/main/pyproject.toml>`__
   following the `PEP 621 <https://www.python.org/dev/peps/pep-0621/>`_ convention all dependencies
   including build time dependencies and optional dependencies are specified in this file.

If a user is missing a soft dependency, we raise a user-friendly error message.
This is handled through our ``_check_soft_dependencies`` defined
`here <https://github.com/alan-turing-institute/sktime/blob/main/sktime/utils/validation/_dependencies.py>`__.

We use continuous integration tests to check if all soft
dependencies are properly isolated to specific modules.
If you add a new soft dependency, make sure to add it
`here <https://github.com/alan-turing-institute/sktime/blob/main/build_tools/azure/check_soft_dependencies.py>`__
together with the module that depends on it.

API design
----------

The general design approach of sktime is described in the
paper `“Designing Machine Learning Toolboxes: Concepts, Principles and
Patterns” <https://arxiv.org/abs/2101.04938>`__.

.. note::

   This is a first draft of the paper.
   Feedback and improvement suggestions are very welcome!


Deprecation
-----------

.. note::

    For planned changes and upcoming releases, see our :ref:`roadmap`.

Description
~~~~~~~~~~~

Before removing or changing sktime's public API, we need to deprecate it.
This gives users and developers time to transition to the new functionality.

Once functionality is deprecated, it will be removed in the next minor release.
We follow `semantic versioning <https://semver.org>`_, where the version number denotes <major>.<minor>.<patch>.
For example, if we add the deprecation warning in release v0.9.0, we remove
the functionality in release v0.10.0.

Our current deprecation process is as follows:

* We raise a `FutureWarning <https://docs.python.org/3/library/exceptions.html#FutureWarning>`_. The warning message should the give the version number when the functionality will be removed and describe the new usage.

* We add a to-do comments to the lines of code that can be removed, with the version number when the code can be removed. For example, :code:`TODO: remove in v0.10.0`.

* We remove all deprecated functionality as part of the release process, searching for the to-do comments.

We use the `deprecated <https://deprecated.readthedocs.io/en/latest/index.html>`_ package for depreciation helper functions.

To deprecate functionality, we use the :code:`deprecated` decorator.
When importing it from :code:`deprecated.sphinx`, it automatically adds a deprecation message to the docstring.
You can deprecate functions, methods or classes.

Examples
~~~~~~~~

In the examples below, the :code:`deprecated` decorator will raise a FutureWarning saying that the functionality has been deprecated since version 0.8.0 and will be remove in version 0.10.0.

Functions
~~~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    @deprecated(version="0.8.0", reason="my_old_function will be removed in v0.10.0", category=FutureWarning)
    def my_old_function(x, y):
        return x + y

Methods
~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    class MyClass:

        @deprecated(version="0.8.0", reason="my_old_method will be removed in v0.10.0", category=FutureWarning)
        def my_old_method(self, x, y):
            return x + y

Classes
~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    @deprecated(version="0.8.0", reason="MyOldClass will be removed in v0.10.0", category=FutureWarning)
    class MyOldClass:
        pass
