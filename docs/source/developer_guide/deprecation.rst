.. _developer_guide_deprecation:

===========
Deprecation
===========

.. note::

    For planned changes and upcoming releases, see our :ref:`roadmap`.

Description
===========

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
========

In the examples below, the :code:`deprecated` decorator will raise a FutureWarning saying that the functionality has been deprecated since version 0.8.0 and will be remove in version 0.10.0.

Functions
---------

.. code-block::

    from deprecated.sphinx import deprecated

    @deprecated(version="0.8.0", reason="my_old_function will be removed in v0.10.0", category=FutureWarning)
    def my_old_function(x, y):
        return x + y

Methods
-------

.. code-block::

    from deprecated.sphinx import deprecated

    class MyClass:

        @deprecated(version="0.8.0", reason="my_old_method will be removed in v0.10.0", category=FutureWarning)
        def my_old_method(self, x, y):
            return x + y

Classes
-------

.. code-block::

    from deprecated.sphinx import deprecated

    @deprecated(version="0.8.0", reason="MyOldClass will be removed in v0.10.0", category=FutureWarning)
    class MyOldClass:
        pass
