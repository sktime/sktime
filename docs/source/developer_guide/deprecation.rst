.. _developer_guide_deprecation:

===========
Deprecation
===========

.. note::

    For upcoming changes and next releases, see our `milestones <https://github.com/alan-turing-institute/sktime/milestones?direction=asc&sort=due_date&state=open>`_.
    For our long-term plan, see our :ref:`roadmap`.

Before we can make changes to sktime's user interface, we need to make sure that users have time to make the necessary adjustments in their code.
For this reason, we first need to deprecate functionality and change it only in a next release.

When to deprecate code
======================

Our releases follow `semantic versioning <https://semver.org>`_.
A version number denotes <major>.<minor>.<patch> versions.

Our current deprecation policy is that we remove functionality after one minor release.
For example, if some functionality has been deprecated in v0.9.0, it will be removed in v0.10.0.

How to deprecate code
=====================

Our deprecation process is as follows:

* Raise a :code:`FutureWarning`. The warning message should give the version number when the functionality will be changed and describe the new usage.
* Add the following to-do comment to code that can be removed: :code:`TODO: remove in <version-number>`. For example, :code:`TODO: remove in v0.10.0`.
* Remove all deprecated functionality as part of the release process, searching for the to-do comments.

To deprecate functionality, we use the `deprecated <https://deprecated.readthedocs.io/en/latest/index.html>`_ package.
The package provides depreciation helper functions such as the :code:`deprecated` decorator.
When importing it from :code:`deprecated.sphinx`, it automatically adds a deprecation message to the docstring.
You can decorate functions, methods or classes.

Examples
--------

In the examples below, the :code:`deprecated` decorator will raise a :code:`FutureWarning`` saying that the functionality has been deprecated since version 0.9.0 and will be remove in version 0.10.0.

Functions
~~~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    @deprecated(version="0.9.0", reason="my_old_function will be removed in v0.10.0", category=FutureWarning)
    def my_old_function(x, y):
        return x + y

Methods
~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    class MyClass:

        @deprecated(version="0.9.0", reason="my_old_method will be removed in v0.10.0", category=FutureWarning)
        def my_old_method(self, x, y):
            return x + y

Classes
~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    @deprecated(version="0.9.0", reason="MyOldClass will be removed in v0.10.0", category=FutureWarning)
    class MyOldClass:
        pass
