.. _developer_guide_deprecation:

===========
Deprecation
===========

Before we can make changes to sktime's user interface, we need to make sure that users have time to make the necessary adjustments in their code.
For this reason, we first need to deprecate functionality and change it only in a next release.

.. note::

    For upcoming changes and next releases, see our `Milestones <https://github.com/alan-turing-institute/sktime/milestones?direction=asc&sort=due_date&state=open>`_.
    For our long-term plan, see our :ref:`roadmap`.

Deprecation policy
==================

sktime `releases <https://github.com/alan-turing-institute/sktime/releases>`_ follow `semantic versioning <https://semver.org>`_.
A release number denotes <major>.<minor>.<patch> versions.

Our current deprecation policy is that we remove functionality after one minor release cycle.
For example, if some functionality has been deprecated in v0.9.0 or any patch of that minor version, we will raise a warning throughout the next minor version v0.10.* and all of its patches.
The functionality will then be removed in v0.11.0.

Deprecation process
===================

The deprecation process is as follows:

* **Raise a warning.** For all deprecated functionality, we raise a :code:`FutureWarning`. The warning message should give the version number when the functionality will be changed and describe the new usage.
* **Add a TODO comment in the code.** Add the following TODO comment to all pieces of code that can be removed: :code:`TODO: remove in <version-number>`. For example, :code:`TODO: remove in v0.11.0`. The TODO comment should describe which steps need to be taken for deprecation (e.g. removal of arguments, removal of functions or blocks of code). If changes need to be applied across multiple places, place multiple TODO comments.
* **Remove deprecated functionality.** As part of the release process, all deprecated functionality that is due to be removed will be removed by searching for the TODO comments.


To raise the warning, we use the `deprecated <https://deprecated.readthedocs.io/en/latest/index.html>`_ package.
The package provides depreciation helper functions such as the :code:`deprecated` decorator.
When importing it from :code:`deprecated.sphinx`, it automatically adds a deprecation message to the docstring.
You can decorate functions, methods or classes.

Examples
--------

In the examples below, the :code:`deprecated` decorator will raise a :code:`FutureWarning` saying that the functionality has been deprecated since version 0.9.0 and will be removed in version 0.11.0.

Functions
~~~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    # TODO: remove in v0.11.0
    @deprecated(version="0.9.0", reason="my_old_function will be removed in v0.11.0", category=FutureWarning)
    def my_old_function(x, y):
        return x + y

Methods
~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    class MyClass:

        # TODO: remove in v0.11.0
        @deprecated(version="0.9.0", reason="my_old_method will be removed in v0.11.0", category=FutureWarning)
        def my_old_method(self, x, y):
            return x + y

Classes
~~~~~~~

.. code-block::

    from deprecated.sphinx import deprecated

    # TODO: remove in v0.11.0
    @deprecated(version="0.9.0", reason="MyOldClass will be removed in v0.11.0", category=FutureWarning)
    class MyOldClass:
        pass
