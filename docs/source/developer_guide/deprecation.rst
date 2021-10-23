.. _developer_guide_deprecation:

===========
Deprecation
===========

Before removing or changing sktime's public API, we need to deprecate it.
This gives users and developers time to transition to the new functionality.

Once functionality is deprecated, it will be removed in the next release.
For example, if we add the deprecation warning in release v0.9.0, we remove
the functionality in release v0.10.0.

Our current deprecation process is as follows:

* We raise a `FutureWarning <https://docs.python.org/3/library/exceptions.html#FutureWarning>`_. The warning message should the give the version number when the functionality will be removed and describe the new usage.

* We add a to-do comments to the lines of code that can be removed, with the version number when the code can be removed. For example, :code:`TODO: remove in v0.10.0`.

* We remove all deprecated functionality as part of the release process, searching for the to-do comments.

We use the `deprecated <https://deprecated.readthedocs.io/en/latest/index.html>`_ package for easy depreciation helper functions.

For planned changes and upcoming releases, see our :ref:`roadmap`.
