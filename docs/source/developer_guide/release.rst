.. _release:

Releases
========

This section is for core developers. To make a new release, you need
push-to-write access on our main branch.

`sktime` is distributed as the so-called wheels, for
different operating systems and Python versions.

.. note::

   For more details, see the `Python guide for packaging <https://packaging.python.org/guides/>`__.


We use :ref:`continuous integration <continuous_integration>` services to automate the building of wheels on different platforms.
The release process is triggered by pushing a non-annotated `tagged
commit <https://git-scm.com/book/en/v2/Git-Basics-Tagging>`__ using
`semantic versioning <https://semver.org>`__.
Pushing a new tag will build the wheels for different platforms and upload them to PyPI.

You can see all available wheels `here <https://pypi.org/simple/sktime/>`__.

To make the release process easier, we have an interactive script that
you can follow. Simply run:

.. code:: bash

   make release

This calls
`build_tools/make_release.py <https://github.com/alan-turing-institute/sktime/blob/main/build_tools/make_release.py>`__
and will guide you through the release process.
