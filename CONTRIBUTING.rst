How to contribute
=================

Welcome to our how-to-contribute guide!

If you get stuck, feel free to `contact us`_ or `raise an issue`_.

sktime follows `scikit-learn`_'s API whenever possible, it'll be useful to a look at their `developers’ guide`_.

Before starting to develop, you will need to install the development version of sktime. To install the development version, please see our `advanced
installation
instructions <https://alan-turing-institute.github.io/sktime/installation.html#development-version>`__.


The preferred workflow for contributing to sktime is to fork the `main
repository <https://github.com/alan-turing-institute/sktime/>`__ on
GitHub, clone, and develop on a branch. Steps:

1. Fork the `project
   repository <https://github.com/alan-turing-institute/sktime>`__ by
   clicking on the 'Fork' button near the top right of the page. This
   creates a copy of the code under your GitHub user account. For more
   details on how to fork a repository see `this
   guide <https://help.github.com/articles/fork-a-repo/>`__.
   
2. Clone your fork of the sktime repo from your GitHub account to your
   local disk:

.. code-block:: bash

    git clone git@github.com:YourLogin/sktime.git
    cd sktime


3. Configure a remote for your fork to the upstream.
   
.. code-block:: bash

   git remote -v
   git remote add upstream https://github.com/alan-turing-institute/sktime.git
   
4. Verify the new upstream repository you've specified for your fork.

.. code-block:: bash
   
   git remote -v
   > origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (fetch)
   > origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
   > upstream  https://github.com/alan-turing-institute/sktime.git (fetch)
   > upstream  https://github.com/alan-turing-institute/sktime.git (push)
   
5. Sync the ``dev`` branch of your fork with the upstream repository

.. code-block:: bash

   git fetch upstream
   git checkout dev
   git merge upstream/dev


6. Create a new ``feature`` branch from the ``dev`` branch to hold your
   changes:

.. code-block:: bash

    git checkout dev
    git checkout -b my-feature-branch

Always use a ``feature`` branch. It's good practice to never work on the
``master`` branch!

7. Develop the feature on your feature branch. Add changed files using
   ``git  add`` and then ``git commit`` files to record your changes in
   Git:

.. code-block:: bash

    git add modified_files
    git commit

8. When finished, push the changes to your GitHub account with:

.. code-block:: bash

    git push -u origin my-feature-branch

9. Follow `these
   instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`__
   to create a pull request from your fork. This will send an email to
   the committers.

If any of the above seems like magic to you, please look up the `Git
documentation <https://git-scm.com/documentation>`__ on the web, or ask
a friend or another contributor for help.



Pull request checklist
----------------------

We recommended that your contribution complies with the following rules
before you submit a pull request:

-  Follow the `PEP8 <https://www.python.org/dev/peps/pep-0008/>`__
   coding guidelines. A good example can be found
   `here <https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01>`__.
   In addition, we add the following guidelines:

   -  Use underscores to separate words in non-class names:
      ``n_instances`` rather than\ ``ninstances``.
   -  Avoid multiple statements on one line. Prefer a line return after
      a control flow statement (``if``/``for``).
   -  Use absolute imports for references inside sktime.
   -  Please don’t use ``import *`` in any case. It is considered
      harmful by the official Python recommendations. It makes the code
      harder to read as the origin of symbols is no longer explicitly
      referenced, but most important, it prevents using a static
      analysis tool like pyflakes to automatically find bugs.
   -  Use the `numpy docstring
      standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`__
      in all your docstrings.

-  Give your pull request a helpful title that summarises what your
   contribution does. In some cases ``Fix <ISSUE TITLE>`` is enough.
   ``Fix #<ISSUE NUMBER>`` is not enough.

-  Often pull requests resolve one or more other issues (or pull
   requests). If merging your pull request means that some other
   issues/PRs should be closed, you should `use keywords to create link
   to
   them <https://github.com/blog/1506-closing-issues-via-pull-requests/>`__
   (e.g., ``Fixes #1234``; multiple issues/PRs are allowed as long as
   each one is preceded by a keyword). Upon merging, those issues/PRs
   will automatically be closed by GitHub. If your pull request is
   simply related to some other issues/PRs, create a link to them
   without using the keywords (e.g., ``See also #1234``).
-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

Filing bugs
-----------

We use GitHub issues to track all bugs and feature requests; feel free
to open an issue if you have found a bug or wish to see a feature
implemented.

It is recommended to check that your issue complies with the following
rules before submitting:

-  Verify that your issue is not being currently addressed by other
   `issues <https://github.com/alan-turing-institute/sktime/issues>`__
   or `pull
   requests <https://github.com/alan-turing-institute/sktime/pulls>`__.

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks. See `Creating and highlighting code
   blocks <https://help.github.com/articles/creating-and-highlighting-code-blocks>`__.

-  Please be specific about what estimators and/or functions are
   involved and the shape of the data, as appropriate; please include a
   `reproducible <https://stackoverflow.com/help/mcve>`__ code snippet
   or link to a `gist <https://gist.github.com>`__. If an exception is
   raised, please provide the traceback.


.. _contact us: https://gitter.im/sktime/community
.. _scikit-learn: https://scikit-learn.org/stable/
.. _getting-started guide: https://scikit-learn.org/stable/getting_started.html
.. _developers’ guide: https://scikit-learn.org/stable/developers/index.html
.. _chat to us: https://gitter.im/sktime/community
.. _raise an issue: https://github.com/alan-turing-institute/sktime/issues/new/choose
.. _clone:
.. _fork: https://help.github.com/en/articles/fork-a-repo
.. _syncfork: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork


