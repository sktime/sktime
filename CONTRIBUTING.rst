How to contribute
=================

Getting started
---------------

Welcome to our how to get started guide! If you get stuck, feel free to `contact us`_.

1. We assume familiarity with `scikit-learn`_. If you haven’t work with
   scikit-learn before, take a look at their `getting-started guide`_.
   For more, just search the web, there are plenty of online videos and
   tutorials! Even if you’ve used scikit-learn before, it may still be
   useful to read through their `developers’ guide`_.

2. If you’re familiar with scikit-learn, a good starting point is to run our example notebooks on
   `Binder`_. This runs the notebooks in the cloud and doesn’t require
   you to install anything. Play around with the different notebooks to
   get to know sktime - if something doesn’t look right, `chat to us`_
   or `raise an issue`_.

3. Install sktime locally. This requires a working Python installation.
   To install Python, we recommend downloading the `Anaconda Distribution`_ and following their installation instructions `here`_.
   For the example notebooks, you'll also need Jupyter Notebooks (which should come with the
   default Anaconda Distribution). Here’s a `guide`_ on how to get
   started with Jupyter. We also recommend using `conda environments`_.
   Once you have a working Python environment, you should be able
   to run ``pip install sktime``.

4. Having installed sktime, you can try to run the example notebooks locally. You need
   to download or `clone`_ the repository from GitHub. You should then
   be able to `activate your conda environment`_ and launch Jupyter
   notebooks by running ``jupyter notebook`` in the root directory of
   the downloaded repository.

5. In order to contribute to sktime, you need to `fork`_ the repository
   on GitHub and create a local
   `clone <https://help.github.com/en/articles/cloning-a-repository>`__
   of it, following the instructions in our `contributing guidelines`_.

6. You may want install the development version of sktime based on the
   clone of your sktime fork. Check out our advanced installation
   instructions
   `here <https://github.com/alan-turing-institute/sktime/blob/dev/README.rst>`__
   - again, if you get stuck, simply `chat to us`_.

7. See if you can run the unit tests using `pytest`_ by running
   ``pytest`` in the root directory of the repository.

Once everything is set up and working, you can start contributing!

.. _Code of Conduct: https://github.com/alan-turing-institute/sktime/blob/master/CODE_OF_CONDUCT.rst
.. _create a PR: https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTING.rst
.. _contact us: https://gitter.im/sktime/community
.. _scikit-learn: https://scikit-learn.org/stable/
.. _getting-started guide: https://scikit-learn.org/stable/getting_started.html
.. _developers’ guide: https://scikit-learn.org/stable/developers/index.html
.. _Binder: https://mybinder.org/v2/gh/alan-turing-institute/sktime/master?filepath=examples
.. _chat to us: https://gitter.im/sktime/community
.. _raise an issue: https://github.com/alan-turing-institute/sktime/issues/new/choose
.. _Anaconda Distribution: https://www.anaconda.com
.. _here: https://docs.anaconda.com/anaconda/user-guide/getting-started/
.. _guide: https://jupyter.org/install.html
.. _conda environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _clone:
.. _activate your conda environment: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
.. _fork: https://help.github.com/en/articles/fork-a-repo
.. _contributing guidelines: https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTING.md
.. _pytest: https://docs.pytest.org/en/latest/contents.html

Contributing
------------

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

3. Create a new ``feature`` branch from the ``dev`` branch to hold your
   changes:

.. code-block:: bash

    git checkout dev
    git checkout -b my-feature-branch

Always use a ``feature`` branch. It's good practice to never work on the
``master`` branch!

4. Develop the feature on your feature branch. Add changed files using
   ``git  add`` and then ``git commit`` files to record your changes in
   Git:

.. code-block:: bash

    git add modified_files
    git commit

5. When finished, push the changes to your GitHub account with:

.. code-block:: bash

    git push -u origin my-feature-branch

6. Follow `these
   instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`__
   to create a pull request from your fork. This will send an email to
   the committers.

If any of the above seems like magic to you, please look up the `Git
documentation <https://git-scm.com/documentation>`__ on the web, or ask
a friend or another contributor for help.

To install the development version, please see our `advanced
installation
instructions <https://alan-turing-institute.github.io/sktime/installation.html#development-version>`__.

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


