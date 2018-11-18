How to contribute
-----------------

The preferred workflow for contributing to sktime is to fork the
[main repository](https://github.com/kiraly-group/sktime) on
GitHub, clone, and develop on a branch. Steps:

1. Fork the [project repository](https://github.com/kiraly-group/sktime)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the sktime repo from your GitHub account to your local 
disk:

   ```bash
   $ git clone git@github.com:YourLogin/sktime.git
   $ cd sktime
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature-branch
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git
 add`` and then ``git commit`` files to record your changes in Git:
   ```bash
   $ git add modified_files
   $ git commit
   ```

5. When finished, push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature-branch
   ```

5. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend 
or another contributor for help.

Pull Request Checklist
----------------------

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  Follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) coding 
guidelines. A good example can be found [here](https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01).
In addition, we add the following guidelines:
    - Use underscores to separate words in non class names: n_samples rather than
  nsamples.
    - Avoid multiple statements on one line. Prefer a line return after a 
  control flow statement (`if`/`for`).
    - Use relative imports for references inside sktime.
    - Unit tests are an exception to the previous rule; they should use 
  absolute imports, exactly as client code would. A corollary is that, if 
  `sktime.foo` exports a class or function that is implemented in `sktime.foo.bar.baz`, 
  the test should import it from `sktime.foo`.
    - Please don’t use `import *` in any case. It is considered harmful by the 
 official Python recommendations. It makes the code harder to read as the 
 origin of symbols is no longer explicitly referenced, but most important, 
 it prevents using a static analysis tool like pyflakes to automatically 
 find bugs in sktime.
    - Use the [numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) in all your docstrings.




  

-  Use, when applicable, the validation tools and scripts in the
   `sklearn.utils` submodule.  

-  Give your pull request a helpful title that summarises what your
   contribution does. In some cases `Fix <ISSUE TITLE>` is enough.
   `Fix #<ISSUE NUMBER>` is not enough.

-  Often pull requests resolve one or more other issues (or pull requests).
   If merging your pull request means that some other issues/PRs should
   be closed, you should
   [use keywords to create link to them](https://github.com/blog/1506-closing-issues-via-pull-requests/)
   (e.g., `Fixes #1234`; multiple issues/PRs are allowed as long as each one
   is preceded by a keyword). Upon merging, those issues/PRs will
   automatically be closed by GitHub. If your pull request is simply related
   to some other issues/PRs, create a link to them without using the keywords
   (e.g., `See also #1234`).

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  Please prefix the title of your pull request with `[MRG]` (Ready for
   Merge), if the contribution is complete and ready for a detailed review.
   Two core developers will review your code and change the prefix of the pull
   request to `[MRG + 1]` and `[MRG + 2]` on approval, making it eligible
   for merging. An incomplete contribution -- where you expect to do more work before
   receiving a full review -- should be prefixed `[WIP]` (to indicate a work
   in progress) and changed to `[MRG]` when it matures. WIPs may be useful
   to: indicate you are working on something to avoid duplicated work,
   request broad review of functionality or API, or seek collaborators.
   WIPs often benefit from the inclusion of a
   [task list](https://github.com/blog/1375-task-lists-in-gfm-issues-pulls-comments)
   in the PR description.

-  When adding additional functionality, provide at least one
   example script in the ``examples/`` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available in scikit-learn.

-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and
   the example.

-  The documentation should also include expected time and space
   complexity of the algorithm and scalability, e.g. "this algorithm
   can scale to a large number of samples > 100000, but does not
   scale in dimensionality: n_features is expected to be lower than
   100".

You can also check for common programming errors with the following
tools:

-  Code with good unittest **coverage** (at least 80%), check with:

  ```bash
  $ pip install pytest pytest-cov
  $ pytest --cov sklearn path/to/tests_for_package
  ```

-  No flake8 warnings, check with:

  ```bash
  $ pip install flake8
  $ flake8 path/to/module.py
  ```

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the GitHub issue).

Filing bugs
-----------
We use GitHub issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the
following rules before submitting:

-  Verify that your issue is not being currently addressed by other
   [issues](https://github.com/scikit-learn/sktime/issues?q=)
   or [pull requests](https://github.com/sktime/sktime/pulls?q=).

-  Please ensure all code snippets and error messages are formatted in
   appropriate code blocks.
   See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).

-  Please be specific about what estimators and/or functions are involved
   and the shape of the data, as appropriate; please include a
   [reproducible](https://stackoverflow.com/help/mcve) code snippet
   or link to a [gist](https://gist.github.com). If an exception is raised,
   please provide the traceback.
