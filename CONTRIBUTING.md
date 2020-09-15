How to contribute
=================

Welcome to our contributing guidelines! sktime is a community-driven project and your help is extremely welcome! If you get stuck, please don't hesitate to [chat with us](https://gitter.im/sktime/community) or [raise an issue](https://github.com/alan-turing-institute/sktime/issues/new/choose).

Contents
--------

* [Areas of contribution](#Areas-of-contribution)
* [Getting started](#Getting-started)
* [Git and GitHub workflow](#Git-and-GitHub-workflow)
* [Continuous integration](#Continuous-integration)
* [Documentation](#Documentation)
* [Coding style](#Coding-style)
* [Pull request checklist](#Pull-request-checklist)
* [Reporting bugs](#Reporting-bugs)


Areas of contribution
---------------------

We value all kinds of contributions - not just code. We follow the
 [allcontributors specification](https://allcontributors.org) and recognise various types of contributions. 
 
Check out our [list of contributors](https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTORS.md). 

The following table gives an overview of key contribution areas. For a more detailed overview, go to our [development roadmap](https://github.com/alan-turing-institute/sktime/issues/228). 

| Area | Description | 
|---|---|
| Documentation | Improve or add docstrings, glossary terms, the user guide, and the example notebooks. |
| Testing | Report bugs, improve or add unit tests, conduct field testing on real-world data sets. | 
| Code | Improve or add functionality, fix bugs. | 
| Mentoring | Onboarding and mentoring of new contributors |
| Outreach | Organize talks, tutorials or workshops, write blog posts. |
| Maintenance | Improve development operations (continuous integration pipeline, GitHub bots), manage and review issues/pull requests. |
| API design | Design interfaces for estimators and other functionality. | 
| Project management. | Finding funding, organising meetings, initiating new collaborations. |


Getting started
---------------

### Good-first issues

A good place to start is our list of [good-first issues](https://github.com/alan-turing-institute/sktime/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22). If you are interested in one of them, simply comment on the issue or [chat to us](https://gitter.im/sktime/community).

### Mentorship programme
We are particularly motivated to support new and/or anxious contributors and people who are looking to learn and develop their skills. For this reason, we have launched sktime's own mentorship programme. 

Find out more and apply on our [website](https://sktime.org/mentoring.html)!


Git and GitHub workflow
-----------------------

The preferred workflow for contributing to sktime's repository is to fork the [main repository](https://github.com/alan-turing-institute/sktime/) on GitHub, clone, and develop on a new branch.

1.  Fork the [project repository](https://github.com/alan-turing-institute/sktime) by clicking on the \'Fork\' button near the top right of the page. This creates a copy of the code under your GitHub user account. For more details on how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. [Clone](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) your fork of the sktime repo from your GitHub account to your local disk:

    ```bash
    git clone git@github.com:USERNAME/sktime.git
    cd sktime
    ```

3. Configure and link the remote for your fork to the upstream repository:

    ```bash
    git remote -v
    git remote add upstream https://github.com/alan-turing-institute/sktime.git
    ```

4. Verify the new upstream repository you\'ve specified for your fork:

    ```bash
    git remote -v
    > origin    https://github.com/USERNAME/YOUR_FORK.git (fetch)
    > origin    https://github.com/YOUR_USERNAME/YOUR_FORK.git (push)
    > upstream  https://github.com/alan-turing-institute/sktime.git (fetch)
    > upstream  https://github.com/alan-turing-institute/sktime.git (push)
    ```

5. [Sync](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork) the `master` branch of your fork with the upstream repository:

    ```bash
    git fetch upstream
    git checkout master --track origin/master
    git merge upstream/master
    ```

6. Create a new `feature` branch from the `master` branch to hold your changes:

    ```bash
    git checkout master
    git checkout -b <my-feature-branch>
    ```

   Always use a `feature` branch. It\'s good practice to never work on the `master` branch! Name the `feature` branch after your contribution.

7. Develop your contribution on your feature branch. Add changed files using `git add` and then `git commit` files to record your changes in Git:

    ```bash
    git add <modified_files>
    git commit
    ```

8. When finished, push the changes to your GitHub account with:

    ```bash
    git push --set-upstream origin my-feature-branch
    ```

9. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork) to create a pull request from your fork. If your work is still work in progress, you can open a draft pull request. We recommend to open a pull request early, so that other contributors become aware of your work and can give you feedback early on. 

10. To add more changes, simply repeat steps 7 - 8. Pull requests are
 updated automatically if you push new changes to the same branch.

If any of the above seems like magic to you, please look up the [Git documentation](https://git-scm.com/documentation) on the web. If you get stuck, feel free to [chat with us](https://gitter.im/sktime/community) or [raise an issue](https://github.com/alan-turing-institute/sktime/issues/new/choose).

Continuous integration
----------------------

We use [pytest](https://docs.pytest.org/en/latest/) for unit testing, and continuous integration services on GitHub to automatically check if new pull requests do not break anything and comply with sktime's API.

sktime follows [scikit-learn](https://scikit-learn.org/stable/)'s API whenever possible, it'll be useful to take a look at their [developers' guide](https://scikit-learn.org/stable/developers/index.html).

To check if your code passes all tests locally, you need to install the development version of sktime and all extra dependencies. Steps:

1.  Install all extra requirements from the root directory of sktime:

    ```bash
    pip install -r build_tools/requirements.txt
    ```

2.  Install the development version from the root directory:

    ```bash
    pip install --editable .
    ```

    This installs a development version of sktime which will include all of your changes. For trouble shooting on different operating systems, please see our detailed [installation instructions](https://sktime.org/installation.html).

3.  To run all unit tests, run:

    ```bash
    pytest sktime/
    ```

Documentation
-------------

To build our online documentation and website locally, you need to install a few additional dependencies listed in [docs/requirements.txt](https://github.com/alan-turing-institute/sktime/blob/master/docs/requirements.txt). From the root directory, run:
 
 ```bash
pip install -r docs/requirements.txt
```  
For trouble shooting on different operating systems, please see our detailed [installation instructions](https://sktime.org/installation.html).

To build the website locally, run:

```bash
make docs
```

You can find the generated files in the `sktime/docs/_build/` folder. To view the website, open `sktime/docs/_build/html/index.html` with your preferred web browser.

Dependencies
------------

If you add a new dependency or change the version of a dependency, you need to update one or more of the following files: 

 - [sktime/setup.py](https://github.com/alan-turing-institute/sktime/blob/master/setup.py) for package installation, 
 - [build_tools/requirements.txt](https://github.com/alan-turing-institute/sktime/blob/master/build_tools/requirements.txt) for continuous integration and distribution,
 - [docs/requirements.txt](https://github.com/alan-turing-institute/sktime/blob/master/docs/requirements.txt) for generating the documentation,
 - [.binder/requirements.txt](https://github.com/alan-turing-institute/sktime/blob/master/.binder/requirements.txt) for launching notebooks on Binder.

We try to keep the number of core dependencies small and rely on other
 pacakges as soft dependencies when possible. 

 
Coding style
------------

We follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) coding guidelines. A good example can be found [here](https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01).

We use [flake8](https://flake8.pycqa.org/en/latest/) to automatically check whether your contribution complies with the PEP8 style. To check if your code locally, you can install and run flake8 in the root directory of sktime:

```bash
pip install flake8
flake8 sktime/
```

For docstrings, we use the [numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html\#docstring-standard).

In addition, we add the following guidelines:

- Please check out our [glossary of terms](https://github.com/alan-turing-institute/sktime/wiki/Glossary).
-   Use underscores to separate words in non-class names: `n_instances` rather than `ninstances`.
-   Avoid multiple statements on one line. Prefer a line return after a control flow statement (`if`/`for`).
-   Use absolute imports for references inside sktime.
-   Please don't use `import *` in any case. It is considered harmful by the official Python recommendations. It makes the code harder to read as the origin of symbols is no longer explicitly referenced, but most important, it prevents using a static analysis tool like pyflakes to automatically find bugs.

Pull request checklist
----------------------

We recommended that your contribution complies with the following rules
before you submit a pull request:

-   Give your pull request a helpful title that summarises what your contribution does. In some cases `Fix <ISSUE TITLE>` is enough. `Fix #<ISSUE NUMBER>` is not enough.
-   Often pull requests resolve one or more other issues (or pull requests). If merging your pull request means that some other issues/pull requests should be closed, you should [use keywords to create links to them](https://github.com/blog/1506-closing-issues-via-pull-requests/) (for example, `Fixes #1234`; multiple issues/PRs are allowed as long as each one is preceded by a keyword). Upon merging, those issues/pull requests will automatically be closed by GitHub. If your pull request is simply related to some other issues/PRs, create a link to them without using the keywords (for example, `See also #1234`).
-   All public methods should have informative docstrings with sample usage presented as doctests when appropriate.

Reporting bugs
--------------

We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the following rules before submitting:

- Verify that your issue is not being currently addressed by other [issues](https://github.com/alan-turing-institute/sktime/issues) or [pull requests](https://github.com/alan-turing-institute/sktime/pulls).
- Please ensure all code snippets and error messages are formatted in appropriate code blocks. See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).
- Please be specific about what estimators and/or functions are involved and the shape of the data, as appropriate; please include a [reproducible](https://stackoverflow.com/help/mcve) code snippet or link to a [gist](https://gist.github.com). If an exception is raised, please provide the traceback.

