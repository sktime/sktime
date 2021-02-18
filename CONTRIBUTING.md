How to contribute
=================

Welcome to our contributing guidelines!

sktime is a community-driven project and your help is extremely welcome. If you get stuck, please don't hesitate to [chat with us](https://gitter.im/sktime/community) or [raise an issue](https://github.com/alan-turing-institute/sktime/issues/new/choose).

To find out more about how to take part in sktime's community, check out our [governance document](https://www.sktime.org/en/latest/governance.html).


Contents
--------

- [How to get started](#how-to-get-started)
- [Where to contribute](#where-to-contribute)
  - [Areas of contribution](#areas-of-contribution)
  - [Roadmap](#roadmap)
- [Acknowledging contributions](#acknowledging-contributions)
- [Reporting bugs](#reporting-bugs)
- [Git and GitHub workflow](#git-and-github-workflow)
- [Continuous integration](#continuous-integration)
  - [Code quality checks](#code-quality-checks)
  - [Unit testing](#unit-testing)
  - [Test coverage](#test-coverage)
- [API design](#API-design)
- [Documentation](#documentation)
- [Dependencies](#dependencies)
- [Coding style](#coding-style)
- [Infrastructure](#infrastructure)
- [Release instructions](#release-instructions)


How to get started
------------------

We are particularly motivated to support new and/or anxious contributors and people who are looking to learn and develop their skills.

* **The Turing Way**. A great [handbook](https://the-turing-way.netlify.app/welcome) and [community](https://github.com/alan-turing-institute/the-turing-way) for open science to find lots of useful resources. Check out their [Guide for Reproducible Research](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html) to get started and learn more about open-source collaboration.
* **GitHub's Open Source Guides.** Take a look at their [How to Contribute Guide](https://opensource.guide/how-to-contribute/) to find out more about what it means to contribute.
* **scikit-learn's developer guide.** sktime follows [scikit-learn](https://scikit-learn.org/stable/)'s API whenever possible. We assume basic familiarity with scikit-learn. If you're new to scikit-learn, take a look at their [getting-started guide](https://scikit-learn.org/stable/getting_started.html). If you're already familiar with scikit-learn, you may still learn something new from their [developers' guide](https://scikit-learn.org/stable/developers/index.html).
* **Good-first issues.** A good place for starting to contribute to sktime is our list of [good-first issues](https://github.com/alan-turing-institute/sktime/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22). If you are interested in one of them, please comment on the issue or [chat to us](https://gitter.im/sktime/community).
* **Mentorship programme.** We have also launched sktime's own mentorship programme. You can find out more and apply on our [website](https://www.sktime.org/en/latest/mentoring.html)!


Where to contribute
-------------------

### Areas of contribution

We value all kinds of contributions - not just code. The following table gives an overview of key contribution areas.

| Area               | Description                                                                                                           |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- |
| Documentation      | Improve or add docstrings, glossary terms, the user guide, and the example notebooks                                  |
| Testing            | Report bugs, improve or add unit tests, conduct field testing on real-world data sets                                 |
| Code               | Improve or add functionality, fix bugs                                                                                |
| Mentoring          | Onboarding and mentoring of new contributors                                                                          |
| Outreach           | Organize talks, tutorials or workshops, write blog posts                                                              |
| Maintenance        | Improve development operations (continuous integration pipeline, GitHub bots), manage and review issues/pull requests |
| API design         | Design interfaces for estimators and other functionality                                                              |
| Project management | Finding funding, organising meetings, initiating new collaborations                                                   |

### Roadmap

For a more detailed overview of current and future work, check out our [development roadmap](https://github.com/alan-turing-institute/sktime/issues/228).


Acknowledging contributions
---------------------------

We follow the [all-contributors specification](https://allcontributors.org) and recognise various types of contributions. Take a look at our past and current [contributors](https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTORS.md)!

If you are a new contributor, please make sure we add you to our list of contributors. All contributions are recorded in [.all-contributorsrc](https://github.com/alan-turing-institute/sktime/blob/master/.all-contributorsrc).

If we have missed anything, please [chat with us](https://gitter.im/sktime/community), [raise an issue](https://github.com/alan-turing-institute/sktime/issues/new/choose) or create a PR!


Reporting bugs
--------------

We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish to see a feature implemented.

It is recommended to check that your issue complies with the following rules before submitting:

- Verify that your issue is not being currently addressed by other [issues](https://github.com/alan-turing-institute/sktime/issues) or [pull requests](https://github.com/alan-turing-institute/sktime/pulls).
- Please ensure all code snippets and error messages are formatted in appropriate code blocks. See [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks).
- Please be specific about what estimators and/or functions are involved and the shape of the data, as appropriate; please include a [reproducible](https://stackoverflow.com/help/mcve) code snippet or link to a [gist](https://gist.github.com). If an exception is raised, please provide the traceback.


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

We use continuous integration services on GitHub to automatically check if new pull requests do not break anything and meet code quality standards such as a common [coding style](#Coding-style).

### Code quality checks
To check if your code meets our code quality standards, you can automatically run these checks before you make a new commit using the [pre-commit](https://pre-commit.com) workflow:

1. Install pre-commit:

  ```bash
 pip install pre-commit
 ```

2. Set up pre-commit:
  ```bash
  pre-commit install
  ```

Once installed, pre-commit will automatically run our code quality checks on the files you changed whenenver you make a new commit.

You can find our pre-commit configuration in [.pre-commit-config.yaml](https://github.com/alan-turing-institute/sktime/blob/master/.pre-commit-config.yaml). Our flake8 configuration can be found in [setup.cfg](https://github.com/alan-turing-institute/sktime/blob/master/setup.cfg).

If you want to exclude some line of code from being checked, you can add a `# noqa` (no quality assurance) comment at the end of that line.

### Unit testing
We use [pytest](https://docs.pytest.org/en/latest/) for unit testing. To check if your code passes all tests locally, you need to install the development version of sktime and all extra dependencies.

1.  Install all extra requirements from the root directory of sktime:

    ```bash
    pip install -r build_tools/requirements.txt
    ```

2. Install the development version of sktime:

    ```bash
    pip install -e .
    ```

    This installs an editable [development version](https://pip.pypa.io/en/stable/reference/pip_install/#editable-installs) of sktime which will include the changes you make. For trouble shooting on different operating systems, please see our detailed [installation instructions](https://www.sktime.org/en/latest/installation.html).

2.  To run all unit tests, run:

    ```bash
    pytest sktime/
    ```

### Test coverage
We use [coverage](https://coverage.readthedocs.io/en/coverage-5.3/) via the [pytest-cov](https://github.com/pytest-dev/pytest-cov) plugin and [codecov](https://codecov.io) to measure and compare test coverage of our code.

API design
----------

The general design approach we follow in sktime is described in the paper ["Designing Machine Learning Toolboxes: Concepts, Principles and Patterns"](https://arxiv.org/abs/2101.04938). This is a first draft of the paper, feedback and improvement suggestions are very welcome!

Documentation
-------------

We use [sphinx](https://www.sphinx-doc.org/en/master/) and [readthedocs](https://readthedocs.org/projects/sktime/) to build and deploy our online documention. You can find our online documentation [here](https://www.sktime.org/en/latest/).

The source files used to generate the online documentation can be found in [docs/source/](https://github.com/alan-turing-institute/sktime/tree/master/docs/source). For example, the main configuration file for sphinx is [conf.py](https://github.com/alan-turing-institute/sktime/blob/master/docs/source/conf.py) and the main page is [index.rst](https://github.com/alan-turing-institute/sktime/blob/master/docs/source/index.rst). To add new pages, you need to add a new `.rst` file and include it in the `index.rst` file.

To build the documentation locally, you need to install a few extra dependencies listed in [docs/requirements.txt](https://github.com/alan-turing-institute/sktime/blob/master/docs/requirements.txt).

1. Install extra requirements from the root directory, run:

    ```bash
    pip install -r docs/requirements.txt
    ```

2. To build the website locally, run:

    ```bash
    make docs
    ```

You can find the generated files in the `sktime/docs/_build/` folder. To view the website, open `sktime/docs/_build/html/index.html` with your preferred web browser.


Dependencies
------------

We try to keep the number of core dependencies to a minimum and rely on other packages as soft dependencies when feasible.

> A soft dependency is a dependency that is only required to import certain modules, but not necessary to use most functionality. A soft dependency is not installed automatically when the package is installed. Instead, users need to install it manually if they want to use a module that requires a soft dependency.

If you add a new dependency or change the version of an existing one, you need to update the following files:

 - [sktime/setup.py](https://github.com/alan-turing-institute/sktime/blob/master/setup.py) for package installation and minimum version requirements,
 - [build_tools/requirements.txt](https://github.com/alan-turing-institute/sktime/blob/master/build_tools/requirements.txt) for continuous integration and distribution,
 - [docs/requirements.txt](https://github.com/alan-turing-institute/sktime/blob/master/docs/requirements.txt) for building the documentation,
 - [.binder/requirements.txt](https://github.com/alan-turing-institute/sktime/blob/master/.binder/requirements.txt) for launching notebooks on Binder.

If a user is missing a soft dependency, we want to raise a more user-friendly error message than just a `ModuleNotFound` exception. This is handled through our `_check_soft_dependencies` defined [here](https://github.com/alan-turing-institute/sktime/blob/master/sktime/utils/check_imports.py).

We also use contiunous integration tests to check if all soft dependencies are properly isolated to specific modules. So, if you add a soft dependency, please make sure to add it [here](https://github.com/alan-turing-institute/sktime/blob/master/build_tools/azure/check_soft_dependencies.py) together with the module that depends on it.


Coding style
------------

We follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) coding guidelines. A good example can be found [here](https://gist.github.com/nateGeorge/5455d2c57fb33c1ae04706f2dc4fee01).

We use the [pre-commit](#Code-quality-checks) workflow together with [black](https://black.readthedocs.io/en/stable/) and [flake8](https://flake8.pycqa.org/en/latest/) to automatically apply consistent formatting and check whether your contribution complies with the PEP8 style.

For docstrings, we use the [numpy docstring standard](https://numpydoc.readthedocs.io/en/latest/format.html\#docstring-standard).

In addition, we add the following guidelines:

- Please check out our [glossary of terms](https://github.com/alan-turing-institute/sktime/wiki/Glossary).
-   Use underscores to separate words in non-class names: `n_instances` rather than `ninstances`.
-   Avoid multiple statements on one line. Prefer a line return after a control flow statement (`if`/`for`).
-   Use absolute imports for references inside sktime.
-   Please don't use `import *` in the source code. It is considered harmful by the official Python recommendations. It makes the code harder to read as the origin of symbols is no longer explicitly referenced, but most important, it prevents using a static analysis tool like pyflakes to automatically find bugs.


Infrastructure
--------------

This section gives an overview of the infrastructure and continuous integration services we use.

| Platform                                                                  | Operation                                                                       | Configuration                                                                                                                                                                  |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Appveyor](https://ci.appveyor.com/project/mloning/sktime)                | Build/test/distribute on Windows                                                | [.appveyor.yml](https://github.com/alan-turing-institute/sktime/blob/master/.appveyor.yml)                                                                                       |
| [Azure Pipelines](https://dev.azure.com/mloning/sktime)                   | Build/test/distribute on Linux ([manylinux](https://github.com/pypa/manylinux)) | [azure-pipelines.yml](https://github.com/alan-turing-institute/sktime/blob/master/azure-pipelines.yml)                                                                         |
| [GitHub Actions](https://docs.github.com/en/free-pro-team@latest/actions) | Build/test/distribute on MacOS; Code quality checks                                                           | [.github/workflows/](https://github.com/alan-turing-institute/sktime/blob/master/.github/workflows/)                                           |
| [Read the Docs](https://readthedocs.org)                                  | Build/deploy documentation                                                      | [.readthedocs.yml](https://github.com/alan-turing-institute/sktime/blob/master/.github/workflows/code-quality.yml)                                                             |
| [Codecov](https://codecov.io)                                              | Test coverage                                                                   | [.codecov.yml](https://github.com/alan-turing-institute/sktime/blob/master/.codecov.yml), [.coveragerc](https://github.com/alan-turing-institute/sktime/blob/master/.coveragerc) |

Additional scripts used for building, unit testing and distribution can be found in [build_tools/](https://github.com/alan-turing-institute/sktime/tree/master/build_tools).


Release instructions
--------------------

This section is for core developers. To make a new release, you need push-to-write access on our master branch.

sktime is not a pure Python package and depends on some non-Python code including Cython and C. We distribute compiled files, called wheels, for different operating systems and Python versions. More details can be found here:

* [Python guide for packaging](https://packaging.python.org/guides/),
* [Cython guide on compilation/distribution](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html).

We use continuous integration services to automatically build and distribute wheels across platforms and version. The release process is triggered on our continuous integration services by pushing a [tagged commit](https://git-scm.com/book/en/v2/Git-Basics-Tagging) using [semantic versioning](https://semver.org). Pushing a new tag will trigger a new build on the continuous integration services which will provide the wheels for different platforms and automatically upload them to PyPI. You can see all uploaded wheels [here](https://pypi.org/simple/sktime/).

To make the release process easier, we have an interactive script that you can follow. Simply run:

```bash
make release
```

This calls [build_tools/make_release.py](https://github.com/alan-turing-institute/sktime/blob/master/build_tools/make_release.py) and will guide you through the release process.
