.. -*- mode: rst -*-

.. |github| image:: https://img.shields.io/github/workflow/status/alan-turing-institute/sktime/build-and-test?logo=github
.. _github: https://github.com/alan-turing-institute/sktime/actions?query=workflow%3Abuild-and-test

.. |appveyor| image:: https://img.shields.io/appveyor/ci/mloning/sktime/main?logo=appveyor
.. _appveyor: https://ci.appveyor.com/project/mloning/sktime

.. |pypi| image:: https://img.shields.io/pypi/v/sktime?color=orange
.. _pypi: https://pypi.org/project/sktime/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/sktime
.. _conda: https://anaconda.org/conda-forge/sktime

.. |discord| image:: https://img.shields.io/static/v1?logo=discord&label=discord&message=chat&color=lightgreen
.. _discord: https://discord.com/invite/gqSab2K

.. |gitter| image:: https://img.shields.io/static/v1?logo=gitter&label=gitter&message=chat&color=lightgreen
.. _gitter: https://gitter.im/sktime/community

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/alan-turing-institute/sktime/main?filepath=examples

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg
.. _zenodo: https://doi.org/10.5281/zenodo.3749000

.. |azure| image:: https://img.shields.io/azure-devops/build/mloning/30e41314-4c72-4751-9ffb-f7e8584fc7bd/1/main?logo=azure-pipelines
.. _azure: https://dev.azure.com/mloning/sktime/_build

.. |codecov| image:: https://img.shields.io/codecov/c/github/alan-turing-institute/sktime?label=codecov&logo=codecov
.. _codecov: https://codecov.io/gh/alan-turing-institute/sktime

.. |readthedocs| image:: https://readthedocs.org/projects/sktime/badge/?version=latest
.. _readthedocs: https://www.sktime.org/en/latest/?badge=latest

.. |twitter| image:: https://img.shields.io/twitter/follow/sktime_toolbox?label=%20Twitter&style=social
.. _twitter: https://twitter.com/sktime_toolbox

.. |python| image:: https://img.shields.io/pypi/pyversions/sktime
.. _python: https://www.python.org/

.. |codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _codestyle: https://github.com/psf/black

.. |contributors| image:: https://img.shields.io/github/contributors/alan-turing-institute/sktime?color=pink&label=all-contributors
.. _contributors: https://github.com/alan-turing-institute/sktime/blob/main/CONTRIBUTORS.md

.. |tutorial| image:: https://img.shields.io/youtube/views/wqQKFu41FIw?label=watch&style=social
.. _tutorial: https://www.youtube.com/watch?v=wqQKFu41FIw&t=14s


Welcome to sktime
=================

sktime is an extensible, composable scikit-learn-like framework toolbox for machine learning and AI with time series.
sktime provides unified interfaces for different time series related learning tasks and compositors for
easy pipelining, tuning, and learning task reduction.
sktime aims to be an interface integrative and community-led nexus, in a friendly and collaborative ecosystem of data science & AI.

The sktime community is open to anyone, with regular events and a `mentorship programme <https://github.com/sktime/mentoring>`_.

.. list-table::
   :header-rows: 0

   * - `Features`_
     - `Vision`_
     - `Installation`_
     - `Quickstart`_
     - `Documentation`_
     - `Get involved`_

.. list-table::
   :header-rows: 0

   * - **CI**
     - |github|_ |appveyor|_ |azure|_ |codecov|_
   * - **Docs**
     - |readthedocs|_ |binder|_ |tutorial|_
   * - **Community**
     - |contributors|_ |gitter|_ |discord|_ |twitter|_
   * - **Code**
     - |pypi|_ |conda|_ |python|_ |codestyle|_ |zenodo|_

Features
--------

sktime currently provides support for the following learning tasks, including pipelining, tuning, and composition:

.. list-table::
   :header-rows: 1

   * - Learning task
     - status
     - tutorial
     - API ref
     - ext.template
     - code
     - roadmap
   * - Forecasting
     - stable
     - `tutorial <https://github.com/alan-turing-institute/sktime/blob/main/examples/01_forecasting.ipynb>`__
     - `API ref <https://www.sktime.org/en/latest/api_reference.html#sktime-forecasting-time-series-forecasting>`__
     - `ext.template <https://github.com/alan-turing-institute/sktime/blob/main/extension_templates/forecasting.py>`__
     - `code <https://github.com/alan-turing-institute/sktime/tree/main/sktime/forecasting>`__
     - `roadmap <https://github.com/alan-turing-institute/sktime/issues/228>`__
   * - Time series annotation
     - experimental
     - n/a
     - n/a
     - n/a
     - `code <https://github.com/alan-turing-institute/sktime/tree/main/sktime/annotation>`__
     - `roadmap <https://github.com/alan-turing-institute/sktime/issues/228>`__
   * - Time series clustering
     - experimental
     - n/a
     - n/a
     - `ext.template <https://github.com/alan-turing-institute/sktime/blob/main/extension_templates/clustering.py>`__
     - `code <https://github.com/alan-turing-institute/sktime/tree/main/sktime/clustering>`__
     - `roadmap <https://github.com/alan-turing-institute/sktime/issues/228>`__
   * - Time series classification
     - stable/refactor
     - `tutorial <https://github.com/alan-turing-institute/sktime/blob/main/examples/02_classification_univariate.ipynb>`__
     - `API ref <https://www.sktime.org/en/latest/api_reference.html#sktime-classification-time-series-classification>`__
     - `ext.template <https://github.com/alan-turing-institute/sktime/blob/main/extension_templates/classification.py>`__
     - `code <https://github.com/alan-turing-institute/sktime/tree/main/sktime/classification>`__
     - `roadmap <https://github.com/alan-turing-institute/sktime/issues/228>`__
   * - Time series regression
     - experimental
     - n/a
     - `API ref <https://www.sktime.org/en/latest/api_reference.html#sktime-classification-time-series-regression>`__
     - n/a
     - `code <https://github.com/alan-turing-institute/sktime/tree/main/sktime/regression>`__
     - `roadmap <https://github.com/alan-turing-institute/sktime/issues/228>`__

sktime further provides unified interfaces for the following time series related objects:

.. list-table::
   :header-rows: 1

   * - Object type
     - status
     - tutorial
     - API ref
     - ext.template
     - code
     - roadmap
   * - transformers, feature extractors
     - stable/refactor
     - n/a
     - `API ref <https://www.sktime.org/en/latest/api_reference.html#sktime-transformations-time-series-transformers>`__
     - n/a
     - `code <https://github.com/alan-turing-institute/sktime/tree/main/sktime/transformations>`__
     - `roadmap <https://github.com/alan-turing-institute/sktime/issues/228>`__
   * - time series distances, kernels
     - experimental
     - n/a
     - n/a
     - n/a
     - `code <https://github.com/alan-turing-institute/sktime/tree/main/sktime/dist_kernels>`__
     - `roadmap <https://github.com/alan-turing-institute/sktime/issues/228>`__

Community contributions to stable and experimental tasks are welcome - especially
implementing estimators on the roadmap, contributing tutorials, documentation, improving the interface.
See below for contributing.

For deep learning strategies, see our companion package: `sktime-dl <https://github.com/sktime/sktime-dl>`_.

Vision
------

* **by the community**, for the community. Developed by a friendly, decentral community - `we welcome all contributors <https://gitter.im/sktime/community>`__.
* **without marketing bias** - sktime does not push "favourite algorithms" on its users, sktime is not a marketing activity.
* the **right tool for the right task** - helping users to diagnose their learning problem and suitable scientific model types
* **embedded in state-of-art OS ecosystems** - interoperable with scikit-learn, statsmodels, pmdarima, tsfresh, and other community favourites.
* **rich composition and reduction functionality** - build tuning and feature extraction pipelines, solve forecasting tasks with scikit-learn regressors, etc
* **clean, descriptive specification syntax** - based on modern object-oriented design principles for data science
* an **integrator of functionality and provisioner of friendly interfaces**, not an "all-bells-and-whistles-one-stop-shop-the-package-that-replaces-everything"
* **fair model assessment and benchmarking** - build your models, inspect your models, check your models, avoid pitfalls.
* **easily extensible** - easy `blueprints to add your own algorithms interface-ready <https://github.com/alan-turing-institute/sktime/tree/main/extension_templates>`__, no tedious hacking of internals required

Installation
------------

The package is available via PyPI using:

.. code-block:: bash

    pip install sktime

Alternatively, you can install it via conda:

.. code-block:: bash

    conda install -c conda-forge sktime

The package is actively being developed and some features may
not be stable yet.

Development version
~~~~~~~~~~~~~~~~~~~

To install the development version, please see our
`advanced installation instructions <https://www.sktime.org/en/latest/installation.html>`__.


Quickstart
----------

Forecasting
~~~~~~~~~~~

.. code-block:: python

    from sktime.datasets import load_airline
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    mean_absolute_percentage_error(y_test, y_pred)
    >>> 0.08661467738190656

For more, check out the `forecasting tutorial <https://github.com/alan-turing-institute/sktime/blob/main/examples/01_forecasting
.ipynb>`__.

Time Series Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.datasets import load_arrow_head
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load_arrow_head(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier = TimeSeriesForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy_score(y_test, y_pred)
    >>> 0.8679245283018868

For more, check out the `time series classification tutorial <https://github.com/alan-turing-institute/sktime/blob/main/examples/02_classification_univariate.ipynb>`__.

Documentation
-------------

* PyData Amsterdam 2020 tutorial: `[video] <https://www.youtube.com/watch?v=Wf2naBHRo8Q>`__, `[notebooks] <https://github.com/sktime/sktime-tutorial-pydata-amsterdam-2020>`__
* `Tutorial notebooks <https://github.com/alan-turing-institute/sktime/tree/main/examples>`__ - you can run them on Binder_ without having to install anything!
* `User guide <https://www.sktime.org/en/latest/user_guide.html>`__
* `API reference <https://www.sktime.org/en/latest/api_reference.html>`__


Get involved
------------

There are many ways to join the sktime community:

* join the developer slack - contact us on gitter or via info@sktime.org
* make a pull request with a code/doc contribution directly, see our `contributing guide <https://www.sktime.org/en/latest/contributing.html>`__
* contribute ideas to the roadmap by making an `enhancement proposal <https://github.com/sktime/enhancement-proposals>`__
* contribute to the `regular community meetings <https://github.com/sktime/community-council>`__
* join one of the onboarding events or developer events (announced on issue tracker, gitter, and twitter)
* sign up for our `community mentoring scheme <https://github.com/sktime/mentoring>`__

For contributions, we follow the `all-contributors specification <https://github.com/alan-turing-institute/sktime/blob/main/CONTRIBUTORS.md>`__ - and all kinds of contributions are welcome!

If you have a question, `chat <https://gitter.im/sktime/community?source=orgpage>`__ with us or `raise an issue <https://github.com/alan-turing-institute/sktime/issues/new/choose>`__. Your help and feedback is extremely welcome!

Development roadmap
-------------------

Read our detailed roadmap `here <https://www.sktime.org/en/latest/roadmap.html>`_.


How to cite sktime
------------------

If you use sktime in a scientific publication, we would appreciate citations to the following paper:

`Markus Löning, Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Jason Lines, Franz Király (2019): “sktime: A Unified Interface for Machine Learning with Time Series” <http://learningsys.org/neurips19/assets/papers/sktime_ml_systems_neurips2019.pdf>`__

Bibtex entry:

.. code-block:: latex

    @inproceedings{sktime,
        author = {L{\"{o}}ning, Markus and Bagnall, Anthony and Ganesh, Sajaysurya and Kazakov, Viktor and Lines, Jason and Kir{\'{a}}ly, Franz J},
        booktitle = {Workshop on Systems for ML at NeurIPS 2019},
        title = {{sktime: A Unified Interface for Machine Learning with Time Series}},
        date = {2019},
    }
