.. -*- mode: rst -*-

|github|_ |appveyor|_ |azure|_ |codecov|_ |readthedocs|_ |pypi|_ |gitter|_ |binder|_ |zenodo|_ |twitter|_

.. |github| image:: https://img.shields.io/github/workflow/status/alan-turing-institute/sktime/build-and-test?logo=github
.. _github: https://github.com/alan-turing-institute/sktime/actions?query=workflow%3Abuild-and-test

.. |appveyor| image:: https://img.shields.io/appveyor/ci/mloning/sktime/master?logo=appveyor
.. _appveyor: https://ci.appveyor.com/project/mloning/sktime

.. |pypi| image:: https://img.shields.io/pypi/v/sktime
.. _pypi: https://pypi.org/project/sktime/

.. |gitter| image:: https://img.shields.io/gitter/room/alan-turing-institute/sktime?logo=gitter
.. _gitter: https://gitter.im/sktime/community

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _binder: https://mybinder.org/v2/gh/alan-turing-institute/sktime/master?filepath=examples

.. |zenodo| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3749000.svg
.. _zenodo: https://doi.org/10.5281/zenodo.3749000

.. |azure| image:: https://img.shields.io/azure-devops/build/mloning/30e41314-4c72-4751-9ffb-f7e8584fc7bd/1/master?logo=azure-pipelines
.. _azure: https://dev.azure.com/mloning/sktime/_build

.. |codecov| image:: https://img.shields.io/codecov/c/github/alan-turing-institute/sktime?logo=Codecov
.. _codecov: https://codecov.io/gh/alan-turing-institute/sktime

.. |readthedocs| image:: https://readthedocs.org/projects/sktime/badge/?version=latest
.. _readthedocs: https://www.sktime.org/en/latest/?badge=latest

.. |twitter| image:: https://img.shields.io/twitter/follow/sktime_toolbox?label=%20Twitter&style=social
.. _twitter: https://twitter.com/sktime_toolbox


sktime
======

sktime is a Python machine learning toolbox for time series with a unified interface for multiple learning tasks. We currently support:

* Forecasting,
* Time series classification,
* Time series regression.

sktime provides dedicated time series algorithms and `scikit-learn
<https://github.com/scikit-learn/scikit-learn>`__ compatible tools
for building, tuning, and evaluating composite models.

For deep learning methods, see our companion package: `sktime-dl <https://github.com/sktime/sktime-dl>`_.

------------------------------------------------------------

Installation
------------

The package is available via PyPI using:

.. code-block:: bash

    pip install sktime

The package is actively being developed and some features may
not be stable yet.

Development Version
~~~~~~~~~~~~~~~~~~~

To install the development version, please see our
`advanced installation instructions <https://www.sktime.org/en/latest/installation.html>`__.

------------------------------------------------------------

Quickstart
----------

Forecasting
~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from sktime.datasets import load_airline
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import smape_loss

    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
    forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    smape_loss(y_test, y_pred)
    >>> 0.1722386848882188

For more, check out the `forecasting tutorial <https://github
.com/alan-turing-institute/sktime/blob/master/examples/01_forecasting
.ipynb>`__.

Time Series Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sktime.datasets import load_arrow_head
    from sktime.classification.compose import TimeSeriesForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load_arrow_head(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier = TimeSeriesForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy_score(y_test, y_pred)
    >>> 0.7924528301886793

For more, check out the `time series classification tutorial <https://github
.com/alan-turing-institute/sktime/blob/master/examples
/02_classification_univariate.ipynb>`__.

------------------------------------------------------------

Documentation
-------------

* Watch our online tutorial on Machine Learning with Time Series at the PyData Amsterdam 2020: `[video] <https://www.youtube.com/watch?v=Wf2naBHRo8Q>`__, `[repo] <https://github.com/sktime/sktime-tutorial-pydata-amsterdam-2020>`__
* Check out our `example notebooks <https://github.com/alan-turing-institute/sktime/tree/master/examples>`__ - you can run them on Binder_ without having to install anything!
* Read our detailed `API reference <https://www.sktime.org>`__.

------------------------------------------------------------

API Overview
------------

sktime is a unified toolbox for machine learning with time series. Time
series give rise to multiple learning tasks (e.g.
forecasting and time series classification). The goal of sktime is to
provide all the necessary tools to solve these tasks, including dedicated time
series algorithms as well as tools for building, tuning and evaluating
composite models.

Many of these tasks are related. An algorithm that can
solve one of them can often be re-used to help solve another one, an idea
called reduction. sktime's unified interface allows to easily adapt an
algorithm for one task to another.

For example, to use a regression algorithm to solve a forecasting task, we
can simply write:

.. code-block:: python

    import numpy as np
    from sktime.datasets import load_airline
    from sktime.forecasting.compose import ReducedRegressionForecaster
    from sklearn.ensemble import RandomForestRegressor
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import smape_loss

    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
    regressor = RandomForestRegressor()
    forecaster = ReducedRegressionForecaster(regressor, window_length=12)
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    smape_loss(y_test, y_pred)
    >>> 0.12726230426056875

For more details, check out our `paper
<http://learningsys.org/neurips19/assets/papers/sktime_ml_systems_neurips2019.pdf>`__.

Currently, sktime provides:

* State-of-the-art algorithms for time series classification and regression, ported from the Java-based `tsml <https://github.com/uea-machine-learning/tsml/>`__ toolkit, as well as forecasting,
* Transformers, including single-series transformations (e.g. detrending or deseasonalization) and series-as-features transformations (e.g. feature extractors), as well as tools to compose different transformers,
* Pipelining,
* Tuning,
* Ensembling, such as a fully customisable random forest for time-series classification and regression, as well as ensembling for multivariate problems,

For a list of implemented methods, see our `estimator overview <https://github.com/alan-turing-institute/sktime/blob/master/ESTIMATOR_OVERVIEW.md>`_.

In addition, sktime includes an experimental high-level API that unifies multiple learning tasks, partially inspired by the APIs of `mlr <https://mlr.mlr-org.com>`__ and `openML <https://www.openml.org>`__.


------------------------------------------------------------

Development Roadmap
-------------------
sktime is under active development. We're looking for new contributors, all
contributions are welcome!

1. Multivariate/panel forecasting based on a modified `pysf <https://github.com/alan-turing-institute/pysf/>`__ API,
2. Unsupervised learning, including time series clustering,
3. Time series annotation, including segmentation and outlier detection,
4. Specialised data container for efficient handling of time series/panel data in a modelling workflow and separation of time series meta-data,
5. Probabilistic modelling framework for time series, including survival and point process models based on an adapted `skpro <https://github.com/alan-turing-institute/skpro/>`__ interface.

For more details, read this `issue <https://github.com/alan-turing-institute/sktime/issues/228>`_.

------------------------------------------------------------

How to contribute
-----------------
* First check out our `guide on how to contribute <https://www.sktime.org/en/latest/contributing.html>`__.
* `Chat <https://gitter.im/sktime/community?source=orgpage>`__ with us or `raise an issue <https://github.com/alan-turing-institute/sktime/issues/new/choose>`__ if you get stuck or have questions.
* Please also read our `Code of Conduct <https://github.com/alan-turing-institute/sktime/blob/master/CODE_OF_CONDUCT.rst>`__ and `Governance <https://www.sktime.org/en/latest/governance.html>`__ document.

For former and current contributors, see our `overview <https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTORS.md>`_.

------------------------------------------------------------

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
