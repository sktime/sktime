.. -*- mode: rst -*-

|travis|_ |appveyor|_ |pypi|_ |gitter|_ |Binder|_

.. |travis| image:: https://img.shields.io/travis/com/alan-turing-institute/sktime/master?logo=travis
.. _travis: https://travis-ci.com/alan-turing-institute/sktime

.. |appveyor| image:: https://img.shields.io/appveyor/ci/mloning/sktime/master?logo=appveyor
.. _appveyor: https://ci.appveyor.com/project/mloning/sktime

.. |pypi| image:: https://img.shields.io/pypi/v/sktime
.. _pypi: https://pypi.org/project/sktime/

.. |gitter| image:: https://img.shields.io/gitter/room/alan-turing-institute/sktime?logo=gitter
.. _gitter: https://gitter.im/sktime/community

.. |binder| image:: https://mybinder.org/badge_logo.svg
.. _Binder: https://mybinder.org/v2/gh/alan-turing-institute/sktime/master?filepath=examples


sktime
======

sktime is a `scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ compatible Python toolbox for machine
learning with time series. sktime currently supports:

* State-of-the-art time series classification and time series regression algorithms,
* Classical forecasting including reduction strategies,
* Benchmarking and post-hoc evaluation methods based on `mlaut <https://github.com/alan-turing-institute/mlaut/>`__.

sktime has a number of `extension packages <https://github.com/sktime/>`__. For deep learning, see: `sktime-dl
<https://github.com/sktime/sktime-dl>`_.

sktime is under active development and we are looking for contributors.

Installation
------------
The package is available via PyPI using:

:code:`pip install sktime`

But note that the package is actively being developed and currently not feature stable.

Development version
~~~~~~~~~~~~~~~~~~~
To install the development version, please see our
`advanced installation instructions <https://alan-turing-institute.github.io/sktime/extension.html>`__.


Documentation
-------------
* Read the detailed `API reference <https://alan-turing-institute.github.io/sktime/>`__,
* Check out our `examples notebooks <https://github.com/alan-turing-institute/sktime/tree/master/examples>`__ or run them interactively on Binder_,
* Take a look at our previous `tutorials and sprints <https://github.com/sktime/sktime-workshops>`__.


API Overview
------------
sktime extends the standard scikit-learn API to handle modular machine learning workflows for time series data.
The goal is to create a unified interface for various distinct but closely related learning tasks that arise in a temporal data context, such as time series classification and forecasting. To find our more, take a look at our `paper <http://arxiv.org/abs/1909.07872>`__.

Currently, the package implements:

* Various state-of-the-art algorithms for time series classification and regression, ported from the Java-based `tsml <https://github.com/uea-machine-learning/tsml/>`__ toolkit,
* Transformers, including series-to-series transforms (e.g. Fourier transform), series-to-primitives transforms a.k.a. feature extractors (e.g. mean, variance), sub-divided into fittables (on table) and row-wise applicates,
* Pipelining, allowing to chain multiple transformers with a final estimator,
* Meta-estimators such as reduction strategies, grid-search tuners and ensembles, including ensembles for multivariate time series classification,
* Composite strategies, such as a fully customisable random forest for time-series classification, with interval segmentation and feature extraction,
* Classical forecasting algorithms and reduction strategies to solve forecasting tasks with time series regression algorithms.

In addition, sktime includes a experimental high-level API that unifies multiple learning tasks, partially inspired by the APIs of `mlr <https://mlr.mlr-org.com>`__ and `openML <https://www.openml.org>`__.
In particular, we introduce:

* Task objects that encapsulate meta-data from a dataset and the necessary information about the particular learning task, e.g. the instructions on how to derive the target/labels for classification from the data,
* Strategy objects that wrap estimators and allow to call fit and predict methods using data and a task object.


Development road map
--------------------

1. Development of a time series annotation framework, including segmentation and outlier detection,
2. Integration of supervised/panel forecasting based on a modified `pysf <https://github.com/alan-turing-institute/pysf/>`__ API,
3. Unsupervised methods including time series clustering,
4. Design and implementation of a specialised data container for efficient handling of time series/panel data in a modelling workflow and separation of time series meta-data,
5. Development of a probabilistic modelling framework for time series, including survival and point process models based on an adapted `skpro <https://github.com/alan-turing-institute/skpro/>`__ interface.


Contributions
-------------
We are actively looking for contributors. Please contact @fkiraly or @mloning for volunteering or information on
paid opportunities, or simply `chat <https://gitter.im/sktime/community?source=orgpage>`__ with us
or `raise an issue <https://github.com/alan-turing-institute/sktime/issues/new/choose>`__.

Please also take a look at our `Code of Conduct <https://github.com/alan-turing-institute/sktime/blob/master/CODE_OF_CONDUCT.md>`__ and `contributing guidelines <https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTING.md>`__.

Former and current contributors to the API design and project management include:

* API design: Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Franz Király, Jason Lines, Markus Löning
* Project management: Anthony Bagnall, Franz Király, Jason Lines, Markus Löning


How to cite sktime
------------------

If you use sktime in a scientific publication, we would appreciate citations to the following paper:

* `Markus Löning, Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Jason Lines: “sktime: A Unified Interface for Machine Learning with Time Series”, 2019; arXiv:1909.07872 <http://arxiv.org/abs/1909.07872>`__

Bibtex entry::

    @inproceedings{sktime,
        author = {L{\"{o}}ning, Markus and Bagnall, Anthony and Ganesh, Sajaysurya and Kazakov, Viktor and Lines, Jason and Kir{\'{a}}ly, Franz J},
        booktitle = {Workshop on Systems for ML at NeurIPS 2019},
        title = {{sktime: A Unified Interface for Machine Learning with Time Series}},
        date = {2019},
    }

