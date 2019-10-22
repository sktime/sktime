.. -*- mode: rst -*-

|travis|_ |appveyor|_ |pypi|_ |gitter|_

.. |travis| image:: https://img.shields.io/travis/com/alan-turing-institute/sktime/master?logo=travis
.. _travis: https://img.shields.io/travis/com/alan-turing-institute/sktime/master?logo=travis

.. |appveyor| image:: https://img.shields.io/appveyor/ci/mloning/sktime/master?logo=appveyor
.. _appveyor: https://img.shields.io/appveyor/ci/mloning/sktime/master?logo=appveyor

.. |pypi| image:: https://img.shields.io/pypi/v/sktime
.. _pypi: https://img.shields.io/pypi/v/sktime

.. |gitter| image:: https://img.shields.io/gitter/room/alan-turing-institute/sktime?logo=gitter
.. _gitter: https://img.shields.io/gitter/room/alan-turing-institute/sktime?logo=gitter


sktime
======

`scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ compatible Python toolbox for learning with
time series. sktime currently supports:

* State-of-the-art time series classification and regression algorithms,
* Classical forecasting including reduction strategies to regression,
* Benchmarking and post-hoc evaluation methods based on `mlaut <https://github.com/alan-turing-institute/mlaut/>`__.

For deep learning methods, we have a separate extension package:
`sktime-dl <https://github.com/uea-machine-learning/sktime-dl>`__.

sktime is under active development and you can find our development road map below.
Development takes place in the `sktime <https://github.com/alan-turing-institute/sktime>`__ repository on Github.


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
The full API documentation and an introduction can be found `here <https://alan-turing-institute.github.io/sktime/>`__.
Tutorial notebooks for currently stable functionality are in the
`examples <https://github.com/alan-turing-institute/sktime/tree/master/examples>`__ folder.


Overview
--------
sktime extends the standard scikit-learn API to handle modular workflows for time series and panel data.
Currently, the package implements:

* Various state-of-the-art approaches to time series classification and regression (time series as features),
* Transformation of time series, including series-to-series transforms (e.g. Fourier transform), series-to-primitives transforms a.k.a. feature extractors, (e.g. mean, variance), sub-divided into fittables (on table) and row-wise applicates,
* Pipelining, allowing to chain multiple transformers with a final estimator,
* Meta-estimators such as grid-search tuners and ensembles, including ensembles for multivariate time series classification,
* Off-shelf composite strategies, such as a fully customisable random forest for time-series classification, with interval segmentation and feature extraction,
* Classical forecasting algorithms and reduction strategies to solve forecasting tasks with time series regression algorithms.

In addition, sktime includes a high-level API that unifies multiple learning tasks, such as time series classification
and forecasting, partially inspired by the APIs of `mlr <https://mlr.mlr-org.com>`__ and `openML <https://www.openml
.org>`__. In particular, we introduce:

* Task objects that encapsulate meta-data from a dataset and the necessary information about the particular learning task, e.g. the instructions on how to derive the target/labels for classification from the data,
* Strategy objects that wrap estimators and allow to call fit and predict methods using data and a task object.


Development road map
--------------------

1. Development of a time series annotation framework, including segmentation and outlier detection,
2. Integration of supervised forecasting based on a modified `pysf <https://github.com/alan-turing-institute/pysf/>`__ API,
3. Unsupervised methods including time series clustering,
4. Design and implementation of specialised data container for efficient handling of time series/panel data in a modelling workflow and separation of time series meta-data,
5. Development of probabilistic modelling framework, including survival and point process models based on an adapted `skpro <https://github.com/alan-turing-institute/skpro/>`__ interface.


How to cite sktime
------------------

If you use sktime in a scientific publication, we would appreciate citations to the following paper:

* `Markus Löning, Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Jason Lines: “sktime: A Unified Interface for Machine Learning with Time Series”, 2019; arXiv:1909.07872 <http://arxiv.org/abs/1909.07872>`_

Bibtex entry::

    @misc{sktime,
          author = {Markus Löning and Anthony Bagnall and Sajaysurya Ganesh and Viktor Kazakov
          and Jason Lines and Franz J. Király},
          title = {sktime: A Unified Interface for Machine Learning with Time Series},
          year = {2019},
          eprint = {arXiv:1909.07872},
    }



Contributions
-------------
We are actively looking for contributors. Please contact @fkiraly or @jasonlines for volunteering or information on
paid opportunities, or simply `chat <https://gitter.im/sktime/community?source=orgpage>`__ with us
or `raise an issue <https://github.com/alan-turing-institute/sktime/issues/new/choose>`__.

Former and current contributors to the API design and project management include:

* Design: Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Franz Király, Jason Lines, Markus Löning
* Project management: Anthony Bagnall, Franz Király, Jason Lines, Markus Löning

