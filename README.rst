.. -*- mode: rst -*-

|travis|_ |appveyor|_ |pypi|_ |gitter|_

.. |travis| image:: https://travis-ci.com/alan-turing-institute/sktime.svg?token=kTo6WTfr4f458q1WzPCH&branch=master
.. _travis: https://travis-ci.com/alan-turing-institute/sktime

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/3bs0rpe1id1ots8c/branch/master?svg=true
.. _appveyor: https://ci.appveyor.com/project/mloning/sktime

.. |pypi| image:: https://badge.fury.io/py/sktime.svg
.. _pypi: https://badge.fury.io/py/sktime

.. |gitter| image:: https://badges.gitter.im/sktime/community.svg
.. _gitter: https://gitter.im/sktime/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge

sktime
======

A `scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ compatible Python toolbox for learning with
time series and panel data. Eventually, we would like to support:

* Time series classification and regression,
* Classical forecasting,
* Supervised/panel forecasting,
* Time series segmentation,
* Time-to-event and event risk modelling,
* Unsupervised tasks such as motif discovery, anomaly detection and diagnostic visualization,
* On-line and streaming tasks, e.g. in variation of the above.

For deep learning methods, we have a separate extension package: `sktime-dl <https://github.com/uea-machine-learning/sktime-dl>`_.

The package is under active development. Development takes place in the `sktime <https://github.com/alan-turing-institute/sktime>`__ repository on Github.

Currently, modular modelling workflows for forecasting and supervised learning with time series have been implemented.
As next steps, we will move to supervised forecasting and integration of a modified `pysf <https://github.com/alan-turing-institute/pysf>`__ interface and extensions to the existing frameworks.

Installation
------------
The package is available via PyPI using:

:code:`pip install sktime`

But note that the package is actively being developed and currently not feature stable.

Development version
~~~~~~~~~~~~~~~~~~~
To install the development version, follow these steps:

1. Download the repository: :code:`git clone https://github.com/alan-turing-institute/sktime.git`
2. Move into the root directory of the repository: :code:`cd sktime`
3. Switch to development branch: :code:`git checkout dev`
4. Make sure your local version is up-to-date: :code:`git pull`
5. Install package: :code:`pip install .`

You currently may have to install :code:`numpy` and :code:`Cython` first using: :code:`pip install numpy`
and :code:`pip install Cython`.

Overview
--------

Low-level interface
~~~~~~~~~~~~~~~~~~~
The low-level interface extends the standard scikit-learn API to handle time series and panel data.
Currently, the package implements:

* Various state-of-the-art approaches to supervised learning with time series features,
* Transformation of time series, including series-to-series transforms (e.g. Fourier transform), series-to-primitives transforms aka feature extractors, (e.g. mean, variance), sub-divided into fittables (on table) and row-wise applicates,
* Pipelining, allowing to chain multiple transformers with a final estimator,
* Meta-learning strategies including tuning and ensembling, accepting pipelines as the base estimator,
* Off-shelf composite strategies, such as a fully customisable random forest for time-series classification, with interval segmentation and feature extraction,
* Classical forecasting algorithms and reduction strategies to solve forecasting tasks with time series regression algorithms.

High-level interface
~~~~~~~~~~~~~~~~~~~~
There are numerous different time series data related learning tasks, for example

* Time series classification and regression,
* Classical forecasting,
* Supervised/panel forecasting,
* Time series segmentation.

The sktime high-level interface aims to create a unified interface for these different learning tasks (partially inspired by the APIs of `mlr <https://mlr.mlr-org.com>`__ and `openML <https://openml.org>`__) through the following two objects:

* :code:`Task` object that encapsulates meta-data from a dataset and the necessary information about the particular supervised learning task, e.g. the instructions on how to derive the target/labels for classification from the data,
* :code:`Strategy` objects that wrap low-level estimators and allows to use :code:`fit` and :code:`predict` methods using data and a task object.


Documentation
-------------
The full API documentation and an introduction can be found `here <https://alan-turing-institute.github.io/sktime/>`__.
Tutorial notebooks for currently stable functionality are in the `examples <https://github.com/alan-turing-institute/sktime/tree/master/examples>`__ folder.


Development road map
--------------------
1. Functionality for the advanced time series tasks. For (supervised) forecasting, integration of a modified `pysf <https://github.com/alan-turing-institute/pysf/>`__ interface. For time-to-event and event risk modell, integration of an adapted `pysf <https://github.com/alan-turing-institute/pysf/>`__ interface.
2. Extension of high-level interface to classical and supervised/panel forecasting, to include reduction strategies in which forecasting or supervised forecasting tasks are reduced to tasks that can be solved with classical supervised learning algorithms or time series classification/regression,
3. Integration of algorithms for classical forecasting (e.g. ARIMA), deep learning strategies, and third-party feature extraction tools,
4. Design and implementation of specialised data-container for efficient handling of time series/panel data in a supervised learning workflow and separation of time series meta-data, re-utilising existing data-containers whenever possible,
5. Automated benchmarking functionality including orchestration of experiments and post-hoc evaluation methods, based on the `mlaut <https://github.com/alan-turing-institute/mlaut/>`__ design.


Contributors
------------
Former and current active contributors are as follows.

Project management: Jason Lines (@jasonlines), Franz Király (@fkiraly)

Design: Anthony Bagnall (@TonyBagnall), Sajaysurya Ganesh (@sajaysurya), Jason Lines (@jasonlines), Viktor Kazakov (@viktorkaz), Franz Király (@fkiraly), Markus Löning (@mloning)

Coding: Sajaysurya Ganesh (@sajaysurya), Anthony Bagnall (@TonyBagnall), Jason Lines (@jasonlines), George Oastler (@goastler), Viktor Kazakov (@viktorkaz), Markus Löning (@mloning)

We are actively looking for contributors. Please contact @fkiraly or @jasonlines for volunteering or information on paid opportunities, or simply raise an issue in the tracker.
