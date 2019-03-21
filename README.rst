.. image:: https://travis-ci.com/alan-turing-institute/sktime.svg?token=kTo6WTfr4f458q1WzPCH&branch=master
    :target: https://travis-ci.com/alan-turing-institute/sktime   
sktime
======

A `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ compatible Python toolbox for learning with
time-series/panel data. Eventually, we would like to support:

* Time-series classification and regression,
* Classical forecasting,
* Supervised/panel forecasting,
* time series segmentation
* time-to-event and event risk modelling

The package is under active development. Development takes place in the `sktime <https://github.com/alan-turing-institute/sktime>`_ repository on Github.

Currently, modular modelling workflows for supervised learning with time series have been implemented.
As next steps, we will move to forecasting and integration of a modified `pysf <https://github.com/alan-turing-institute/pysf/>`_ interface for forecasting and supervised forecasting.


Installation
------------
The package is currently not feature stable, and thus not available directly via PyPI. In the interim, please follow these steps to install the development version:

1. Download the repository if you have not done so already: :code:`git clone https://github.com/alan-turing-institute/sktime.git`
2. Move into the root directory: :code:`cd sktime`
3. Make sure your local version is up-to-date: :code:`git pull`
4. Switch onto the development branch: :code:`git checkout dev`
5. Optionally, activate destination environment for package, e.g. with conda: :code:`conda activate <env>`
6. Install package: :code:`pip install .`


Overview
--------

High-level interface
~~~~~~~~~~~~~~~~~~~~
There are numerous differenc time series data related learning tasks, including

* Time-series classification and regression,
* Classical forecasting,
* Supervised/panel forecasting,
* time series segmentation
* time-to-event and event risk modelling

The sktime high-level interface aims to create a unified interface for these different learning tasks through the following two objects:

* :code:`Task` object that encapsulates meta-data from a dataset and the necessary information about the particular supervised learning task, e.g. the instructions on how to derive the target/labels for classification from the data,
* :code:`Strategy` objects that wrap low-level estimators and allows to use :code:`fit` and :code:`predict` methods using data and a task object.



Low-level interface
~~~~~~~~~~~~~~~~~~~
The low-level interface extends the standard scikit-learn API to handle time series and panel data.
Currently, the package implements and interfaces various state-of-the-art approaches to supervised learning with time series features, including

* Random interval segmentation,
* Time series feature extraction, including series-to-series transforms (e.g. Fourier transform), series-to-primitives transforms (e.g. mean, variance) as well as shapelets,
* Pipelining, allowing to chain multiple transformers with a final classifiers,
* Ensembling, such as to create a fully customisable random forest for time-series classification, accepting pipelines as the base estimator, including pipelines with interval segmentation and feature extraction.


Documentation
-------------
The full API documentation and an introduction can be found `here <https://alan-turing-institute.github.io/sktime/>`_.
Tutorial notebooks for currently stable functionality are `here <https://github.com/alan-turing-institute/sktime/tree/master/examples>`_


Development road map
--------------------
1. Functionality for the advanced time series tasks. For (supervised) forecasting, integration of a modified `pysf <https://github.com/alan-turing-institute/pysf/>`_ interface. For time-to-event and event risk modell, integration of an adapted `pysf <https://github.com/alan-turing-institute/skpro/>`_ interface.
2. Extension of high-level interface to classical and supervised/panel forecasting, to include reduction strategies in which forecasting or supervised forecasting tasks are reduced to tasks that can be solved with classical supervised learning algorithms or time series classification/regression,
3. Integration of algorithms for classical forecasting (e.g. ARIMA), deep learning strategies, and third-party feature extraction tools,
4. Design and implementation of specialised data-container for efficient handling of time-series/panel data in a supervised learning workflow and separation of time-series meta-data, re-utilising existing data-containers wherever possible,
5. Automated benchmarking functionality including orchestration of experiments and post-hoc evaluation methods, based on the `mlaut <https://github.com/alan-turing-institute/pysf/>`_ design.

Contributors
------------
Former and current active contributors are as follows.
Project management: Jason Lines (@jasonlines), Franz J Kiraly (@fkiraly)
Design: Anthony Bagnall, Sajaysurya Ganesh (@sajaysurya), Jason Lines (@jasonlines), Viktor Kazakov (@viktorkaz), Franz J Kiraly (@fkiraly), Markus Löning (@mloning)
Coding: Sajaysurya Ganesh (@sajaysurya), Jason Lines (@jasonlines), Viktor Kazakov (@viktorkaz), Markus Löning (@mloning)

We are actively looking for contributors. Please contact @fkiraly or @jasonlines for volunteering or information on paid opportunities, or simply raise an issue in the tracker.
