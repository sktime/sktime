.. image:: https://travis-ci.com/alan-turing-institute/sktime.svg?token=kTo6WTfr4f458q1WzPCH&branch=dev
    :target: https://travis-ci.com/alan-turing-institute/sktime
    
sktime
======

A `scikit-learn <https://github.com/scikit-learn/scikit-learn>`_ compatible Python toolbox for supervised learning with
time-series/panel data.


The package is under active development. Development takes place in the `sktime <https://github.com/alan-turing-institute/sktime>`_ repository on Github.

Currently, various approaches to time-series classification have been implemented.



Installation
------------
The package is currently not available directly via PyPI. Instead, follow these steps
to install the development version:

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
There are numerous time-series/panel data related supervised learning tasks, including

* Time-series classification/regression,
* Classical forecasting,
* Supervised/panel forecasting,
* Time-to-event/event risk prediction.

The high-level interface creates a unified interface for these different time-series/panel data related tasks through the following two objects:

* :code:`Task` object that encapsulates meta-data from a dataset and the necessary information about the particular supervised learning task, e.g. the instructions on how to derive the target/labels for classification from the data,
* :code:`Strategy` objects that wrap low-level estimators and allows to use :code:`fit` and :code:`predict` methods using data and a task object.



Low-level interface
~~~~~~~~~~~~~~~~~~~
The low-level interface extends the standard scikit-learn API to handle time-series/panel data.
Currently, the package implements various state-of-the-art approaches to time-series classification, including

* Random interval segmentation,
* Time-series feature extraction, including series-to-series transforms (e.g. Fourier transform), series-to-primitives transforms (e.g. mean) as well as shapelets,
* Pipelining, allowing to chain multiple transformers with a final classifiers,
* Ensembling, including fully customisable random forest for time-series classification, accepting pipelines as the base estimator, including pipelines with interval segmentation and feature extraction.


Documentation
-------------
The full API documentation can be found `here <https://alan-turing-institute.github.io/sktime/>`_.


Development road map
--------------------
1. Extension of high-level interface to classical and supervised/panel forecasting, including reduction strategies in which time-series/panel data prediction tasks are reduced to tasks that can be solved with classical supervised learning algorithms,
2. Integration of algorithms for classical forecasting (e.g. ARIMA), deep learning, and third-party feature extraction tools,
3. Design and implementation of specialised data-container for efficient handling of time-series/panel data in a supervised learning workflow and separation of time-series meta-data, re-utilising existing data-containers wherever possible,
4. Implementation of automated benchmarking functionality including orchestration of experiments and post-hoc evaluation methods.