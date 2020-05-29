.. -*- mode: rst -*-

|travis|_ |appveyor|_ |azure|_ |pypi|_ |gitter|_ |binder|_ |zenodo|_

.. |travis| image:: https://img.shields.io/travis/com/alan-turing-institute/sktime/master?logo=travis
.. _travis: https://travis-ci.com/alan-turing-institute/sktime

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

.. |azure| image:: https://img.shields.io/azure-devops/build/mloning/sktime/1/ci?logo=azure-pipelines
.. _azure: https://dev.azure.com/mloning/sktime/_build/latest?definitionId=1&branchName=master


sktime
======

sktime is a `scikit-learn <https://github.com/scikit-learn/scikit-learn>`__ compatible Python toolbox for machine learning with time series. sktime currently supports:

* Time series classification and regression,
* Univariate forecasting.

We provide dedicated time series algorithms and tools for composite model
building, tuning and model evaluation.

We have a number of `extension packages <https://github.com/sktime/>`__. For deep learning, see: `sktime-dl
<https://github.com/sktime/sktime-dl>`_.


Installation
------------

The package is available via PyPI using:

.. code-block:: bash

    pip install sktime

But note that the package is actively being developed and some features may
not yet be stable.

Development version
~~~~~~~~~~~~~~~~~~~

To install the development version, please see our
`advanced installation instructions <https://alan-turing-institute.github.io/sktime/extension.html>`__.


Documentation
-------------

* Check out our `examples notebooks <https://github.com/alan-turing-institute/sktime/tree/master/examples>`__ or run them interactively on Binder_,
* Read the detailed `API reference <https://alan-turing-institute.github.io/sktime/>`__,
* Take a look at our previous `tutorials and sprints <https://github.com/sktime/sktime-workshops>`__.


API Overview
------------

sktime provides a scikit-learn compatible API for multiple distinct, but
closely related time series learning tasks, such as time series
classification, regression and forecasting. For more details on these
problems and how they are related, take a look at our `paper <http://arxiv
.org/abs/1909.07872>`__.

Currently, the package includes dedicated algorithms and tools for composite
model building, tuning and model evaluation:

* State-of-the-art algorithms for time series classification and regression,
ported from the Java-based `tsml <https://github.com/uea-machine-learning/tsml/>`__ toolkit,
* Transformers, including single-series transformation (e.g. detrending or
deseasonalization) and series-as-features transformation (e.g. feature
extractors, as well as tools to compose different transformers
* Pipelining, allowing to chain multiple transformers with a final estimator,
* Tuning using grid-search CV
* Ensembling, such as a fully customisable random forest for time-series
classification and regression, as well as ensembling for multivariate problems,
* Univariate forecasting algorithms (e.g. exponential smoothing)

For a list of implemented methods, see our `estimator overview <https://github.com/alan-turing-institute/sktime/blob/master/ESTIMATOR_OVERVIEW.md>`_.

In addition, sktime includes a experimental high-level API that unifies multiple learning tasks, partially inspired by the APIs of `mlr <https://mlr.mlr-org.com>`__ and `openML <https://www.openml.org>`__.



Development roadmap
-------------------
1. Time series annotation, including segmentation and outlier detection,
2. Multivariate forecasting based on a modified `pysf
<https://github.com/alan-turing-institute/pysf/>`__ API,
3. Unsupervised learning, including time series clustering,
4. Specialised data container for efficient handling of time series/panel data in a modelling workflow and separation of time series meta-data,
5. Probabilistic modelling framework for time series, including survival and point process models based on an adapted `skpro <https://github.com/alan-turing-institute/skpro/>`__ interface.

For more details, see this `issue <https://github.com/alan-turing-institute/sktime/issues/228>`_.

How to contribute
-----------------
We are actively looking for contributors. Please contact @fkiraly or @mloning for volunteering or information on
paid opportunities, or simply `chat <https://gitter.im/sktime/community?source=orgpage>`__ with us
or `raise an issue <https://github.com/alan-turing-institute/sktime/issues/new/choose>`__.

Please also take a look at our `Code of Conduct <https://github.com/alan-turing-institute/sktime/blob/master/CODE_OF_CONDUCT.md>`__ and `guide on how to get started <https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTING.md>`__.

For former and current contributors, see our `overview <https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTORS.md>`_.

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


