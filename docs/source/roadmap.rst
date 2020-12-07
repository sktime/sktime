.. _roadmap:

Development roadmap
-------------------

Welcome to sktime's software development plan.

Contents
^^^^^^^^

.. contents:: :local:


Goals
^^^^^

The main goals of sktime are as follows:

1. Develop a unified Python framework for ML with time series
2. Advance research on algorithm development, real-world applications, and ML software design
3. Build a more connected community (unified ecosystem, governance, workshops)
4. Create and deliver educational material (documentation, tutorials, user guide)


Project overview
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Coordinators
     - Status
   * - :ref:`time-series-classification`
     - @TonyBagnall
     - Stable
   * - :ref:`dev-ops`
     - @mloning
     - Stable
   * - :ref:`forecasting`
     - @fkiraly, @mloning
     - Maturing
   * - :ref:`docs`
     - @fkiraly, @mloning
     - Maturing
   * - :ref:`community`
     - @fkiraly, @mloning, @TonyBagnall
     - Maturing
   * - :ref:`framework`
     - @fkiraly, @mloning, @TonyBagnall
     - Maturing
   * - :ref:`time-series-regression`
     - @TonyBagnall
     - Maturing
   * - :ref:`time-series-annotation`
     - @fkiraly, @mloning
     - Design
   * - :ref:`time-to-event-modelling`
     - @fkiraly, @mloning
     - Planned
   * - :ref:`time-series-clustering`
     - @TonyBagnall
     - Planned


.. _docs:

Documentation
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Description
   * - User guide
     - Develop user guide, taxonomies for tasks and models, write scientific references
   * - Developer guide
     - Improve guide for contributing to sktime and extending existing functionality
   * - Docstrings
     - Improve docstrings
   * - Glossary
     - Add glossary of common terms
   * - Educational material
     - Develop educational material (blog posts, videos, etc)
   * - Reviewer guide
     - Write a reviewer guide (see e.g. this `blog post <https://rgommers.github.io/2019/06/the-cost-of-an-open-source-contribution/>`_\ )

.. _dev-ops:

Development operations
^^^^^^^^^^^^^^^^^^^^^^

* Further improve continuous integration and development operations
* Make unit testing framework public so that other packages can check for consistency with sktime's API specification by importing and running basic unit tests

.. _community:

Community building
^^^^^^^^^^^^^^^^^^

* Connect methodology experts with domain experts who work with time series data
* Cater development more specifically to domain experts who work with time series data
* Organize outreach events to grow developer community (e.g. sprints, workshops)
* Mentorship programmes to onboard new contributors
* Domain-specific user trainings (e.g. medical data training)
* Enhance governance structures for package affiliation, industry involvement, and to ensure inclusive, diverse and sustainable community
* Develop collaboration with existing package developers to work towards a more unified ecosystem


.. _framework:

Framework
^^^^^^^^^

sktime develops a unified framework toolbox for machine learning with time series. This requires to design and implement:

* Standardized interface for different time series learning tasks;
* Reduction approaches between learning tasks, allowing algorithms for one task to be applied to another task;
* Tools for model composition (pipelines, tuning, etc.);
* Tools for model evaluation and comparative benchmarking;
* Standardized data representation for time series;

.. _time-series-classification:

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Description
   * - Multivariate data
     - Extend algorithms to handle multivariate data
   * - Parallelization
     - Parallelize algorithms using joblib or numba
   * - Unequal length data
     - Extend algorithms to handle unequal length data
   * - New algorithms
     - Add new algorithms
   * - Data simulators
     - Add data simulators for unit testing and algorithm explanation/interpretability

.. _time-series-regression:

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Description
   * - Refactor time series classifiers
     - Refactor time series classifiers into time series regressor

.. _time-series-clustering:

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Description
   * - 2nd degree transformer framework
     - Design and implement 2nd degree transformer framework
   * - New algorithms
     - Add new clustering algorithms based on scikit-learn's implementation and sktime's time series distances
   * - New distances
     - Add new time series distances

.. _forecasting:

Forecasting
^^^^^^^^^^^

The term forecasting is often used for different learning tasks. We currently support classical forecasting of a single series with potential exogenous variables.

Other common tasks are:

* Vector forecasting
* Supervised forecasting
* Panel forecasting

.. list-table::
   :header-rows: 1

   * - Project
     - Description
   * - Multivariate data
     - Extend algorithms to handle multivariate/exogenous data, add new composition tools for multivariate time series data
   * - New algorithms
     - Add new algorithms
   * - Prediction intervals
     - Extend algorithms to compute prediction intervals
   * - Fitted parameter interface
     - Extend algorithms to support fitted parameter interface
   * - Interface algorithms
     - Interface algorithms from existing packages
   * - Data simulators
     - Add data simulators for unit testing and algorithm explanation/interpretability


.. _time-series-annotation:

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

Common time series annotation tasks are:

* Anomaly detection
* Segmentation

.. _time-to-event-modelling:

Time-to-event modelling
^^^^^^^^^^^^^^^^^^^^^^^

* Interface to probability distribution APIs
* Probabilistic supervised learning
* Survival modelling
* Point processes modelling
