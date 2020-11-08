.. _roadmap:

Development roadmap
-------------------

Welcome to sktime's development roadmap.

Release cycle
^^^^^^^^^^^^^

sktime is under active development. Given the early stage of development, we
currently do not follow a regular release cycle.

and
we are aiming
at
making a
new release
at
cycle:
bigger release every 3
months,
more
frequent smaller patches
* @mloning is lead developer and responsible for making releases
* we list general directions that core contributors are interested to see developed in sktime,
* the fact that an item is listed here is in no way a promise that it will happen, as resources are limited, rather, it is an indication that help is welcomed on this topic:

Project overview
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Coordinators
     - Status
   * - `Time series classification <#Time-series-Classification>`_
     - @TonyBagnall
     - Stable
   * - `Development operations <#Development-operations>`_
     - @mloning
     - Stable
   * - `Forecasting <#Forecasting>`_
     - @fkiraly, @mloning
     - Maturing
   * - `Documentation <#Documentation>`_
     - @fkiraly, @mloning
     - Maturing
   * - `Community building <#Community-building>`_
     - @fkiraly, @mloning, @TonyBagnall
     - Maturing
   * - `Framework <#Framework>`_
     - @fkiraly, @mloning, @TonyBagnall
     - Maturing
   * - `Time series regression <#Time-series-regression>`_
     - @TonyBagnall
     - Maturing
   * - `Time series annotation <#Time-series-annotation>`_
     - @fkiraly, @mloning
     - Design
   * - `Time-to-event modelling <#Time-to-event-modelling>`_
     - @fkiraly, @mloning
     - Planned
   * - `Time series clustering <#Time-series-clustering>`_
     - @TonyBagnall
     - Planned


Documentation
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Links
     - Description
   * - Website
     -
     - Improve website
   * - User guide
     - #377
     - Develop user guide, taxonomies for tasks and models, write scientific references
   * - Developer guide
     -
     - Improve guide for contributing to sktime and extending existing functionality
   * - Reviewer guide
     -
     - Add a reviewer guide to our contributing guide (see e.g. this `blog post <https://rgommers.github.io/2019/06/the-cost-of-an-open-source-contribution/>`_\ )
   * - Glossary
     - #363
     - Add glossary of common terms
   * - Docstrings
     -
     - Improve docstrings
   * - Blog posts
     -
     - Write blog posts


Development operations
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Links
     - Description
   * - Continuous Integration
     - #196
     - Improve continuous integration and development operations
   * - Unit testing framework
     -
     -


Community building
^^^^^^^^^^^^^^^^^^


* connect methodology experts with domain experts who work with time series data
* cater development more specifically to domain experts who work with time series data
* organize outreach events to grow developer community (e.g. sprints, workshops)
* mentorship programmes to onboard new contributors
* domain-specific user trainings (e.g. medical data training)
* Enhance governance structures for package affiliation, industry involvement, and to ensure inclusive, diverse and sustainable community
* Develop collaboration with existing package developers to work towards a more unified ecosystem

Framework development
^^^^^^^^^^^^^^^^^^^^^


* design and implement standardized interface for different time series learning tasks
* implement tools for reduction approaches, making algorithms for one task applicable to solve another task
* implement re-usable tools for building composite models such as pipelines
* implement tools for model evaluation and comparative benchmarking, including common baseline models
* establish standard data representation for time series, establish common frameworks and APIs for different learning tasks and modelling approaches as standards, integrate existing packages into common framework (data container)
* integrate awkward-array

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^


* develop and implement state-of-the-art time series algorithms

.. list-table::
   :header-rows: 1

   * - Project
     - Links
     - Description
   * - Multivariate data
     -
     - Extend algorithms to handle multivariate data
   * - Parallelization
     - #381
     - Parallelize algorithms using joblib or numba
   * - Unequal length data
     - #230
     - Extend algorithms to handle unequal length data
   * - New algorithms
     - #259
     - Add new algorithms
   * - Data simulators
     - #353
     - Add data simulators for unit testing and algorithm explanation/interpretability


Time series regression
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Links
     - Description
   * - Refactor time series classifiers
     - #212
     - Refactor time series classifiers into time series regressor


Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Project
     - Links
     - Description
   * - 2nd degree transformer framework
     - #52, #105
     - Design and implement 2nd degree transformer framework
   * - New algorithms
     -
     - Add new clustering algorithms based on scikit-learn's implementation and sktime's time series distances
   * - New distances
     -
     - Add new time series distances


Forecasting
^^^^^^^^^^^


* "classical" forecasting of a single series with potential exogenous variables
* vector forecasting
* panel forecasting

.. list-table::
   :header-rows: 1

   * - Project
     - Links
     - Description
   * - Multivariate data
     -
     - Extend algorithms to handle multivariate/exogenous data, add new composition tools for multivariate time series data
   * - New algorithms
     - #220
     - Add new algorithms
   * - Prediction intervals
     -
     - Extend algorithms to compute prediction intervals
   * - Fitted parameter interface
     -
     - Extend algorithms to support fitted parameter interface
   * - Interface algorithms
     -
     - Interface algorithms from existing packages
   * - Data simulators
     - #353
     - Add data simulators for unit testing and algorithm explanation/interpretability
   * - API design of supervised forecasting
     - #66
     - Design supervised forecasting API based on pysf


Time series annotation
^^^^^^^^^^^^^^^^^^^^^^


* anomaly detection
* segmentation

.. list-table::
   :header-rows: 1

   * - Project
     - Links
     - Description
   * - API design
     - #260
     - Design time series annotation API


Time-to-event modelling
^^^^^^^^^^^^^^^^^^^^^^^


* interface to probability distribution APIs
* probabilistic supervised learning
* survival modelling
* point processes
