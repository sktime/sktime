.. _roadmap:

Software management plan
========================

Contributors: @mloning, @fkiraly, @TonyBagnall

Contets
-------



Introduction
------------

* Welcome to our software management plan and roadmap

Mission statement
^^^^^^^^^^^^^^^^^

..

   sktime enables understandable and composable machine learning with time series. It provides scikit-learn compatible algorithms and model composition tools, supported by a clear taxonomy of learning tasks, with instructive documentation and a friendly community.


Key themes:


* making the time series analysis ecosystem more understandable, usable and interoperable
* compatibility with other time series libraries and foundational libraries (e.g. scikit-learn)
* providing state-of-the-art time series analysis capabilities
* building a more connected time series analysis community by connecting methodology experts with domain experts who work with time series data

Project output
^^^^^^^^^^^^^^

* unified software framework toolbox for doing machine learning with time series
* documentation, tutorials and other educational materials
* research publications on algorithm development, comparative benchmarking, and ML software design
* workshops, sprints and talks for contributors and domain experts who work with time series

Governance
^^^^^^^^^^

* link to governance document, including Code of Conduct

License
^^^^^^^

* permissive BSD-3-clause

How to cite sktime
^^^^^^^^^^^^^^^^^^

* zenodo
* paper

Dependencies
^^^^^^^^^^^^

close integration with

* Python ecosystem for scientific computing, including Numpy, pandas, scikit-learn and numba among others
* Python ecosystem for time series analysis

Development operations
^^^^^^^^^^^^^^^^^^^^^^

* adopted open-source best practices for research software
* well tested, common unit testing framework
* online continuous integration services
* online code reviews on GitHub
* automatic code quality checks
* automatic release pipeline to compile, package and upload compiled files to PyPI
* distribution of pre-compiled files for different operation systems for user friendly installation

----

Project roadmap
---------------

Overview
^^^^^^^^

* release cycle: bigger release every 3 months, more frequent smaller patches
* @mloning is lead developer and responsible for making releases
* we list general directions that core contributors are interested to see developed in sktime,
* the fact that an item is listed here is in no way a promise that it will happen, as resources are limited, rather, it is an indication that help is welcomed on this topic:

.. list-table::
   :header-rows: 1

   * - Project
     - Coordinators
     - Status
   * - `Documentation <#Documentation>`_
     - @lynnssi, @mloning, @fkiraly
     -
   * - `Development operations <#Development-operations>`_
     - @mloning
     -
   * - `Community building <#Community-building>`_
     - @TonyBagnall, @mloning, @fkiraly
     -
   * - `Framework <#Framework>`_
     - @TonyBagnall, @mloning, @fkiraly
     -
   * - `Time series classification <#Time-series-Classification>`_
     - @TonyBagnall, @mloning
     -
   * - `Time series regression <#Time-series-regression>`_
     - @TonyBagnall, @mloning
     -
   * - `Time series clustering <#Time-series-clustering>`_
     - @TonyBagnall, @fkiraly, @mloning
     -
   * - `Forecasting <#Forecasting>`_
     - @mloning, @fkiraly
     -
   * - `Time series annotation <#Time-series-annotation>`_
     - @mloning, @fkiraly
     -
   * - `Time-to-event modelling <#Time-to-event-modelling>`_
     - @fkiraly, @mloning
     -


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

Ackwnoledgements
----------------


* Open Life Science programme https://openlifesci.org
* Software Sustainability Institute https://www.software.ac.uk/resources/guides/software-management-plans
* Elixir https://elixir-europe.org/events/webinar-software-management-plans
