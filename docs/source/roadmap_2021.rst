=======
Roadmap
=======

Welcome to sktime's roadmap.

Contributors: :user:`mloning`, :user:`fkiraly`, :user:`sveameyer13`, :user:`lovkush-a`, :user:`bilal-196`, :user:`GuzalBulatova`, :user:`chrisholder`, :user:`satya-pattnaik`, :user:`aiwalter`

Created during the 2021 sktime dev days, 25/06/2021.

----

Project aims
------------
The aim of sktime is to:

* Develop a unified framework for machine learning with time series in Python
* Advance research on algorithm development and software design for machine learning toolboxes
* Build a more connected community of researchers and domain experts who work with time series
* Create and deliver educational material including documentation and user guides

Work streams
------------

Documentation
~~~~~~~~~~~~~
* Core documentation needs to be created "properly"
* Improve tutorials, examples
* Improve extension guidelines
* For research algorithms, possibly pairing up researchers with 'engineer' to improve readability/documentation

Community building
~~~~~~~~~~~~~~~~~~
- Integrate "off-line" contributors
- For research algorithms, possibly pairing up researchers with "engineer" to improve readability/documentation
- Establish regular technical and social meetings

Refactoring and extending existing modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Support for data input types and conversion (e.g. awkward-array)
* Distance metrics
* Reduction interface
* Advanced pipelining
* Forecasting
    * Prediction intervals and probabilistic forecasting
    * Streaming data interface, "update" capability of estimators
    * multivariate/vector forecasting
    * consistent handling of exogeneous variables
    * fitted parameter interface
* Time series classification/regression/clustering
    * add support for unequal length time series
    * add data simulators for algorithm comparison and unit testing
* Clustering
    * interface scikit-learn estimators
    * implement time series specific estimators (e.g. k-shapes)
* Series annotation
    * implement more estimators for outlier anomaly/detection and segmentation

Adding new modules and algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Panel annotation
* Probabilistic interface, event modelling(time-to-event modeling, survival analysis)
* Panel & supervised forecasting
* Time series regression
* Sequence-similarity tasks
* Uniform reduction interface between tasks

Software engineering & dev ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Improve dependency management
* Create template repository for companion packages
* Improve continuous integration & deployment
    - Refactoring unit tests
    - Extending unit tests
    - Speed up unit tests
    - Make unit tests for estimators importable from other packages
