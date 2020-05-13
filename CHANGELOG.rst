Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and we adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

We keep track of changes in this file since v0.4.0.


[Unreleased]
------------
Added
~~~~~
-

Changed
~~~~~~~
-

Removed
~~~~~~~
-

Fixed
~~~~~
-

Deprecated
~~~~~~~~~~
-


[0.4.0] - 2019-05-xx
--------------------

Added
~~~~~
- Forecasting framework, including encapsulated algorithms (forecasters),
  composite model building functionality (meta-forecasters), and tools
  for tuning and model evaluation
- Consistent unit tests
- Consistent input checks
- Enforced PEP8 linting via flake8
- Changelog
- Support for Python 3.8
- Support for manylinux wheels

Changed
~~~~~~~
- Revised all estimators to comply with common interface and to ensure
  scikit-learn compatibility

Removed
~~~~~~~
- A few redundant classes (e.g. `Pipeline`) in favour of scikit-learn's
  implementations


Fixed
~~~~~
- Deprecation and future warnings from scikit-learn
- User warnings from statsmodels
