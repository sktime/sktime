.. _testing_framework:

``Sktime`` testing framework overview
=====================================

``sktime`` uses ``pytest`` for testing interface compliance of estimators, and correctness of code.
This page gives an overview over the tests, and introductions on how to add tests, or how to extend the testing framework.

.. contents::
   :local:

Test module architecture
------------------------

``sktime`` testing happens on three layers, roughly corresponding to the inheritance layers of estimators.

* testing interface compliance with the ``BaseObject`` and ``BaseEstimator`` specifications, in
``tests/test_all_estimators.py``
* testing interface compliance of concrete estimators with their scitype base class, for instance
``forecasting/tests/test_all_forecasters.py``
* testing individual functionality of estimators or other code, in individual files in ``tests`` folders

Module conventions are as follows:

* each module contains a ``tests`` folder, which contains tests specific to that module.
Sub-modules may also contain ``tests`` folders.
* ``tests`` folders may contain ``_config.py`` files to collect test configuration settings for that module
* generic utilities for tests are located in the module ``utils._testing``.
 Tests for these utilities should be contained in the ``utils._testing.tests`` folder.
* each test module corresponding to a learning task and estimator scitype should
 contain a test ``test_all_[name_of_scitype].py`` file that tests interface compliance of all estimators adhering to the scitype.
 For instance, ``forecasting/tests/test_all_forecasters.py``, or ``distances/tests/test_all_dist_kernels.py``.
* Learning task specific tests should not duplicate generic estimator tests in ``test_all_estimators.py``

Test code architecture
----------------------

.. _pytestuse: https://docs.pytest.org/en/6.2.x/example/index.html

``sktime`` test files should use best ``pytest`` practice such as fixtures or test parameterization where possible,
instead of custom logic, see `pytest documentation <pytestuse>`_.



.. code-block::

    from deprecated.sphinx import deprecated

    @deprecated(version="0.8.0", reason="my_old_function will be removed in v0.10.0", category=FutureWarning)
    def my_old_function(x, y):
        return x + y

