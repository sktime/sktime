.. _developer_guide_add_estimators:

=======================
Implementing Estimators
=======================

This page describes how to implement ``sktime`` compatible estimators, and how to ensure and test compatibility.
There are additional steps for estimators that are contributed to ``sktime`` directly.


Implementing an ``sktime`` compatible estimator
===============================================

The high-level steps to implement ``sktime`` compatible estimators are as follows:

1.  identify the type of the estimator: forecaster, classifier, etc
2.  copy the extension template for that kind of estimator to its intended location
3.  complete the extension template
4.  run the ``sktime`` test suite and/or the ``check_estimator`` utility (see `here <https://www.sktime.net/en/latest/developer_guide/add_estimators.html#using-the-check-estimator-utility>`__)
5.  if the test suite highlights bugs or issues, fix them and go to 4

For more guidance on how to implement your own estimator, see this `tutorial at pydata <https://github.com/sktime/sktime-workshop-pydata-london-2022>`__ on testing interface conformance.


What is my learning task?
-------------------------

``sktime`` is structured along modules encompassing specific learning tasks,
e.g., forecasting or time series classification.
For brevity, we define an estimator's scientific type or "scitype" by the formal learning task that it solves.
For example, the scitype of an estimator that solves the forecasting task is "forecaster".
The scitype of an estimator that solves the time series classification task is "time series classifier".

Estimators for a given scitype should be located in the respective module.
The estimator scitypes also map onto the different extension templates found in
the `extension_templates <https://github.com/sktime/sktime/tree/main/extension_templates>`__
directory of ``sktime``.

Usually, the scitype of a given estimator is directly determined by what the estimator does.
This is also, often, explicitly signposted in publications related to the estimator.
For instance, most textbooks mention ARIMA in the context of forecasting, so in that hypothetical situation
it makes sense to consider the "forecaster" template.
Then, inspect the template and check whether the methods of the class map clearly onto routines of the estimator.
If not, another template might be more appropriate.

The most common point of confusion here is between transformers and other estimator types,
since transformers are often used as parts of algorithms of other type.

If unsure, feel free to post your question on one of ``sktime``'s social channels.
Don't panic - it is not uncommon that academic publications are not clear about the type of an estimator,
and correct categorization may be difficult even to experts.


What are ``sktime`` extension templates?
----------------------------------------

Extension templates are convenient "fill-in" templates for implementers of new estimators.
They fit into ``sktime``'s unified interface as follows:

*   for each scitype, there is a public user interface, defined by the respective base class.
    For instance, ``BaseForecaster`` defines the ``fit`` and ``predict`` interfaces for forecasters.
    All forecasters will implement ``fit`` and ``predict`` the same way, by inheritance from ``BaseForecaster``.
    The public interface follows the "strategy" object orientation pattern.
*   for each scitype, there is a private extender interface, defined by the extension contract in the extension template.
    For instance, the ``forecaster.py`` extension template for forecasters explains what to fill in for a concrete forecaster
    inheriting from ``BaseForecaster``. In most extension templates, users should implement private methods ("inner" methods),
    e.g., ``_fit`` and ``_predict`` for forecasters. Boilerplate code rests within the public part of the interface, in ``fit`` and ``predict``.
    The extender interface follows the "template" object orientation pattern.

Extenders familiar with ``scikit-learn`` extension should note the following difference to ``scikit-learn``:

the public interface, e.g., ``fit`` and ``predict``, is never overridden in ``sktime`` (concrete) estimators.
Implementation happens in the private, extender sided interface, e.g., ``_fit`` and ``_predict``.

This allows to avoid boilerplate replication, such as ``check_X`` etc in ``scikit-learn``.
This also allows richer boilerplate, such as automated vectorization functionality or input conversion.


How to use ``sktime`` extension templates
-----------------------------------------

To use the ``sktime`` extension templates, copy them to the intended location of the estimator.
Inside the extension templates, necessary actions are marked with ``todo``.
The typical workflow goes through the extension template by searching for ``todo``, and carrying out
the action described next to the ``todo``.

Extension templates typically have the following ``todo``:

*   choosing name and parameters for the estimator
*   filling in the ``__init__``: writing parameters to ``self``, calling ``super``'s ``__init__``
*   filling in docstrings of the module and the estimator. This is recommended as early as parameters have been settled on,
    it tends to be useful as a specification to follow in implementation.
*   filling in the tags for the estimator. Some tags are "capabilities", i.e., what the estimator can do, e.g., dealing with nans.
    Other tags determine the format of inputs seen in the "inner" methods ``_fit`` etc, these tags are usually called ``X_inner_mtype`` or similar.
    This is useful in case the inner functionality assumes ``numpy.ndarray``, or ``pandas.DataFrame``, and helps avoid conversion boilerplate.
    The type strings can be found in ``datatypes.MTYPE_REGISTER``. For a tutorial on data type conventions, see ``examples/AA_datatypes_and_datasets``.
*   Filling in the "inner" methods, e.g., ``_fit`` and ``_predict``. The docstrings and comments in the extension template should be followed here.
    The docstrings also describe the guarantees on the inputs to the "inner" methods, which are typically stronger than the guarantees on
    inputs to the public methods, and determined by values of tags that have been set.
    For instance, setting the tag ``y_inner_mtype`` to ``pd.DataFrame`` for a forecaster guarantees that the ``y`` seen by ``_fit`` will be
    a ``pandas.DataFrame``, complying with additional data container specifications in ``sktime`` (e.g., index types).
*   filling in testing parameters in ``get_test_params``. The selection of parameters should cover major estimator internal case distinctions
    to achieve good coverage.

Some common caveats, also described in extension template text:

*   ``__init__`` parameters should be written to ``self`` and never be changed
*   special case of this: estimator components, i.e., parameters that are estimators, should generally be
    cloned (via ``sklearn.clone``), and method should be called only on the clones
*   methods should generally avoid side effects on arguments
*   non-state changing methods should not write to ``self`` in general
*   typically, implementing ``get_params`` and ``set_params`` is not needed, since ``sktime``'s ``BaseEstimator`` inherits from ``sklearn``'s.
    Custom ``get_params``, ``set_params`` are typically needed only for complex cases only heterogeneous composites, e.g., pipelines with
    parameters that are nested structures containing estimators.


How to test interface conformance
---------------------------------

For a video tutorial and more examples on the below, please visit our
`tutorial at pydata <https://github.com/sktime/sktime-workshop-pydata-london-2022>`__.

Using the ``check_estimator`` utility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Usually, the simplest way to test interface conformance with ``sktime`` is via the
``check_estimator`` methods in the ``utils.estimator_checks`` module.

When invoked, this will collect tests in ``sktime`` relevant for the estimator type and
run them on the estimator.

This can be used for manual debugging in a notebook environment.
Example of running the full test suite for ``NaiveForecaster``:

.. code-block:: python

    from sktime.utils.estimator_checks import check_estimator
    from sktime.forecasting.naive import NaiveForecaster
    check_estimator(NaiveForecaster)

The ``check_estimator`` utility will return, by default, a ``dict``, indexed by test/fixture combination strings,
that is, a test name and the fixture combination string in squared brackets.
Example: ``'test_repr[NaiveForecaster-2]'``, where ``test_repr`` is the test name, and ``NaiveForecaster-2`` the fixture combination string.

Values of the return ``dict`` are either the string ``"PASSED"``, if the test succeeds, or the exception that the test would raise at failure.
``check_estimator`` does not raise exceptions by default, the default is returning them as dictionary values.
To raise the exceptions instead, e.g., for debugging, use the argument ``raise_exceptions=True``,
which will raise the exceptions instead of returning them as dictionary values.
In that case, there will be at most one exception raised, namely the first exception encountered in the test execution order.

To run or exclude certain tests, use the ``tests_to_run`` or ``tests_to_exclude`` arguments.
Values provided should be names of tests (str), or a list of names of tests.
Note that test names exclude the part in squared brackets.

Example, running the test ``test_constructor`` with all fixtures:

.. code-block:: python

    check_estimator(NaiveForecaster, tests_to_run="test_constructor")

``{'test_constructor[NaiveForecaster]': 'PASSED'}``

To run or exclude certain test-fixture-combinations, use the ``fixtures_to_run`` or ``fixtures_to_exclude`` arguments.
Values provided should be names of test-fixture-combination strings (str), or a list of such.
Valid strings are precisely the dictionary keys when using ``check_estimator`` with default parameters.

Example, running the test-fixture-combination ``"test_repr[NaiveForecaster-2]"``:

.. code-block:: python

    check_estimator(NaiveForecaster, fixtures_to_run="test_repr[NaiveForecaster-2]")

``{'test_repr[NaiveForecaster-2]': 'PASSED'}``

A useful workflow for using ``check_estimator`` to debug an estimator is as follows:

1. Run ``check_estimator(MyEstimator)`` to find failing tests
2. Subset to failing tests or fixtures using ``fixtures_to_run`` or ``tests_to_run``
3. If the failure is not obvious, set ``raise_exceptions=True`` to raise the exception and inspect the traceback.
4. If the failure is still not clear, use advanced debuggers on the line of code with ``check_estimator``.

Running the test suite in a repository clone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the target location of the estimator is within ``sktime``, then the ``sktime`` test
suite can be run instead. The ``sktime`` test suite (and CI/CD) is ``pytest`` based, ``pytest`` will automatically
collect all estimators of a certain type and tests applying for a given estimator.

For an overview of the testing framework, see the "testing framework" documentation.
Generic interface conformance tests are contained in the classes ``TestAllEstimators``, ``TestAllForecasters``, and so on.
``pytest`` test-fixture-strings for an estimator ``EstimatorName`` will always contain ``EstimatorName`` as a substring,
and are identical with the test-fixture-strings returned by ``check_estimator``.

To run tests only for a given estimator from the console, the command ``pytest -k "EstimatorName"`` can be used.
This will typically have the same effect as using ``check_estimator(EstimatorName)``, only via direct ``pytest`` call.
When using Visual Studio Code or pycharm, tests can also be sub-set using GUI filter
functionality - for this, refer to the respecetive IDE documentation on test integration.

To identify codebase locations of tests applying to a specific estimator,
a quick approach is searching the codebase for test strings produced by ``check_estimator``, preceded by ``def`` (for function/method definition).

Testing within a third party extension package
----------------------------------------------

For third party extension packages to ``sktime`` (open or closed),
or third party modules that aim for interface compliance with ``sktime``,
the ``sktime`` test suite can be imported and extended in the following ways:

* importing ``check_estimator``, this will carry out the tests defined in ``sktime``
  in a single go. ``check_estimator`` can be run within any test framework, including
  ``unittest`` and ``pytest``.

* importing ``parametrize_with_checks`` from ``sktime.utils.estimator_checks``.
  When used in a ``pytest`` test suite, this will parametrize a test function with
  all tests defined in ``sktime`` for a list of estimator classes or instances,
  running each estimator-test combination as a separate test case.
  This pattern requires adding the following test function to the test suite:

    .. code-block:: python

        from sktime.utils.estimator_checks import parametrize_with_checks

        @parametrize_with_checks(OBJS_TO_TEST)
        def test_sktime_api_compliance(obj, test_name):
            check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)

*   importing test classes, e.g., ``test_all_estimators.TestAllEstimators`` or
    ``test_all_forecasters.TestAllForecasters``. The imports will be discovered directly
    by ``pytest``. The test suite also be extended by inheriting from the test classes.


Adding an ``sktime`` compatible estimator to ``sktime``
=======================================================

When adding an ``sktime`` compatible estimator to ``sktime`` itself, a number of
additional things need to be done:

*   ensure that code also meets ``sktime's`` :ref:`documentation <developer_guide_documentation>` standards.
*   add the estimator to the ``sktime`` API reference. This is done by adding a reference to the estimator in the
    correct ``rst`` file inside ``docs/source/api_reference``.
*   authors of the estimator should add themselves to the ``"authors"`` and ``"maintainers"`` tag of the estimator, as owners of the contributed estimator.
*   if the estimator relies on soft dependencies, or adds new soft dependencies, the steps in the :ref:`"dependencies"
    developer guide <dependencies>` should be followed
*   ensure that the estimator passes the entire local test suite of ``sktime``, with the estimator in its target location.
    To run tests only for the estimator, the command ``pytest -k "EstimatorName"`` can be used (or vs code GUI filter functionality)
*   ensure that test parameters in ``get_test_params`` are chosen such that runtime of estimator specific tests remains in the seconds order
    on ``sktime`` remote CI/CD

Don't panic - when contributing to ``sktime``, core developers will give helpful pointers on the above in their PR reviews.

It is recommended to open a draft PR to get feedback early.

Estimators dependent on cython
------------------------------

To add an estimator to ``sktime`` that depends on cython, the following additional steps are needed:

*   all cython code should be present in a separate package on ``pypi`` and/or ``conda-forge``.
    No cython dependent code should be added directly to ``sktime``.
    Below, we call this separate package ``home-package``, for simplicity of reference.
*   In ``home-package``, it is recommended to test the estimator via ``check_estimator``,
    on the same test matrix as ``sktime``: all supported python versions; MacOS, Linux, Windows.
*   In ``sktime``, an interface to the algorithm should be added.
    This can be a simple import from ``home-package``,
    if the algorithm in ``home-package`` already passes ``check_estimator``.
*   Alternatively, the algorithm can be interfaced via a delegator as a delegate,
    tags and method overrides can be added in the delegator. See, e.g., ``MrSQM`` for this.
*   For the ``sktime`` interface, the ``requires_cython`` tag should be set to ``True``,
    and the ``python_dependencies`` tag should be set to the string ``"home-package"``.

If all has been setup correctly, the estimator will be tested in ``sktime`` by the
CI element ``test-cython-estimators``.
Note that this CI element does not cover the full test matrix
of python version and operating systems, this should be done in the upstream package.
