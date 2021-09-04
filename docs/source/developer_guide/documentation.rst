.. _developer_guide_documentation:

=============
Documentation
=============

Providing instructive documentation is a key part of ``sktime's`` mission. In order to meet this,
developers are expected to follow ``sktime's`` documentation standards.

These include:

* Documenting code using NumPy docstrings and sktime conventions
* Following ``sktime's`` docstring convention for public code artifacts and modules
* Adding new public functionality to the :ref:`api_refernce` and :ref:`user guide <user_guide>`

More detailed information on ``sktime's`` documentation format is provided below.

Docstring Conventions
=====================

sktime uses the numpydoc_ Sphinx extension and follows
`NumPy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`.

To ensure docstrings meet expectations, sktime uses a combination of validations built into numpydoc_,
pydocstyle_ (set to the NumPy convention) pre-commit checks, automated testing of docstring examples to ensure
the code runs without error, and :ref:`reviewer <reviewer_guide_doc>` feedback.

sktime Specific Conventions
---------------------------

Beyond basic NumPy docstring formatting conventions, developers should focus on:

- Ensuring all parameters and attributes (classes) are documented completely and consistantly
- Including links to the relevant topics in the :ref:`glossary` or :ref:`user_guide` in the extended summary
- Including an `Examples` section that demonstrates at least basic functionality in all public code artifacts
- Adding a `See Also` section that references related sktime code artifacts as applicable
- Including citations to relevant sources in a `References` section


.. note::

    In many cases a parameter, attribute return object, or error may be described in many docstrings across sktime. To avoid confusion, developers should
    make sure their docstrings are as similar as possible to existing docstring descriptions of the the same parameter, attribute, return object
    or error.

Accordingly, sktime estimators and most other public code artifcations should generally include the following NumPy docstring convention sections:

1. Summary
2. Extended Summary
3. Parameters
4. Attributes (classes only)
5. Returns or Yields (as applicable)
6. Raises (as applicable)
7. See Also (as applicable)
8. Notes (as applicable)
9. References (as applicable)
10. Examples

Summary and Extended Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As is typicaly the summary should be a single line, followed by a properly somewhat longer (and properly formatted) extended summary.
The extended summary should include a user friendly explanation of the code artificats functionality
and links to relevant content in the :ref:`glossary` and :ref:`user guide <user_guide>`.

For all sktime estimators and other code artifacts that implement an algorith,ms (e.g. performance metrics),
the extended summary should also include a short, user-friendly synopsis of the algorithm being implemented.

See Also
~~~~~~~~

This section should reference other ``sktime`` code artifcats related to the code artifact being documented by the docstring. Developers should use
judgement in determining related code artifcats. For example, rather than listin all other performance metrics, a percentage error based performance metric
might only list other percentage error based performance metrics.  Likewise, a distance based classifier might list other distance based classifiers but
not include other types of time series classifiers.

Notes
~~~~~

The notes section can include several types of information, including:

- Mathematical details of a code object or other important implementation details
- Links to alternative implementations of the code artifact that are external to ``sktime`` (e.g. the Java implementation of a sktime
time series classifier)
- state changing methods (sktime estimator classes)

References
~~~~~~~~~~

sktime estimators that implement a concrete algorithm should generally include citations to the original research article, textbook or other resource
that describes the algorithm. Other code artifcations

Examples
~~~~~~~~

Most code artifacts in sktime should include an examples section. At a minimum this should include a single example that illustrates basic functionality.
The examples should use either a built-in sktime dataset or other simple data (e.g. randomly generated data, etc) generated using a sktime dependency
(e.g. NumPy, pandas, etc) and whereever possible only depend on sktime or its core dependencies. Examples should also be designed to run quickly where possible.
For quick running code artifacts, additional examples can be included to illustrate the affect of different parameter settings.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/index.html
.. _pydocstyle: http://www.pydocstyle.org/en/stable/
