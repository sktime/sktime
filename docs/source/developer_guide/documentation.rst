.. _developer_guide_documentation:

=============
Developing Documentation
=============

Providing instructive documentation is a key part of ``sktime's`` mission. In order to meet this,
developers are expected to follow ``sktime's`` documentation standards.

These include:

* Documenting code using NumPy docstrings and sktime conventions
* Following ``sktime's`` docstring convention for public code artifacts and modules
* Adding new public functionality to the :ref:`api_reference` and :ref:`user guide <user_guide>`

More detailed information on ``sktime's`` documentation format is provided below.

.. contents::
   :local:

Docstring Conventions
---------------------

sktime uses the numpydoc_ Sphinx extension and follows
`NumPy docstring format <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

To ensure docstrings meet expectations, sktime uses a combination of validations built into numpydoc_,
pydocstyle_ pre-commit checks (set to the NumPy convention) and automated testing of docstring examples to ensure
the code runs without error. However, the automated docstring validation in pydocstyle_ only covers basic formatting.
Passing these tests is necessary to meet the sktime docstring conventions, but is not sufficient for doing so.

To ensure docstrings meet sktime's conventions, developers are expected to check their docstrings against numpydoc_
and sktime conventions and :ref:`reviewer's <reviewer_guide_doc>` are expected to also focus feedback on docstring
quality.

sktime Specific Conventions
---------------------------

Beyond basic NumPy docstring formatting conventions, developers should focus on:

- Ensuring all parameters (classes, functions, methods) and attributes (classes) are documented completely and consistently
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

The summary should be a single line, followed by a (properly formatted) extended summary.
The extended summary should include a user friendly explanation of the code artifacts functionality.

For all sktime estimators and other code artifacts that implement an algorithm (e.g. performance metrics),
the extended summary should include a short, user-friendly synopsis of the algorithm being implemented. When the algorithm is implemented
using multiple sktime estimators, the synopsis should first provide a high-level summary of the estimator components (e.g. transformer1 is applied then a classifier).
Additional user-friendly details of the algorithm should follow (e.g. describe how the transformation and classifier work).

The extended summary should also include links to relevant content in the :ref:`glossary` and :ref:`user guide <user_guide>`.

If a "term" already exists in the glossary and the developer wants to link it directly they can use:

.. code-block::

    :term:`the glossary term`

In other cases you'll want to use different phrasing but link to an existing glossary term, and the developer can use:

.. code-block::

    :term:`the link text <the glossary term>`

In the event a term is not already in the glossary, developers should add the term to the glossary (sktime/docs/source/glossary.rst) and include a reference (as shown above)
to the added term.

Likewise, a developer can link to a particular area of the user guide by including an explicit cross-reference and following the steps for referencing in Sphinx
(see the helpful description on `Sphinx cross-references <https://docs.readthedocs.io/en/stable/guides/cross-referencing-with-sphinx.html>`_ posted by Read the Docs).
Again developers are encouraged to add important content to the user guide and link to it if it does not already exist.

See Also
~~~~~~~~

This section should reference other ``sktime`` code artifcats related to the code artifact being documented by the docstring. Developers should use
judgement in determining related code artifcats. For example, rather than listin all other performance metrics, a percentage error based performance metric
might only list other percentage error based performance metrics.  Likewise, a distance based classifier might list other distance based classifiers but
not include other types of time series classifiers.

Notes
~~~~~

The notes section can include several types of information, including:

- Mathematical details of a code object or other important implementation details (using ..math or :math:`` functionality)
- Links to alternative implementations of the code artifact that are external to ``sktime`` (e.g. the Java implementation of an sktime time series classifier)
- state changing methods (sktime estimator classes)

References
~~~~~~~~~~

sktime estimators that implement a concrete algorithm should generally include citations to the original research article, textbook or other resource
that describes the algorithm. Other code artifacts can include references as warranted (for example, references to relevant papers are included in
sktime's performance metrics).

This should be done by adding references into the references section of the docstring, and then typically linking to these in other parts of the docstring.

The references you intend to link to within the docstring should follow a very specific format to ensure they render correctly.
See the example below. Note the space between the ".." and opening bracket, the space after the closing bracket,
and how all the lines after the first line are aligned immediately with the opening bracket.
Additional references should be added in exactly the same way, but the number enclosed in the bracket should be incremented.

.. code-block:: rst

    .. [1] Some research article, link or other type of citation.
       Long references wrap onto multiple lines, but you need to
       indent them so they start aligned with opening bracket on first line.

To link to the reference labeled as "[1]", you use "[1]_". This only works within the same docstring. Sometimes this is not rendered correctly if the "[1]_" link is
preceded or followed by certain characters. If you run into this issue, try putting a space before and following the "[1]_" link.

To list a reference but not link it elsewhere in the docstring, you can leave out the ".. [1]" directive as shown below.

.. code-block:: rst

    Some research article, link or other type of citation.
    Long references wrap onto multiple lines. If you are
    not linking the reference you can leave off the ".. [1]".

Examples
~~~~~~~~

Most code artifacts in sktime should include an examples section. At a minimum this should include a single example that illustrates basic functionality.
The examples should use either a built-in sktime dataset or other simple data (e.g. randomly generated data, etc) generated using an sktime dependency
(e.g. NumPy, pandas, etc) and whereever possible only depend on sktime or its core dependencies. Examples should also be designed to run quickly where possible.
For quick running code artifacts, additional examples can be included to illustrate the affect of different parameter settings.

Examples of Good sktime Docstrings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are a few examples of sktime code artifacts with good documentation.

Estimators
^^^^^^^^^^

BOSSEnsemble_

ContractableBOSS_

Performance Metrics
^^^^^^^^^^^^^^^^^^^

MeanAbsoluteScaledError_

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/index.html
.. _pydocstyle: http://www.pydocstyle.org/en/stable/
.. _BOSSEnsemble: https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.classification.dictionary_based.BOSSEnsemble.html#sktime.classification.dictionary_based.BOSSEnsemble
.. _ContractableBOSS: https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.classification.dictionary_based.ContractableBOSS.html#sktime.classification.dictionary_based.ContractableBOSS
.. _MeanAbsoluteScaledError: https://www.sktime.org/en/latest/api_reference/auto_generated/sktime.performance_metrics.forecasting.MeanAbsoluteScaledError.html

.. _sphinx: https://www.sphinx-doc.org/
.. _readthedocs: https://readthedocs.org/projects/sktime/

Documentation Build
-------------------

We use `sphinx`_ to build our documentation and `readthedocs`_ to host it.
You can find our latest documentation `here <https://www.sktime.org/en/latest/>`_.

The source files can be found
in `docs/source/ <https://github.com/sktime/sktime/tree/main/docs/source>`_.
The main configuration file for sphinx is
`conf.py <https://github.com/sktime/sktime/blob/main/docs/source/conf.py>`__
and the main page is
`index.rst <https://github.com/sktime/sktime/blob/main/docs/source/index.rst>`__.
To add new pages, you need to add a new ``.rst`` file and include it in
the ``index.rst`` file.

To build the documentation locally, you need to install a few extra
dependencies listed in
`pyproject.toml <https://github.com/sktime/sktime/blob/main/pyproject.toml>`__.

1. To install extra dependencies from the root directory, run:

   .. code:: bash

      pip install .[docs]

2. To build the website locally, run:

   .. code:: bash

      cd docs
      make html
