.. _llm_prompts:

LLM Prompts for Sktime Extension Development
============================================

.. note::

   This section is located under "Development" as it primarily targets extension authors and developers who are creating custom sktime estimators. While the original proposal suggested placing it under "More" in the navbar, this location is most appropriate since the content focuses on estimator development using sktime's extension templates.

This page provides helpful prompts for using Large Language Models (LLMs) to assist with developing extensions for sktime, particularly when creating custom estimators using extension templates.

Introduction
------------

As sktime continues to evolve, LLM tools are becoming increasingly useful for supporting the development process. The extension templates provided in sktime are excellent resources for creating new estimators, but many users may not be aware of their full potential.

This section provides copy-and-paste prompts that can be used with LLM tools to help implement your own sktime estimators. These prompts leverage the existing extension templates and best practices developed over years of sktime development.

The prompts below incorporate actual content from the extension templates, which are automatically included during the documentation build process. This ensures that the prompts remain up-to-date with the latest template formats.

Basic Prompts for Estimator Creation
------------------------------------

Creating a Custom Forecaster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this prompt to create a custom forecaster using the forecasting extension template. This prompt incorporates actual content from the template:

.. code-block:: text

    I want to implement a custom forecaster for sktime using the forecasting extension template.
    My forecaster is called [FORECASTER_NAME] and it [BRIEF_DESCRIPTION_OF_FORECASTER].
    It has parameters [PARAMETER_LIST] with default values [DEFAULT_VALUES].
    The forecasting algorithm works by [ALGORITHM_DESCRIPTION].
    Please provide the implementation based on the sktime forecasting extension template,
    including the _fit and _predict methods, proper docstrings, tags, and get_test_params method.

    Here is the current template structure for reference:

.. extension-template-include:: forecasting_simple.py
   :lines: 1-30

Creating a Custom Classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this prompt to create a custom time series classifier. This incorporates content from the classification template:

.. code-block:: text

    I want to implement a custom time series classifier for sktime using the classification extension template.
    My classifier is called [CLASSIFIER_NAME] and it [BRIEF_DESCRIPTION_OF_CLASSIFIER].
    It has parameters [PARAMETER_LIST] with default values [DEFAULT_VALUES].
    The classification algorithm works by [ALGORITHM_DESCRIPTION].
    Please provide the implementation based on the sktime classification extension template,
    including the _fit and _predict methods, proper docstrings, tags, and get_test_params method.

    Here is the current template structure for reference:

.. extension-template-include:: classification.py
   :lines: 1-30

Creating a Custom Transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this prompt to create a custom transformer. This incorporates content from the transformer template:

.. code-block:: text

    I want to implement a custom transformer for sktime using the transformer extension template.
    My transformer is called [TRANSFORMER_NAME] and it [BRIEF_DESCRIPTION_OF_TRANSFORMER].
    It has parameters [PARAMETER_LIST] with default values [DEFAULT_VALUES].
    The transformation algorithm works by [ALGORITHM_DESCRIPTION].
    Please provide the implementation based on the sktime transformer extension template,
    including the _fit, _transform, and _inverse_transform methods (if applicable),
    proper docstrings, tags, and get_test_params method.

    Here is the current template structure for reference:

.. extension-template-include:: transformer_simple.py
   :lines: 1-30

Detailed Problem-Solving Prompts
--------------------------------

Baseline Implementation for Forecasting Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use this prompt when you have a specific forecasting problem and want a recommended baseline:

.. code-block:: text

    I have a forecasting problem where [PROBLEM_DESCRIPTION].
    The data characteristics are [DATA_CHARACTERISTICS] with frequency [FREQUENCY],
    and I have [NUMBER_OF_SERIES] series with [LENGTH_OF_SERIES] observations each.
    There are [MISSING_DATA/SEASONALITY/TREND/ETC] considerations.
    What sktime forecaster would you recommend as a baseline,
    and could you provide an implementation following sktime conventions?

Enhanced Prompts with Example Code
----------------------------------

For more detailed assistance, you can provide example code from similar estimators. When working with LLMs, you can include actual template code:

.. code-block:: text

    I want to implement a custom forecaster similar to [EXISTING_ESTIMATOR_NAME] in sktime.
    Here is the current sktime forecasting template structure:

.. extension-template-include:: forecasting.py
   :start-after: Purpose of this implementation template
   :end-before: Mandatory methods to implement

    I want to modify it to [SPECIFIC_MODIFICATIONS].
    Please provide the implementation following sktime conventions, including:

    1. Proper class inheritance from BaseForecaster
    2. Correct tags for scitype, input types, etc.
    3. Implementation of _fit and _predict methods
    4. Proper docstrings following numpydoc conventions
    5. get_test_params method for testing
    6. Any special capabilities or requirements

    Also include example usage in the docstring.

Automated Documentation Prompts
-------------------------------

To document your estimator properly:

.. code-block:: text

    Generate comprehensive docstrings for this sktime estimator following numpydoc conventions.
    The class name is [CLASS_NAME] and it is a [ESTIMATOR_TYPE].
    Parameters: [PARAMETER_LIST_WITH_TYPES].
    The estimator implements [ALGORITHM_DESCRIPTION].
    It should include: a brief summary, detailed description, parameters section,
    attributes section (ending with underscore), references if any, and examples section.

Testing and Validation Prompts
------------------------------

To ensure your estimator passes sktime's testing framework:

.. code-block:: text

    Help me write the get_test_params method for my sktime estimator.
    The estimator [ESTIMATOR_DESCRIPTION] has parameters [PARAMETER_LIST].
    I need test parameters that cover different internal cases
    while keeping test runtime low (seconds range).
    Provide 2-3 parameter sets that would be appropriate for the check_estimator tests.

Automated Prompt Updates
------------------------

This documentation page implements automatic synchronization with the extension templates. During the documentation build process, the ``extension-template-include`` directive automatically pulls content from the extension template files in the ``extension_templates`` directory. This ensures that:

1. The prompts always reference the most current template format
2. When extension templates are updated, the corresponding documentation updates automatically
3. Users always have access to accurate template content in their LLM prompts

This addresses the requirement to automatically update prompts when the underlying source code changes. As extension templates are modified in the ``extension_templates`` directory, these changes will be reflected in this documentation during the next build process.

Advanced Usage Tips
-------------------

- When using LLMs for estimator development, always review the generated code for compliance with sktime's API and coding standards
- Make sure to set appropriate tags that reflect your estimator's actual capabilities
- Follow sktime's naming conventions and parameter design patterns
- Include proper error handling and input validation
- Test your implementation using `check_estimator(YourEstimator)` before proceeding

For more information about implementing estimators in sktime, see the `developer guide estimators page <developer_guide/add_estimators.html>`_.
