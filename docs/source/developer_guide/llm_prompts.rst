.. _developer_guide_llm_prompts:

==============================================
LLM Prompts for Creating Custom Estimators
==============================================

This page provides ready-to-use prompts for large language models (LLMs) such as
ChatGPT, Claude, GitHub Copilot, and others.
These prompts are designed to help you quickly scaffold new ``sktime`` compatible
estimators using AI assistance.


How to use these prompts
========================

Each prompt below is the full text of an ``sktime`` extension template, which
defines the interface contract for a specific type of estimator (forecaster,
transformer, classifier, etc.).

To use a prompt with your preferred LLM:

1. **Copy** the full template code block below for the estimator type you want.
2. **Paste** it into your LLM conversation, followed by a description of the algorithm you want to implement.
3. **Ask the LLM** to fill in the ``# todo`` sections, replacing them with your algorithm logic.

**Example prompt structure:**

.. code-block:: text

    Below is the sktime extension template for a forecaster.
    Please implement <your algorithm description> by filling in the # todo sections.
    Keep the interface intact - only fill in _fit() and _predict() with the algorithm logic.

    <paste template here>

.. note::

    These templates are the source of truth for the sktime interface contract.
    They are automatically included from the
    `extension_templates <https://github.com/sktime/sktime/tree/main/extension_templates>`__
    directory, so they are always up to date with the latest sktime version.

.. tip::

    For best results, also provide your LLM with an example of a similar existing
    sktime estimator (e.g., ``NaiveForecaster`` for a forecaster prompt).
    This gives the model additional context about implementation conventions.


Prompt: Custom Forecaster (Simple)
=====================================

Use this prompt when you want to implement a **time series forecaster** -
an estimator that takes a time series ``y`` as input and produces future forecasts.

This is the ``supersimple`` variant, covering only ``fit`` and ``predict``.
For probabilistic forecasting, hierarchical data, or composition (pipelines),
use the full template at
`extension_templates/forecasting.py <https://github.com/sktime/sktime/blob/main/extension_templates/forecasting.py>`__.

.. dropdown:: Copy prompt: Custom Forecaster (Simple)
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/forecasting_supersimple.py
        :language: python


Prompt: Custom Forecaster (with pipelines/composition)
==========================================================

Use this prompt when your forecaster **wraps or composes** another estimator,
or when you need probabilistic forecasting, hierarchical support, or other
advanced features.

.. dropdown:: Copy prompt: Custom Forecaster (Full)
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/forecasting_simple.py
        :language: python


Prompt: Custom Transformer (Series-to-Series)
==============================================

Use this prompt when you want to implement a **time series transformer** -
an estimator that takes a time series and outputs a transformed time series.
Typical examples: smoothing, deseasonalization, differencing.

This is the ``supersimple`` variant for series-to-series transformations.
For feature extraction (series-to-tabular), see the feature prompt below.
For advanced cases, see
`extension_templates/transformer.py <https://github.com/sktime/sktime/blob/main/extension_templates/transformer.py>`__.

.. dropdown:: Copy prompt: Custom Transformer (Series-to-Series)
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/transformer_supersimple.py
        :language: python


Prompt: Custom Transformer (Series-to-Features)
================================================

Use this prompt when you want to implement a transformer that extracts
**feature vectors** from time series (e.g., summary statistics, spectral features,
word counts).

.. dropdown:: Copy prompt: Custom Transformer (Series-to-Features)
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/transformer_supersimple_features.py
        :language: python


Prompt: Custom Time Series Classifier
======================================

Use this prompt when you want to implement a **time series classifier** -
an estimator that assigns class labels to time series.

.. dropdown:: Copy prompt: Custom Time Series Classifier
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/classification.py
        :language: python


Prompt: Custom Time Series Clusterer
=====================================

Use this prompt when you want to implement a **time series clusterer** -
an estimator that groups time series into clusters without class labels.

.. dropdown:: Copy prompt: Custom Time Series Clusterer
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/clustering.py
        :language: python


Prompt: Custom Anomaly/Change Point Detector
=============================================

Use this prompt when you want to implement a **time series detector** -
an estimator that detects anomalies, change points, or segments in time series.

.. dropdown:: Copy prompt: Custom Anomaly/Change Point Detector
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/detection.py
        :language: python


Prompt: Custom Parameter Estimator
=====================================

Use this prompt when you want to implement a **parameter estimator** -
an estimator that estimates distributional or statistical parameters from time
series data (e.g., seasonal period length, stationarity tests).

.. dropdown:: Copy prompt: Custom Parameter Estimator
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/param_est.py
        :language: python


Prompt: Custom Time Series Splitter
=====================================

Use this prompt when you want to implement a **time series splitter** -
a strategy that splits time series into training and test sets for
cross-validation or evaluation.

.. dropdown:: Copy prompt: Custom Time Series Splitter
    :color: primary
    :icon: paste

    .. literalinclude:: ../../../extension_templates/split.py
        :language: python


Tips for effective LLM-assisted sktime development
====================================================

1. **Describe your algorithm clearly** - mention the paper or method name, key hyperparameters, and any dependencies.
2. **Ask for ``get_test_params``** - make sure the LLM fills in ``get_test_params`` with minimal valid parameters for testing.
3. **Validate with ``check_estimator``** - after generating, run:

   .. code-block:: python

       from sktime.utils.estimator_checks import check_estimator
       from mymodule import MyForecaster

       check_estimator(MyForecaster())

4. **Iterate** - if the first result has issues, share the error message with the LLM and ask for fixes.
5. **Use similar estimators as examples** - finding a similar existing estimator in ``sktime`` and providing its source code alongside the template greatly improves LLM output quality.

For a complete guide on contributing estimators to ``sktime``, see
:ref:`developer_guide_add_estimators`.
