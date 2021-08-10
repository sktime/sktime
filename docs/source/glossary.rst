.. _glossary:

Glossary of Common Terms
========================

The glossary below defines common terms and API elements used throughout
sktime.

.. note::

    The glossary is under development. Important terms are still missing.
    Please create a pull request if you want to add one.


.. glossary::
    :sorted:

    Scitype
        See :term:`scientific type`.

    Scientific type
        A class or object type to denote a category of objects defined by a
        common interface and data scientific purpose. For example, "forecaster"
        or "classifier".

    Forecasting
        A learning task focused on prediction future values of a time series. For more details, see the :ref:`user_guide_forecasting`.

    Time series
        ...

    Time series classification
        A learning task ...

    Time series regression
        A learning task ...

    Time series clustering
        A learning task ...

    Time series annotation
        A learning task ...

    Panel time series
        A form of time series data where the same time series are observed observed for multiple observational units. The observed series may consist of :term:`univariate time series` or
        :term:`multivariate time series`. Accordingly, the data varies across time, observational unit and series (i.e. variables).

    Univariate time series
        A single time series. While univariate analysis often only uses information contained in the series itself,
        univariate time series regression and forecasting can also include :term:`exogenous` data.

    Multivariate time series
        Multiple time series. Typically observed for the same observational unit. In regresssion and forecasting, multivariate time series
        is typically used to refer to cases where the series evolve together over time. This is related, but different than the cases where
        a :term:`Univeriate time series` is dependent on :term:`exogenous` data.

    Endogenous
        ...

    Exogenous
        ...

    Reduction
        ...
