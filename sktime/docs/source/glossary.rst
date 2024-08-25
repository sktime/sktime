.. _glossary:

Glossary of Common Terms
========================

The glossary below defines common terms and API elements used throughout
sktime.

.. glossary::
    :sorted:

    mtype
        ``sktime`` supports multiple in-memory specifications for time series data
        and other objects. Such an in-memory specification is called ``mtype``
        (short for "machine type").
        Each ``mtype`` is represented by a string - e.g., ``pd-multiindex``,
        which defines the data format and the data structure.
        For example, a ``pd-multiindex`` mtype is a collection of time series,
        represented as a 2-level ``MultiIndex``-ed ``pandas.DataFrame``, with
        columns representing variables, rows indexed by ``(instance, timepoint)``,
        where the ``timepoint`` level must be range-like or datetime-like.
        Each mtype implements an abstract data type, a (data) :term:`scitype`,
        for instance ``Panel`` which refers to the abstract type of a collection
        of time series, with instance, timepoint and variable dimensions.
        In this terminology, the ``pd-multiindex`` mtype implements the (abstract)
        ``Panel`` scitype. Data containers can be checked for compliance with
        a given mtype using the :func:`sktime.datatypes.check_is_mtype` function;
        all mtypes can be listed in ``sktime.datatypes.MTYPE_REGISTER``.
        For more details on the general concept, see
        `the datatypes and datasets user guide :doc:</examples/AA_datatypes_and_datasets>`.

    scitype
        Short for scientific type, denotes the abstract type of an ``sktime`` object,
        data container or estimator. One example of an estimator scitype is ``"forecaster"``,
        which denotes the abstract concept of a forecaster with (abstract) ``fit``, ``predict``,
        ``update`` methods. An example of a data scitype is ``Panel``, denoting
        the abstract concept of an indexed collection of time series.
        Scitypes are represented by strings, and are implemented by concrete types.
        For data containers, concrete types are :term:`mtype`-s (see there), for
        estimators, concrete types are python base interfaces, such as defined
        by ``BaseForecaster`` or ``BaseClassifier``. Valid estimator scitypes,
        with their corresponding base classes,
        are listed in ``sktime.registry.BASE_CLASS_SCITYPE_LIST``.
        All estimators of a given scitype can be listed using
        ``sktime.registry.all_estimators``, and the scitype of a given estimator
        can be inferred by the ``sktime.registry.scitype`` utility.
        Compliance with concrete implementations of data scitypes can be checked using
        the ``sktime.datatypes.check_is_scitype`` utility; for estimators, compliance
        is checked using ``sktime.utils.check_estimator``.
        For more details on data scitpyes, see :term:`mtype`.
        For more details on estimator scitypes, see the user guides on individual
        learning tasks.

    Scientific type
        See :term:`scitype`.

    Tag
        Tags are string keyed value fields, used to identify properties of an object,
        or set flags for internal boilerplate. An example of a tag is
        ``capability:multivariate``, a boolean flag, which indicates whether the object
        offers genuine support for multivariate time series.
        Objects with a given capability - that is, objects filtering by
        certain tag value - can be listed or filtered using
        ``sktime.registry.all_estimators``.
        In ``sktime``, most objects are ``scikit-base`` objects and implement
        the tag interface via ``get_tag`` or ``get_tags``.
        Some tags are for internal or extender use only, e.g., ``X_inner_mtype``,
        which allows an extender to specify the mtype of the inner data container
        they would like to work with.
        A list of all tags and their meaning, optionally filtered by the :term:`scitype`
        of object they apply to, can be obtained from
        ``sktime.registry.all_tags``. Further details on tags, for developers,
        can be found in the specification sheet that is part of the :term:`extension templates`.

    Extension templates
        ``sktime`` is designed to be easily extendable, with 3rd and 1st party
        additions in the form of API compliant objects. To facilitate this, ``sktime``
        provides a set of extension templates for power users to implement their own
        objects, such as forecasters, transformers, classifiers.
        The extension templates are found in the ``extension_templates`` folder,
        these are fill-in-the-blank templates that can be used to create new
        objects compliant with the ``sktime`` API.
        Each template is specific to the :term:`scitype` of the object to be implemented,
        and there are different templates for a given :term:`scitype`, depending on
        simplicity vs feature richness.
        The templates instruct a power user on setting of :term:`tags`,
        and implementation of :term:`scitype`-specific methods. The methods are usually
        private, e.g., ``_fit``, ``_predict``, while boilerplate is taken care of
        by the base class.
        For further details and a step-by-step tutorial on 1st and 3rd party
        extensions, see the guide on :ref:`developer_guide_add_estimators`.
        For power users familiar with software engineering
        patterns: the extension templates make use of the template pattern for
        the extension contract, ensuring compliance with the strategy pattern for
        the user contract, defined by the :term:`scitype` specific interface.

    Estimator
        An algorithm of a specific :term:`scitype`, implementing the python
        class interface defined by the scitype.
        Individual estimators correspond to concrete classes, implementing the
        interface defined by the base class for the scitype.
        For example, the ``ARIMA`` class is an estimator of :term:`scitype` ``"forecaster"``.
        Users should distinguish the python class, which can be seen as a blueprint,
        from an instance, which is a concrete object created from the blueprint,
        with specific parameter settings, and which can be fitted or applied to data.
        Somewhat confusingly, both the class (blueprint) and the instance (concrete object)
        are often referred to as "estimator" in ``scikit-learn`` parlance.
        Users should also take note of the distinction between "concrete class" in
        software engineering terms, which is the ``ARIMA`` (python) class, as it implements
        ``BaseForecaster`` (the "abstract class"), and the  "concrete object",
        which is a python instance of a python class.
        Estimators are objects with a ``fit`` method - not all :term:`scitype`-s
        in ``sktime`` are estimators, e.g., performance metrics.

    Composite estimator
        An :term:`estimator` that consists of multiple other component estimators which
        can vary. An example is a pipeline consisting of a transformer and
        forecaster. The term can refer both to the class and its instance.
        For composite estimators, a :term:`tag` can depend on components, such as
        ``capability:missing_data``,
        and a :term:`scitype` that depends on the components' scitypes, e.g., the
        scitype of a pipeline being a forecaster or a classifier, depending on
        whether its last element is a forecaster or a classifier.
        Users familiar with software engineering patterns should note that this term
        may be used in a different sense than "composite pattern":
        in the context of ``scikit-learn``, the "composite estimator"
        combines both the composite pattern and the strategy pattern.

    Hyperparameter:
        A parameter of a machine learning model that is set at construction.
        Usually, this affects the model's performance.
        Examples include the learning rate in a neural network,
        the number of trees in a random forest, or the regularization parameter
        in a linear model.

    Forecasting
        A learning task focused on prediction future values of a time series. For more details, see the :ref:`user_guide_introduction`.

    Time series
         Data where the :term:`variable` measurements are ordered over time or an index indicating the position of an observation in the sequence of values.

    Time series classification
        A learning task focused on using the patterns across instances between the time series and a categorical target variable.

    Time series regression
        A learning task focused on using the patterns across instances between the time series and a continuous target variable.

    Time series clustering
        A learning task focused on discovering groups consisting of instances with similar time series.

    Time series annotation
        A learning task focused on labeling the timepoints of a time series. This includes the related tasks of outlier detection, anomaly detection, change point detection and segmentation.

    Panel time series
        A form of time series data where the same time series are observed observed for multiple observational units. The observed series may consist of :term:`univariate time series` or
        :term:`multivariate time series`. Accordingly, the data varies across time, observational unit and series (i.e. variables).

    Univariate time series
        A single time series. While univariate analysis often only uses information contained in the series itself,
        univariate time series regression and forecasting can also include :term:`exogenous` data.

    Multivariate time series
        Multiple time series. Typically observed for the same observational unit. Multivariate time series
        is typically used to refer to cases where the series evolve together over time. This is related, but different than the cases where
        a :term:`univariate time series` is dependent on :term:`exogenous` data.

    Endogenous
        Within a learning task endogenous variables are determined by exogenous variables or past timepoints of the variable itself. Also referred to
        as the dependent variable or target.

    Exogenous
        Within a learning task exogenous variables are external factors whose pattern of impact on tasks' endogenous variables must be learned.
        Also referred to as independent variables or features.

    Reduction
        Reduction refers to decomposing a given learning task into simpler tasks that can be composed to create a solution to the original task.
        In sktime reduction is used to allow one learning task to be adapted as a solution for an alternative task.

    Variable
        Refers to some measurement of interest. Variables may be cross-sectional (e.g. time-invariant measurements like a patient's place of birth) or
        :term:`time series`.

    Timepoint
        The point in time that an observation is made. A time point may represent an exact point in time (a timestamp),
        a timeperiod (e.g. minutes, hours or days), or simply an index indicating the position of an observation in the sequence of values.

    Instance
        A member of the set of entities being studied and which an ML practitioner wishes to generalize. For example,
        patients, chemical process runs, machines, countries, etc. May also be referred to as samples, examples, observations or records
        depending on the discipline and context.

    Trend
        When data shows a long-term increase or decrease, this is referred to as a trend. Trends can also be non-linear.

    Seasonality
        When a :term:`time series` is affected by seasonal characteristics such as the time of year or the day of the week, it is called a seasonal pattern.
        The duration of a season is always fixed and known.

    Tabular
        Is a setting where each :term:`timepoint` of the :term:`univariate time series` being measured for each instance are treated as features and
        stored as a primitive data type in the DataFrame's cells. E.g., there are N :term:`instances <instance>` of time series and each has T
        :term:`timepoints <timepoint>`, this would yield a pandas DataFrame with shape (N, T): N rows, T columns.

    Framework
        A collection of related and reusable software design templates that practitioners can copy and fill in.
        Frameworks emphasize design reuse.
        They capture common software design decisions within a given application domain and distill them into reusable design templates.
        This reduces the design decision they must take, allowing them to focus on application specifics.
        Not only can practitioners write software faster as a result, but applications will have a similar structure.
        Frameworks often offer additional functionality like :term:`toolboxes`.
        Compare with :term:`toolbox` and :term:`application`.

    Toolbox
        A collection of related and reusable functionality that practitioners can import to write applications.
        Toolboxes emphasize code reuse.
        Compare with :term:`framework` and :term:`application`.

    Application
        A single-purpose piece of code that practitioners write to solve a particular applied problem.
        Compare with :term:`toolbox` and :term:`framework`.

    Bagging:
        A technique in ensemble learning where multiple models are trained on different subsets of the training data,
        and individual model outputs are averaged by some rule (e.g., majority vote) to obtain a consensus prediction.

    Ensemble learning:
        A technique in which multiple models are combined to improve the overall performance of a predictive model.

    Feature extraction:
        A technique used to extract useful information from raw data. In time series analysis, this may involve transforming the
        data to a frequency domain, decomposing the signal into components, or extracting statistical features.

    Generalization:
        The ability of a predictive model to perform well on unseen data. A model that overfits to the training data may not
        generalize well, while a model that underfits may not capture the underlying patterns in the data.

    Model selection:
        The process of selecting the best machine learning model for a given task. This may involve comparing the performance
        of different models on a validation set, or using techniques like grid search to find the best hyperparameters for a given model.

    Time series decomposition:
        A technique used to separate a time series into its underlying components, such as trend, seasonality, and noise.
        This can be useful for understanding the patterns in the data and for modeling each component separately.
