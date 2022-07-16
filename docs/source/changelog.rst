.. _changelog:

Changelog
=========

All notable changes to this project will be documented in this file. We keep track of changes in this file since v0.4.0. The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and we adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_. The source code for all `releases <https://github.com/alan-turing-institute/sktime/releases>`_ is available on GitHub.

.. note::

    To stay up-to-date with sktime releases, subscribe to sktime `here
    <https://libraries.io/pypi/sktime>`_ or follow us on `Twitter <https://twitter.com/sktime_toolbox>`_.

For upcoming changes and next releases, see our `milestones <https://github.com/alan-turing-institute/sktime/milestones?direction=asc&sort=due_date&state=open>`_.
For our long-term plan, see our :ref:`roadmap`.

Version 0.13.0 - 2022-07-14
---------------------------

Highlights
~~~~~~~~~~

* ``sktime`` is now ``python 3.10`` compatible, including the developer suite
* all forecasters and transformers can deal with multivariate data, by vectorization (:pr:`2864`, :pr:`2865`, :pr:`2867`, :pr:`2937`) :user:`fkiraly`
* ``BaggingForecaster`` for adding forecast intervals via bagging (:pr:`2248`) :user:`ltsaprounis`
* ``ReconcilerForecaster`` with more options for hierarchical reconciliation (:pr:`2940`) :user:`ciaran-g`
* new forecasters: ``VARMAX``, ``VECM``, ``DynamicFactor``
  (:pr:`2763`, :pr:`2829`, :pr:`2859`) :user:`KatieBuc` :user:`AurumnPegasus` :user:`lbventura` :user:`ris-bali`

Dependency changes
~~~~~~~~~~~~~~~~~~

* Python requirements and soft dependencies are now isolated to estimator classes where possible, see below.
* ``sktime`` now allows ``numpy 1.22``.
* ``prophet`` soft dependency now must be above 1.1, where it no longer depends on ``pystan``.
* indirect soft dependency on ``pystan`` has been removed.
* soft dependency on ``hcrystalball`` has been removed.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``VectorizedDF`` now supports vectorization over columns

Dependency handling
^^^^^^^^^^^^^^^^^^^

* Python requirements and soft dependencies are now isolated to estimator classes via the ``python_version`` and ``python_dependencies`` tags.
  This allows to bundle algorithms together with their dependency requirements.

Forecasting
^^^^^^^^^^^

* all forecasters can now make mulivariate forecasts. Univariate forecasters do so by iterating/vectorizing over variables.
  In that case, individual forecasters, for variables, are stored in the ``forecasters_`` attribute.
* ``ForecastingHorizon`` now stores frequency information in the ``freq`` attribute.
  It can be set in the constructor via the new ``freq`` argument, and is inferred/updated any time data is passed.

Transformations
^^^^^^^^^^^^^^^

* all transformers can now transform multivariate time series. Univariate transformers do so by iterating/vectorizing over variables.
  In that case, individual transformers, for variables, are stored in the ``transformers_`` attribute.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* deprecated: use of ForecastingHorizon methods with ``pd.Timestamp`` carrying ``freq``
  is deprecated and will raise exception from 0.14.0. Use of ``pd.Timestamp`` will remain possible.
  This due to deprecation of the ``freq`` attribute of ``pd.Timestamp`` in ``pandas``.
* from 0.14.0, public ``cutoff`` attribute of forecasters will change to ``pd.Index`` subtype, from index element.
* removed: class ``HCrystalBallForecaster``, see :pr:`2677`.

Performance metrics
^^^^^^^^^^^^^^^^^^^

* removed: ``func`` and ``name`` args from all performance metric constructors.
* changed: the ``greater_is_better`` property is replaced by the ``greater_is_better`` tag.

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* removed: ``"capability:early_prediction"`` tag from ``BaseClassifier`` descendants.
  Early classifiers are their own estimator type now.
  In order to search for early classifiers, use the ``early-classifier`` scitype string instead of the tag.

Transformations
^^^^^^^^^^^^^^^

* removed: ``Differencer`` - ``drop_na`` *argument* has been removed.
  Default of ``na_handling`` changed to ``fill_zero``
* removed: ``lag_config`` argument in ``WindowSummarizer``, please use ``lag_feature`` argument instead.

Enhancements
~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``VectorizedDF`` to support vectorization across columns/variables (:pr:`2864`) :user:`fkiraly`
* [ENH] preserve ``index.freq`` in ``get_cutoff`` (:pr:`2908`) :user:`fkiraly`
* [ENH] extend ``get_cutoff`` to ``pd.Index`` input (:pr:`2939`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] ``BaggingForecaster`` for adding forecast intervals via bagging (:pr:`2248`) :user:`ltsaprounis`
* [ENH] auto-vectorization over columns for univariate estimators - forecasters (:pr:`2865`) :user:`fkiraly`
* [ENH] auto-vectorization over columns for univariate estimators - transformers (:pr:`2867`, :pr:`2937`) :user:`fkiraly`
* [ENH] turn private cutoff of forecasters into an index that carries ``freq`` (:pr:`2909`) :user:`fkiraly`
* [ENH] ``VECM`` forecasting model (:pr:`2829`) :user:`AurumnPegasus`
* [ENH] addressing ``freq`` deprecation in ``ForecastingHorizon`` (:pr:`2932`) :user:`khrapovs` :user:`fkiraly`
* [ENH] statsmodels ``DynamicFactor`` interface (:pr:`2859`) :user:`lbventura` :user:`ris-bali`
* [ENH] ``ReconcilerForecaster`` and hierarchical transformers update (:pr:`2940`) :user:`ciaran-g`
* [ENH] Avoid accessing ``.freq`` from ``pd.Timestamp`` by converting ``cutoff`` to ``pd.Index`` (:pr:`2965`) :user:`khrapovs`
* [ENH] ``statsmodels`` ``VARMAX`` adapter (:pr:`2763`) :user:`KatieBuc`
* [ENH] add check for forecast to have correct columns (:pr:`2972`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] extend ``ColumnSubset`` to work for scalar ``columns`` parameter (:pr:`2906`) :user:`fkiraly`
* [ENH] transformer vectorization: ensure unique column names if unvectorized output is multivariate (:pr:`2958`) :user:`fkiraly`

Fixes
~~~~~

Data loaders
^^^^^^^^^^^^

* [BUG] ``load_UCR_UEA_dataset`` checks for existence of files rather than just directories (:pr:`2899`) :user:`TonyBagnall`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fixing ``get_time_index`` for 1D and 2D ``numpy`` formats (:pr:`2852`) :user:`fkiraly`
* [BUG] Fixing broken conversions from nested data frame (:pr:`2375`) :user:`fkiraly`
* [BUG] preserve ``pd-multiindex`` index names (:pr:`2999`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] loosen index check related tags and fix incorrect pipeline tag inference (:pr:`2842`) :user:`fkiraly`
* [BUG] remove non-standard ``score`` function in ``BaseGridSearch`` (:pr:`2752`) :user:`fkiraly`
* [BUG] fix ``Prophet`` to have correct output column names (:pr:`2973`) :user:`fkiraly`
* [BUG] fixing grid/random search broken delegation (:pr:`2945`) :user:`fkiraly`
* [BUG] forecaster vectorization for ``update`` and proba prediction, bugfixes (:pr:`2960`) :user:`fkiraly`
* [BUG] fix pipeline vectorization for univariate estimators (:pr:`2959`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix bug in verbose mode in CNN TSC models (:pr:`2882`) :user:`tobiasweede`
* [BUG] Early classification test fixes (:pr:`2980`) :user:`MatthewMiddlehurst`

Transformations
^^^^^^^^^^^^^^^

* [BUG] ensure ``IntervalSegmenter`` unique column output (:pr:`2970`) :user:`fkiraly`
* [BUG] fix NaN columns in bootstrap transformers (:pr:`2974`) :user:`fkiraly`
* [BUG] ensure ``TruncationTransformer.transform`` output now has same columns as input (:pr:`2999`) :user:`fkiraly`

Refactored
~~~~~~~~~~

* [ENH] ``NaiveForecaster``: remove manual vectorization layer in favour of base class vectorization (:pr:`2874`) :user:`fkiraly`
* [ENH] remove old ``multiindex-df`` index convention hack from ``VectorizedDF`` (:pr:`2863`) :user:`fkiraly`
* [ENH] delete duplicate classifier tests (:pr:`2912`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] upgrade the ``all`` modules to automatic retrieval (:pr:`2845`) :user:`fkiraly`
* [MNT] Upgrade ``prophet`` to >=1.1 and remove ``pystan`` from ``all_extras`` dependencies (:pr:`2887`) :user:`khrapovs`
* [MNT] Remove ``cmdstanpy`` from ``all_extras`` (:pr:`2900`) :user:`khrapovs`
* [MNT] estimator upper bound tag for selective version compatibility, test exclusion (:pr:`2660`) :user:`fkiraly`
* [MNT] python 3.10 upgrade with estimator version tag (:pr:`2661`) :user:`fkiraly`
* [MNT] package dependency tags (:pr:`2915`, :pr:`2994`) :user:`fkiraly`
* [MNT] soft dependency testing (:pr:`2920`) :user:`fkiraly`
* [MNT] remove Azure build tools and dependency handling instructions (:pr:`2917`) :user:`fkiraly`
* [MNT] Fix changelog generator (:pr:`2892`) :user:`lmmentel`
* [MNT] Update ``numpy`` version bound to ``<=1.22`` (:pr:`2979`) :user:`jlopezpena`
* [MNT] 0.13.0 deprecation actions (:pr:`2895`) :user:`fkiraly`
* [MNT] set number of ``pytest-xdist`` workers to ``auto`` (:pr:`2992`) :user:`fkiraly`
* [MNT] Remove ``hcrystalball`` dependency (:pr:`2858`) :user:`aiwalter`

Documentation
~~~~~~~~~~~~~

* [DOC] updated forecasting tutorial with multivariate vectorization (:pr:`3000`) :user:`fkiraly`
* [DOC] ``all_estimators`` authors variable (:pr:`2861`) :user:`fkiraly`
* [DOC] added missing credits in ``naive.py`` (:pr:`2876`) :user:`fkiraly`
* [DOC] add ``_is_vectorized`` to forecaster extension template exclusion list (:pr:`2878`) :user:`fkiraly`
* [DOC] replace ``AyushmaanSeth`` name with GitHub ID (:pr:`2911`) :user:`fkiraly`
* [DOC] Added docstrings code showing example of using ``metrics`` with ``evaluate`` (:pr:`2850`) :user:`TNTran92`
* [DOC] updated release process to current de-facto process (:pr:`2927`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`a-pasos-ruiz`,
:user:`aiwalter`,
:user:`AurumnPegasus`,
:user:`ciaran-g`,
:user:`fkiraly`,
:user:`haskarb`,
:user:`jlopezpena`,
:user:`KatieBuc`,
:user:`khrapovs`,
:user:`lbventura`,
:user:`lmmentel`,
:user:`ltsaprounis`,
:user:`MatthewMiddlehurst`,
:user:`ris-bali`,
:user:`TNTran92`,
:user:`tobiasweede`,
:user:`TonyBagnall`

Version 0.12.1 - 2022-06-28
---------------------------

Highlights
~~~~~~~~~~

* new ``ReconcilerForecaster`` estimator for reconciling forecasts using base model residuals  (:pr:`2830`) :user:`ciaran-g`
* ``|`` dunder for multiplexing and autoML, shorthand for ``MultiplexTransformer`` (:pr:`2810`) :user:`miraep8`
* lagging transformer ``Lag`` for easy generation of lags (:pr:`2783`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* upper bound ``prophet < 1.1`` due to ``cmdstanpy`` incompatibility

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* ``set_params`` now behaves identically to ``__init__`` call with corresponding parameters, including dynamic setting of tags.
  This is to fully comply with the ``sklearn`` interface assumption that this is the case. (:pr:`2835`) :user:`fkiraly`

Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] ``set_params`` to call ``reset``, to comply with ``sklearn`` parameter interface assumptions (:pr:`2835`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] make ``get_cutoff`` compatible with all time series formats, fix bug for ``VectorizedDF`` input (:pr:`2870`) :user:`fkiraly`
* [ENH] more informative error messages to diagnose wrong input format to forecasters (:pr:`2824`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] subsetting transformations (:pr:`2831`) :user:`fkiraly`

Fixes
~~~~~

Forecasting
^^^^^^^^^^^

* [BUG] fixed forecasters not updating ``cutoff`` when in vectorization mode (:pr:`2870`) :user:`fkiraly`
* [BUG] Fixing type conversion bug for probabilistic interval wrappers ``NaiveVariance`` and ``ConformalInterval`` (:pr:`2815`) :user:`bethrice44`
* [BUG] fix ``Lag`` transformer when ``numpy.int`` was passed as lag integers (:pr:`2832`) :user:`fkiraly`
* [ENH] fix ``get_window`` utility when ``window_length`` was ``None`` (:pr:`2866`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] Vectorization in transformers overwrote ``y`` with ``X`` if ``y`` was passed (:pr:`2844`) :user:`fkiraly`
* [BUG] output type check fix for ambiguous return types in vectorized ``Panel`` case (:pr:`2843`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] add missing ``Sajaysurya`` references (:pr:`2800`) :user:`fkiraly`
* [DOC] add missing ``TonyBagnall`` to contributors of 0.12.0 in changelog (:pr:`2803`) :user:`fkiraly`
* [DOC] adds solution to "no matches found" to troubleshoot section of install guide (:pr:`2786`) :user:`AurumnPegasus`
* [DOC] cleaning up transformer API reference (:pr:`2818`) :user:`fkiraly`
* [DOC] team update: remove ``TonyBagnall`` from CC (:pr:`2794`) :user:`fkiraly`
* [DOC] Added ``diviner`` by Databricks and ``statsforecast`` by Nixtla to related software (:pr:`2873`) :user:`aiwalter`

Maintenance
~~~~~~~~~~~

* [MNT] test univariate forecasting with ``pd.DataFrame`` input and longer ``fh`` (:pr:`2581`) :user:`fkiraly`
* [MNT] Address ``FutureWarnings`` from ``numpy`` (:pr:`2847`) :user:`khrapovs`
* [MNT] Fix loop reassignment (:pr:`2840`) :user:`khrapovs`

Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`AurumnPegasus`,
:user:`bethrice44`,
:user:`ciaran-g`,
:user:`fkiraly`,
:user:`khrapovs`,
:user:`miraep8`


Version 0.12.0 - 2022-06-12
---------------------------

Highlights
~~~~~~~~~~

* Time series classification: deep learning based algorithms, port of ``sktime-dl`` into ``sktime`` (:pr:`2447`) :user:`TonyBagnall`
* forecasting data splitters now support hierarchical data (:pr:`2599`) :user:`fkiraly`
* Updated forecasting and classification notebooks (:pr:`2620`, :pr:`2641`) :user:`fkiraly`
* frequently requested algorithm: Kalman filter transformers (:pr:`2611`) :user:`NoaBenAmi` :user:`lielleravid`
* frequently requested algorithm: top-down reconciler based on forecast proportions (:pr:`2664`) :user:`ciaran-g`
* frequently requested algorithm: empirical and conformal prediction intervals after Stankeviciute et al, 2021 (:pr:`2542`, :pr:`2706`) :user:`bethrice44` :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* new soft dependencies: ``pykalman`` and ``filterpy`` (for Kalman filter transformers)

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* all estimators now reset and execute ``__init__`` at the start of ``fit`` (:pr:`2562`) :user:`fkiraly`.
  Pre ``fit`` initialization and checks can therefore also be written at the end of ``__init__`` now.
* all estimators now possess a ``clone`` method which is function equivalent to ``sklearn``'s' ``clone`` (:pr:`2565`) :user:`fkiraly`.

Forecasting
^^^^^^^^^^^

* ``ExpandingWindowSplitter`` with data individually added is now default ``cv`` in ``BaseForecaster.update_predict`` (:pr:`2679`) :user:`fkiraly`.
  Previously, not specifying ``cv`` would result in an error.

Performance metrics
^^^^^^^^^^^^^^^^^^^

* performance metrics have a new base class design and inheritance structure.
  See ``BaseForecastingErrorMetric`` docstring documentation.
  Changes to the interface are downwards compatible and lay the groundwork for further refactoring.

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* TSR base class was updated to an interface that parallels ``BaseClassifier`` (:pr:`2647`) :user:`fkiraly`.
  See the base class docstrings for specification details.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* removed: ``instance_index`` and ``time_index`` args from ``from_multi_index_to_3d_numpy``. Use ``convert`` or ``convert_to`` instead.

Forecasting
^^^^^^^^^^^

* removed: tag ``fit-in-predict``, now subsumed under ``fit_is_empty``
* deprecated: ``HCrystalBallForecaster``, will be removed in 0.13.0. See :pr:`2677`.

Performance metrics
^^^^^^^^^^^^^^^^^^^

* changed: set ``symmetric`` hyper-parameter default to ``True`` in all relative performance metrics.
* deprecated: ``func`` and ``name`` args will be removed from all performance metric constructors in 0.13.0.
  If these attributes are needed, they should be object or class attributes, and can be optional constructor arguments.
  However, it will no longer be required that all performance metrics have ``func`` and ``name`` as constructor arguments.
* deprecated: the ``greater_is_better`` property will be replaced by the ``greater_is_better`` tag, in 0.13.0.
  Until then, implementers should set the ``greater_is_better`` tag.
  Users can still call the ``greater_is_better`` property until 0.13.0, which will alias the ``greater_is_better`` tag, if set.

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^
* deprecated: ``"capability:early_prediction"`` will be removed in 0.13.0 from ``BaseClassifier`` descendants.
  Early classifiers should inherit from the learning task specific base class ``BaseEarlyClassifier`` instead.

Transformations
^^^^^^^^^^^^^^^

* removed: tag ``fit-in-transform``, now subsumed under ``fit_is_empty``
* removed: ``FeatureUnion``'s ``preserve_dataframe`` parameter
* removed: ``series_as_features.compose`` module, contents are in ``transformations.compose``
* removed: ``transformations.series.window_summarize`` module, contents are in ``transformations.series.summarize``
* changed: ``"drift"``, ``"mean"``, ``"median"``, ``"random"`` methods of ``Imputer`` now use the training set (``fit`` arguments)
  to compute parameters. For pre-0.12.0 behaviour, i.e., using the ``transform`` set, wrap the ``Imputer`` in the ``FitInTransform`` compositor.

Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] add `reset` at the start of all `fit` (:pr:`2562`) :user:`fkiraly`
* [ENH] move `clone` to `BaseObject` (:pr:`2565`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Add support to ``get_slice`` for multi-index and hierarchical data (:pr:`2761`) :user:`bethrice44`

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [ENH] TWE switch to use euclidean distance (:pr:`2639`) :user:`chrisholder`

Forecasting
^^^^^^^^^^^

* [ENH] early fit for ``NaiveVariance`` (:pr:`2546`) :user:`fkiraly`
* [ENH] empirical and conformal probabilistic forecast intervals (Stankeviciute et al, 2021)  (:pr:`2542` :pr:`2706`) :user:`bethrice44` :user:`fkiraly`
* [ENH] ``BaseSplitter`` extension: hierarchical data, direct splitting of series (:pr:`2599`) :user:`fkiraly`
* [ENH] Top-down reconciler based on forecast proportions (:pr:`2664`) :user:`ciaran-g`
* [ENH] ``HCrystalBallForecaster`` deprecation (:pr:`2675`) :user:`aiwalter`
* [ENH] add ``int`` handling to ``Prophet`` (:pr:`2709`) :user:`fkiraly`
* [ENH] Compositor for forecasting of exogeneous data before using in exogeneous forecaster (:pr:`2674`) :user:`fkiraly`
* [ENH] add ``ExpandingWindowSplitter`` as default cv in ``BaseForecaster.update_predict`` (:pr:`2679`) :user:`fkiraly`

Performance metrics
^^^^^^^^^^^^^^^^^^^

* [ENH] new probabilistic metrics for interval forecasts - empirical coverage, constraint violation (:pr:`2383`) :user:`eenticott-shell`
* [ENH] metrics rework part II - metrics internal interface refactor (:pr:`2500`) :user:`fkiraly`
* [ENH] metrics rework part III - folding metric mixins into intermediate class, interface consolidation (:pr:`2502`) :user:`fkiraly`
* [ENH] tests for probabilistic metrics (:pr:`2683`) :user:`eenticott-shell`

Pipelines
^^^^^^^^^

* [ENH] ``make_pipeline`` utility to create linear pipelines of any type (:pr:`2643`) :user:`fkiraly`

Time series classification and regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Transfer deep learning classifiers and regressors from ``sktime-dl`` (:pr:`2447`) :user:`TonyBagnall`
* [ENH] Proximity forest, removal of legacy conversion (:pr:`2518`) :user:`fkiraly`
* [ENH] update TSR base class, kNN time series regression (:pr:`2647`) :user:`fkiraly`
* [ENH] ``DummyClassifier``, naive classifier baseline (:pr:`2707`) :user:`ZiyaoWei`
* [ENH] pipeline for time series classification from sktime transformers and sklearn classifiers (:pr:`2718`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Kalman filter - transformers (:pr:`2611`) :user:`NoaBenAmi` :user:`lielleravid`
* [ENH] transformer adaptor for `pandas` native methods (:pr:`2699`) :user:`fkiraly`
* [ENH] testing hierarchical input to transformers (:pr:`2721`) :user:`fkiraly`
* [ENH] ``MultiplexTransformer`` for multiplexing transformers (:pr:`2738`, :pr:`2778`, :pr:`2780`) :user:`miraep8`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] allow different import and package names in soft dependency check (:pr:`2545`) :user:`fkiraly`
* [ENH] option to exclude tests/fixtures in ``check_estimator`` (:pr:`2756`) :user:`fkiraly`
* [ENH] ``make_mock_estimator`` passing constructor args for the mocked class (:pr:`2686`) :user:`ltsaprounis`
* [ENH] ``test_update_predict_predicted_index`` for continuous data (:pr:`2701`) :user:`ltsaprounis`
* [ENH] interface compliance test to ensure sklearn compliance of constructor (:pr:`2732`) :user:`fkiraly`
* [ENH] ``check_estimators`` to run without soft dependencies (:pr:`2779`) :user:`fkiraly`
* [ENH] forecasting pipeline test which triggers conversions and failure condition in #2739 (:pr:`2790`) :user:`fkiraly`
* [ENH] expose estimator method iteration in ``TestAllEstimators`` as test fixture (:pr:`2781`) :user:`fkiraly`

Governance
^^^^^^^^^^

* [DOC] Add ``khrapovs`` to core devs (:pr:`2743`) :user:`khrapovs`
* [DOC] core dev & gsoc mentor contributions badges (:pr:`2684`) :user:`fkiraly`

Fixes
~~~~~

Clustering
^^^^^^^^^^

* [BUG] fixed constructor non-compliance with sklearn: ``TimeSeriesKMeans`` (:pr:`2773`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``pd.Series`` to ``pd.DataFrame`` mtype conversion in case series has a name (:pr:`2607`) :user:`fkiraly`
* [BUG] corrected ``Series`` to ``Panel`` conversion for numpy formats (:pr:`2638`) :user:`fkiraly`

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [BUG] fixed bug with distance factory 1d arrays (:pr:`2691`) :user:`chrisholder`
* [BUG] fixed constructor non-compliance with sklearn: ``ShapeDTW`` (:pr:`2773`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] Fix incorrect ``update_predict`` arg default and docstring on ``cv`` arg (:pr:`2589`) :user:`aiwalter`
* [BUG] Fix ``Prophet`` with logistic growth #1079 (:pr:`2609`) :user:`k1m190r`
* [BUG] ``ignores-exogeneous-X`` tag correction for ``UnobservedComponents`` (:pr:`2666`) :user:`fkiraly`
* [BUG] fixed ``StackingForecaster`` for exogeneous data (:pr:`2667`) :user:`fkiraly`
* [BUG] fixed ``pmdarima`` interface index handling if ``X`` index set is strictly larger than ``y`` index set (:pr:`2673`) :user:`fkiraly`
* [BUG] Fix duration to ``int`` coercion for ``pd.tseries.offsets.BaseOffset`` (:pr:`2726`) :user:`khrapovs`
* [BUG] fixed overlap in ``NaiveVariance`` train/test set due to inclusive indexing for timestamp limits (:pr:`2760`) :user:`bethrice44`
* [BUG] fixed constructor non-compliance with sklearn: ``AutoETS`` (:pr:`2736`) :user:`fkiraly`
* [BUG] fixed constructor non-compliance with sklearn: ``UnobservedComponents`` (:pr:`2773`) :user:`fkiraly`
* [BUG] fixed ``sarimax_kwargs`` in ``ARIMA`` and ``AutoARIMA`` being incompliant with scikit-learn interface (:pr:`2731`, :pr:`2773`) :user:`fkiraly`
* [BUG] add patch to ensure column/name preservation in ``NaiveForecaster`` (:pr:`2793`) :user:`fkiraly`

Time series classification and regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fixing constructor non-compliance with sklearn: ``KNeighborsTimeSeriesClassifier`` and ``KNeighborsTimeSeriesRegressor`` (:pr:`2737`, :pr:`2773`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] Fixed fit method of Imputer (:pr:`2362`) :user:`aiwalter`
* [BUG] fix typo in author variable for boxcox module (:pr:`2642`) :user:`fkiraly`
* [BUG] ``TransformerPipeline`` fix for vectorization edge cases and sklearn transformers (:pr:`2644`) :user:`fkiraly`
* [BUG] ``SummaryTransformer`` multivariate output fix and tests for series-to-primitives transform output (:pr:`2720`) :user:`fkiraly`
* [BUG] fixing constructor non-compliance with sklearn: ``PCATransformer`` (:pr:`2734`) :user:`fkiraly`


Maintenance
~~~~~~~~~~~

* [MNT] Added ``pytest`` flags to ``setup.cfg`` (:pr:`2535`) :user:`aiwalter`
* [MNT] Added deprecation warning for ``HCrystalBallForecaster`` (:pr:`2675`) :user:`aiwalter`
* [MNT] Replace deprecated argument ``squeeze`` with the method `.squeeze("columns")` in `pd.read_csv` (:pr:`2693`) :user:`khrapovs`
* [MNT] Replace ``pandas.DataFrame.append`` with ``pandas.concat`` to address future deprecation (:pr:`2723`) :user:`khrapovs`
* [MNT] Add [MNT] tag to PR template (:pr:`2727`) :user:`khrapovs`
* [MNT] Removed redundant ``todo`` from ``transformer_simple`` extension template (:pr:`2740`) :user:`NoaBenAmi`
* [MNT] Address various future warnings from ``pandas`` and ``numpy`` (:pr:`2725`) :user:`khrapovs`
* [MNT] testing ``sktime`` without softdeps (:pr:`2719`) :user:`fkiraly`
* [MNT] remove accidental ``codecov`` overwrite from ``nosoftdeps`` (:pr:`2782`) :user:`fkiraly`
* [MNT] deprecation actions scheduled for 0.12.0 release (:pr:`2747`) :user:`fkiraly`

Refactored
~~~~~~~~~~

* [ENH] refactored dunder concatenation logic (:pr:`2575`) :user:`fkiraly`
* [ENH] ``get_test_params`` refactor for ``PyODAnnotator`` (:pr:`2755`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] Forecasting notebook update (:pr:`2620`) :user:`fkiraly`
* [DOC] Links to ``extension_templates`` folder (:pr:`2623`) :user:`Ris-Bali`
* [DOC] Classification notebook clean-up, added new pipelines (:pr:`2641`) :user:`fkiraly`
* [DOC] Text changes example notebooks (:pr:`2648`) :user:`lbventura`
* [DOC] update doc location of ``TimeSeriesForestClassifier`` from ``kernel_based`` to ``interval_based`` in ``get_started.rst`` (:pr:`2722`) :user:`dougollerenshaw`
* [DOC] ``update_predict`` docstrings corrected (:pr:`2671`) :user:`fkiraly`
* [DOC] Fixes in class description ``ExpandingWindowSplitter`` (:pr:`2676`) :user:`keepersas`
* [DOC] Fixed A Few Links on the Website (:pr:`2688`) :user:`asattiraju13`
* [DOC] updated utility API docs (:pr:`2703`) :user:`fkiraly`
* [DOC] Added list of interns to website (:pr:`2708`) :user:`aiwalter`
* [DOC] reserved variables listed in extension templates (:pr:`2769`) :user:`fkiraly`
* [DOC] Fix broken link to governance website page in governance.md (:pr:`2795`) :user:`DBCerigo`
* [DOC] cleaning up forecasting API reference (:pr:`2798`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`asattiraju13`,
:user:`bethrice44`,
:user:`chrisholder`,
:user:`ciaran-g`,
:user:`DBCerigo`,
:user:`dougollerenshaw`,
:user:`eenticott-shell`,
:user:`fkiraly`,
:user:`k1m190r`,
:user:`keepersas`,
:user:`khrapovs`,
:user:`lbventura`,
:user:`lielleravid`,
:user:`ltsaprounis`,
:user:`miraep8`,
:user:`NoaBenAmi`,
:user:`Ris-Bali`,
:user:`TonyBagnall`,
:user:`ZiyaoWei`


Version 0.11.4 - 2022-05-13
---------------------------

Highlights
~~~~~~~~~~

* maintenance update for compatibility with recent ``scikit-learn 1.1.0`` release

Dependency changes
~~~~~~~~~~~~~~~~~

* Added defensive upper bound ``scikit-learn<1.2.0``

Maintenance
~~~~~~~~~~~

* [MNT] fix incompatibility with ``sklearn 1.1.0`` (:pr:`2632`, :pr:`2633`) :user:`fkiraly`
* [MNT] clean-up of ``test_random_state`` (:pr:`2593`) :user:`Ris-Bali`
* [MNT] fix side effects in ``check_estimator`` utility (:pr:`2597`) :user:`fkiraly`
* [MNT] ``_check_dl_dependencies`` warning option (:pr:`2627`) :user:`fkiraly`


Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] components retrieval utility and default `BaseForecaster._update(update_params=False)` for composites (:pr:`2596`) :user:`fkiraly`

Clustering
^^^^^^^^^^

* [ENH] Dynamic Time Warping Barycenter Averaging (DBA) (:pr:`2582`) :user:`chrisholder`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] more informative error message from ``mtype`` if no mtype can be identified (:pr:`2606`) :user:`fkiraly`

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [ENH] Twe distance (:pr:`2553`) :user:`chrisholder`

Forecasting
^^^^^^^^^^^

* [ENH] Extended sliding and expanding window splitters to allow timdelta forecasting horizon (:pr:`2551`) :user:`khrapovs`
* [ENH] Removed ``interval_width`` parameter of Prophet (:pr:`2630`) :user:`phershbe`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* decrease ensemble size for DrCIF (:pr:`2595`) :user:`TonyBagnall`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Created ``_DelegatedTransformer`` (:pr:`2612`) :user:`miraep8`
* [ENH] transformer reconcilers - add tests and improve (:pr:`2577`) :user:`ciaran-g`


Fixes
~~~~~

BaseObject
^^^^^^^^^^

* [BUG] ``BaseObject.reset`` to return ``self`` (:pr:`2613`) :user:`fkiraly`
* [BUG] typo fix in tag deprecation message (:pr:`2616`) :user:`fkiraly`

Clustering
^^^^^^^^^^

* [BUG] Clustering lloyds algorithm early exit incorrectly (:pr:`2572`) :user:`chrisholder`
* [BUG] fixed bug where no average params passed (:pr:`2592`) :user:`chrisholder`
* [BUG] Twe distance running slow due to numpy and numba interaction (:pr:`2605`) :user:`chrisholder`

Forecasting
^^^^^^^^^^^

* [BUG] Forecasting pipeline get/set params fixed for dunder generated pipelines (:pr:`2619`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [BUG] fixing side effects between test runs of the same test in the test suite (:pr:`2558`) :user:`fkiraly`


Contributors
~~~~~~~~~~~~

:user:`chrisholder`,
:user:`ciaran-g`,
:user:`fkiraly`,
:user:`khrapovs`,
:user:`miraep8`,
:user:`phershbe`,
:user:`Ris-Bali`,
:user:`TonyBagnall`


Version 0.11.3 - 2022-04-29
---------------------------

Highlights
~~~~~~~~~~

* ``sktime`` is now compatible with ``scipy 1.8.X`` versions (:pr:`2468`, :pr:`2474`) :user:`fkiraly`
* dunder method for forecasting pipelines: write ``trafo * forecaster * my_postproc`` for ``TransformedTargetForecaster`` pipeline (:pr:`2404`) :user:`fkiraly`
* dunder method for multiplexing/autoML: write ``forecaster1 | forecaster2 | forecaster3`` for ``MultiplexForecaster``, used in tuning over forecasters (:pr:`2540`) :user:`miraep8`
* dunders combine with existing transformer pipeline and feature union, e.g., ``trafo1 * trafo2 * forecaster`` or ``(trafo1 + trafo2) * forecaster``
* prediction intervals for ``UnobservedComponents`` forecaster (:pr:`2454`) :user:`juanitorduz`
* new argument ``return_tags`` of ``all_estimators`` allows listing estimators together with selected tags (:pr:`2410`) :user:`miraep8`

Dependency changes
~~~~~~~~~~~~~~~~~~

* Upper bound on ``scipy`` relaxed to ``scipy<1.9.0``, ``sktime`` is now compatible with ``scipy 1.8.X`` versions.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

All Estimators
^^^^^^^^^^^^^^

All estimators now have a ``reset`` method which resets objects a clean post-init state, keeping hyper-parameters.
Equivalent to ``clone`` but overwrites ``self``.

Forecasting
^^^^^^^^^^^

Forecasters have two new dunder methods. Invoke dunders for easy creation of a pipeline object:

* ``*`` with a transformer creates forecasting pipeline, e.g., ``my_trafo1 * my_forecaster * my_postproc``.
  Transformers before the forecaster are used for pre-processing in a ``TransformedTargetForecaster``.
  Transformers after the forecaster are used for post-processing in a ``TransformedTargetForecaster``.

* ``|`` with another forecaster creates a multiplexer, e.g., ``forecaster1 | forecaster2 | forecaster 3``.
  Result is of class ``MultiplexForecaster`` which can be combined with grid search for autoML style tuning.

Dunder methods are compatible with existing transformer dunders ``*`` (pipeline) and ``+`` (feature union).

Forecaster ``update_predict`` now accepts an additional boolean argument ``reset_forecaster``.
If ``reset_forecaster = True`` (default and current intended behaviour), forecaster state does not change.
If ``reset_forecaster = False``, then ``update``, ``predict`` sequence updates state.

In 0.13.0, the default will change to ``reset_forecaster = False``.

Deprecations
~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

Forecaster ``update_predict`` default behaviour will change from ``reset_forecaster = True`` to ``reset_forecaster = False``, from 0.13.0 (see above).

Transformations
^^^^^^^^^^^^^^^

``Differencer``: ``drop_na`` argument will be deprecated from 0.12.0 and removed in 0.13.0.
It will be replaced bz the ``na_handling`` argument and a default of ``"fill_zero"``.

``WindowSummarizer``: ``lag_config`` will be deprecated from 0.12.0 and removed in 0.13.0.
It will be replaced by the ``lag_feature`` argument and new specification syntax for it.


Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] BaseObject ``reset`` functionality (:pr:`2531`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] new ``_make_panel`` utility, separate from ``_make_panel_X``, with arbitrary return mtype (:pr:`2505`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] prediction intervals for ``UnobservedComponents`` forecaster (:pr:`2454`) :user:`juanitorduz`
* [ENH] remove error message on exogeneous ``X`` from DirRec reducer (:pr:`2463`) :user:`fkiraly`
* [ENH] replace ``np.arange`` by ``np.arghwere`` in splitters to enable time based indexing and selection (:pr:`2394`) :user:`khrapovs`
* [ENH] Test ``SingleWindowSplitter`` with Timedelta forecasting horizon (:pr:`2392`) :user:`khrapovs`
* [ENH] ``Aggregator``: remove index naming requirement (:pr:`2479`) :user:`ciaran-g`
* [ENH] ``MultiplexForecaster`` compatibility with multivariate, probabilistic and hierarchical forecasting (:pr:`2458`) :user:`fkiraly`
* [ENH] ``Differencer`` NA handling - "fill zero" parameter (:pr:`2487`) :user:`fkiraly`
* [ENH] Add ``random_state`` to ``statsmodels`` adapter and estimators (:pr:`2440`) :user:`ris-bali`
* [ENH] Added tests for ``MultiplexForecaster`` (:pr:`2520`) :user:`miraep8`
* [ENH] Added ``|`` dunder method for ``MultiplexForecaster`` (:pr:`2540`) :user:`miraep8`

Registry
^^^^^^^^

* [ENH] add new argument ``return_tags`` to ``all_estimators`` (:pr:`2410`) :user:`miraep8`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] unequal length classifier scenario (:pr:`2516`) :user:`fkiraly`
* [ENH] Tests for multiple classifier input mtypes (:pr:`2508`, :pr:`2523`) :user:`fkiraly`
* [ENH] more forecaster scenarios for testing: using ``X`` (:pr:`2462`) :user:`fkiraly`
* [ENH] Logger update - ``__init__`` removal, private ``log`` attribute (:pr:`2533`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Added ``FitInTransform`` transformer (:pr:`2534`) :user:`aiwalter`

Fixes
~~~~~

Clustering
^^^^^^^^^^

* [BUG] Fixed medoids in kmedoids being taken across all data instead of cluster-wise (:pr:`2548`) :user:`chrisholder`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fixing direct conversions from/to ``numpyflat`` mtype being overriden by indirect ones (:pr:`2517`) :user:`fkiraly`

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [BUG] Distances fixed bug where bounding matrix was being rounded incorrectly (:pr:`2549`) :user:`chrisholder`

Forecasting
^^^^^^^^^^^

* [BUG] refactor ``_predict_moving_cutoff`` and bugfix, outer ``update_predict_single`` should be called (:pr:`2466`) :user:`fkiraly`
* [BUG] fix ``ThetaForecaster.predict_quantiles`` breaking on ``pd.DataFrame`` input (:pr:`2529`) :user:`fkiraly`
* [BUG] bugfix for default ``_predict_var`` implementation (:pr:`2538`) :user:`fkiraly`
* [BUG] ensure row index names are preserved in hierarchical forecasting when vectorizing (:pr:`2489`) :user:`fkiraly`
* [BUG] Fix type checking error due to pipeline type polymorphism when constructing nested pipelines  (:pr:`2456`) :user:`fkiraly`
* [BUG] fix for ``update_predict`` state handling bug, replace detached cutoff by ``deepcopy`` (:pr:`2557`) :user:`fkiraly`
* [BUG] Fixes the index name dependencies in ``WindowSummarizer`` (:pr:`2567`) :user:`ltsaprounis`
* [BUG] Fix non-compliant output of ``ColumnEnsembleForecaster.pred_quantiles``, ``pred_interval`` (:pr:`2512`) :user:`eenticott-shell`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fixed ``ColumnEnsembleClassifier`` handling of unequal length data (:pr:`2513`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] remove ``alpha`` arg from ``_boxcox``, remove private method dependencies, ensure scipy 1.8.0 compatibility (:pr:`2468`) :user:`fkiraly`
* [BUG] fix random state overwrite in ``MiniRocketMultivariate`` (:pr:`2563`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [BUG] fix accidental overwrite of default method/arg sequences in test scenarios (:pr:`2457`) :user:`fkiraly`

Refactored
~~~~~~~~~~

* [ENH] changed references to ``fit-in-transform`` to ``fit_is_empty`` (:pr:`2494`) :user:`fkiraly`
* [ENH] cleaning up ``_panel._convert`` module (:pr:`2519`) :user:`fkiraly`
* [ENH] Legacy test refactor - move ``test_data_processing``, mtype handling in ``test_classifier_output`` (:pr:`2506`) :user:`fkiraly`
* [ENH] ``MockForecaster`` without logging, ``MockUnivariateForecaster`` clean-up (:pr:`2539`) :user:`fkiraly`
* [ENH] metrics rework part I - output format tests (:pr:`2496`) :user:`fkiraly`
* [ENH] simplify ``load_from_tsfile``, support more mtypes (:pr:`2521`) :user:`fkiraly`
* [ENH] removing dead args and functions post ``_predict_moving_cutoff`` refactor (:pr:`2470`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] upgrade codecov uploader and cleanup coverage reporting (:pr:`2389`) :user:`tarpas`
* [MNT] fix soft dependency handling for ``esig`` imports (:pr:`2414`) :user:`fkiraly`
* [MNT] Make the contrib module private (:pr:`2422`) :user:`MatthewMiddlehurst`
* [MNT] disabling aggressive ``dtw_python`` import message (:pr:`2439`) :user:`KatieBuc`
* [MNT] loosen strict upper bound on ``scipy`` to 1.9.0 (:pr:`2474`) :user:`fkiraly`
* [MNT] Remove accidentally committed prob integration notebook  (:pr:`2476`) :user:`eenticott-shell`
* [MNT] speed up Facebook ``Prophet`` tests (:pr:`2497`) :user:`fkiraly`
* [MNT] Proximity forest faster test param settings (:pr:`2525`) :user:`fkiraly`
* [MNT] Fix tests to prevent all guaranteed ``check_estimator`` failures (:pr:`2411`) :user:`danbartl`
* [MNT] added ``pytest-timeout`` time limit of 10 minutes (:pr:`2532`, :pr:`2541`) :user:`fkiraly`
* [MNT] turn on tests for no state change in ``transform``, ``predict`` (:pr:`2536`) :user:`fkiraly`
* [MNT] switch scipy mirror to anaconda on windows to resolve ``gfortran`` ``FileNotFoundError`` in all CI/CD (:pr:`2561`) :user:`fkiraly`
* [MNT] Add a script to generate changelog in ``rst`` format (:pr:`2449`) :user:`lmmentel`

Documentation
~~~~~~~~~~~~~

* [DOC] Added clustering module to API docs (:pr:`2429`) :user:`aiwalter`
* [DOC] updated datatypes notebook (:pr:`2492`) :user:`fkiraly`
* [DOC] Broken Links in Testing Framework Doc (:pr:`2450`) :user:`Tomiiwa`
* [DOC] remove GSoC announcement from landing page after GSoC deadline (:pr:`2543`) :user:`GuzalBulatova`
* [DOC] fix typo in sktime install instructions, causes "invalid requirement error" if followed verbatim (:pr:`2503`) :user:`Samuel-Oyeneye`


Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`chrisholder`,
:user:`ciaran-g`,
:user:`danbartl`,
:user:`eenticott-shell`,
:user:`fkiraly`,
:user:`GuzalBulatova`,
:user:`juanitorduz`,
:user:`KatieBuc`,
:user:`khrapovs`,
:user:`lmmentel`,
:user:`ltsaprounis`,
:user:`MatthewMiddlehurst`,
:user:`miraep8`,
:user:`ris-bali`,
:user:`Samuel-Oyeneye`,
:user:`tarpas`,
:user:`Tomiiwa`

Version 0.11.2 - 2022-04-11
---------------------------

Fixes
~~~~~

* [BUG] temp workaround for unnamed levels in hierarchical X passed to aggregator (:pr:`2432`)  :user:`fkiraly`
* [BUG] forecasting pipeline dunder fix by (:pr:`2431`)  :user:`fkiraly`
* [BUG] fix erroneous direct passthrough in `ColumnEnsembleForecaster` (:pr:`2436`) :user:`fkiraly`
* [BUG] Incorrect indices returned by make_reduction on hierarchical data fixed by (:pr:`2438`) :user:`danbartl`

Version 0.11.1 - 2022-04-10
---------------------------

Highlights
~~~~~~~~~~

* GSoC 2022 application instructions - apply by Apr 19 for GSoC with sktime! (:pr:`2373`) :user:`lmmentel` :user:`Lovkush-A` :user:`fkiraly`
* enhancements and bugfixes for probabilistic and hierarchical forecasting features introduced in 0.11.0
* reconciliation transformers for hierarchical predictions (:pr:`2287`, :pr:`2292`) :user:`ciaran-g`
* pipeline, tuning and evaluation compabitility for probabilistic forecasting (:pr:`2234`, :pr:`2318`) :user:`eenticott-shell` :user:`fkiraly`
* interface to ``statsmodels`` ``SARIMAX`` (:pr:`2400`) :user:`TNTran92`
* reduction with transform-on-y predictors (e.g., lags, window summaries), and for hierarchical data (:pr:`2396`) :user:`danbartl`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* the ``pd-multiindex`` mtype was relaxed to allow arbitrary level names

Forecasting
^^^^^^^^^^^

* probabilistic forecasting interface now also available for auto-vectorization cases
* probabilistic forecasting interface now compatible with hierarchical forecasting interface

Enhancements
~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] tsf loader to allow specification of return mtype (:pr:`2103`) :user:`ltsaprounis`
* [ENH] relax name rules for multiindex - fixed omission in ``from_multi_index_to_nested`` (:pr:`2384`) :user:`ltsaprounis`

Forecasting
^^^^^^^^^^^

* [ENH] require uniqueness from multiple alpha/coverage in interval/quantile forecasts (:pr:`2326`) :user:`fkiraly`
* [ENH] Adding ``fit`` parameters to ``VAR`` constructor #1850 (:pr:`2304`) :user:`TNTran92`
* [ENH] vectorization for probabilistic forecasting methods that return ``pd.DataFrame`` (:pr:`2355`) :user:`fkiraly`
* [ENH] adding compatibility with probabilistic and hierarchical forecasts to ``ForecastingPipeline`` and ``TransformedTargetForecaster`` (:pr:`2318`) :user:`fkiraly`
* [ENH] Allow ``pd.Timedelta`` values in ``ForecastingHorizon`` (:pr:`2333`) :user:`khrapovs`
* [ENH] probabilistic methods for ``ColumnEnsembleForecaster`` (except `predict_proba`) (:pr:`2356`) :user:`fkiraly`
* [ENH] ``NaiveVariance``: verbose arg and extended docstring (:pr:`2395`) :user:`fkiraly`
* [ENH] Grid search with probabilistic metrics (:pr:`2234`) :user:`eenticott-shell`
* [ENH] wrapper for stream forecasting (``update_predict`` use) to trigger regular refit (:pr:`2305`) :user:`fkiraly`
* [ENH] post-processing in ``TransformedTargetForecaster``, dunder method for (transformed `y`) forecasting pipelines (:pr:`2404`) :user:`fkiraly`
* [ENH] suppressing deprecation messages in ``all_estimators`` estimator retrieval, address dtw import message (:pr:`2418`) :user:`katiebuc`
* [ENH] improved error message in forecasters when receiving an incompatible input (:pr:`2314`) :user:`fkiraly`
* [ENH] ``NaiveVariance``: verbose arg and extended docstring (:pr:`2395`) :user:`fkiraly`
* [ENH] Prohibit incompatible splitter parameters (:pr:`2328`) :user:`khrapovs`
* [ENH] added interface to ``statsmodels`` ``SARIMAX`` (:pr:`2400`) :user:`TNTran92`
* [ENH] extending reducers to hierarchical data, adding transformation (:pr:`2396`) :user:`danbartl`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Faster classifier example parameters (:pr:`2378`) :user:`MatthewMiddlehurst`
* [ENH] `BaseObject.is_composite` utility, relax errors in `BaseClassifier` input checks to warnings for composites (:pr:`2366`) :user:`fkiraly`
* [ENH] Capability inference for transformer and classifier pipelines (:pr:`2367`) :user:`fkiraly`


Transformations
^^^^^^^^^^^^^^^

* [ENH] Implement reconcilers for hierarchical predictions - transformers (:pr:`2287`) :user:`ciaran-g`
* [ENH] Hierarchy aggregation transformer (:pr:`2292`) :user:`ciaran-g`
* [ENH] memory for ``WindowSummarizer`` to enable ``transform`` windows to reach into the ``fit`` time period (:pr:`2325`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] Remove jinja2 version (:pr:`2330`) :user:`aiwalter`
* [ENH] test generation error to raise and not return (:pr:`2298`) :user:`fkiraly`
* [ENH] Remove ``pd.Int64Index`` due to impending deprecation (:pr:`2339`, :pr:`2390`) :user:`khrapovs`
* [MNT] removing unused imports from ``tests._config`` (:pr:`2358`) :user:`fkiraly`
* [ENH] scenarios for hierarchical forecasting and tests for probabilistic forecast methods (:pr:`2359`) :user:`fkiraly`
* [MNT] fixing click/black incompatibility in CI (:pr:`2353`, :pr:`2372`) :user:`fkiraly`
* [ENH] tests for ``check_estimator``` tests passing (:pr:`2408`) :user:`fkiraly`
* [ENH] Fix tests to prevent guaranteed ``check_estimator`` failure (:pr:`2405`) :user:`danbartl`

Refactored
~~~~~~~~~~

* [ENH] remove non-compliant ``fit_params`` kwargs throughout the code base (:pr:`2343`) :user:`fkiraly`
* [ENH] Classification expected output test updates (:pr:`2295`) :user:`MatthewMiddlehurst`
* [ENH] Transformers module full refactor - part III, `panel` module (2nd batch) (:pr:`2253`) :user:`fkiraly`
* [ENH] Transformers module full refactor - part IV, `panel` module (3rd batch) (:pr:`2369`) :user:`fkiraly`
* [ENH] test parameter refactor: ``TSInterpolator`` (:pr:`2342`) :user:`NoaBenAmi`
* [ENH] move "sktime forecaster tests" into ``TestAllForecasters`` class (:pr:`2311`) :user:`fkiraly`
* [ENH] upgrade ``BasePairwiseTransformer`` to use `datatypes` input conversions and checks (:pr:`2363`) :user:`fkiraly`
* [ENH] extend ``_HeterogeneousMetaEstimator`` estimator to allow mixed tuple/estimator list (:pr:`2406`) :user:`fkiraly`
* [MNT] test parameter refactor: forecasting reducers and ``ColumnEnsembleClassifier`` (:pr:`2223`) :user:`fkiraly`
* [ENH] refactoring ``test_all_transformers`` to test class architecture (:pr:`2252`) :user:`fkiraly`

Fixes
~~~~~

Forecasting
^^^^^^^^^^^

* [BUG] fix ``_update`` default for late ``fh`` pass case (:pr:`2362`) :user:`fkiraly`
* [ENH] Extract cached ``ForecastingHorizon`` methods to functions and avoid B019 error (:pr:`2364`) :user:`khrapovs`
* [ENH] ``AutoETS`` prediction intervals simplification (:pr:`2320`) :user:`fkiraly`
* [BUG] fixed ``get_time_index`` for most mtypes (:pr:`2380`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] ``TSInterpolator`` and ``nested_univ`` check fix (:pr:`2259`) :user:`fkiraly`
* [BUG][ENH] WindowSummarizer offset fix, easier lag specification (:pr:`2316`) :user:`danbartl`
* [BUG] ``FeatureUnion`` output column names fixed (:pr:`2324`) :user:`fkiraly`
* [ENH][BUG] fixes and implementations of missing ``inverse_transform`` in transformer compositions (:pr:`2322`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] fix 0.11.0 release note highlights formatting (:pr:`2310`) :user:`fkiraly`
* [DOC] typo fix contsructor -> constructor in extension templates (:pr:`2348`) :user:`fkiraly`
* [DPC] fixed the issue with ``'docs/source/developer_guide/testing_framework.rst'`` (:pr:`2335`) :user:`0saurabh0`
* [DOC] Updated conda installation instructions (:pr:`2365`) :user:`RISHIKESHAVAN`
* [DOC] updated extension templates: link to docs and reference to `check_estimator` (:pr:`2303`) :user:`fkiraly`
* [DOC] Improved docstrings in forecasters (:pr:`2314`) :user:`fkiraly`
* [DOC] Added docstring examples to load data functions (:pr:`2393`) :user:`aiwalter`
* [DOC] Added platform badge to README (:pr:`2398`) :user:`aiwalter`
* [DOC] Add GSoC 2022 landing page and announcement (:pr:`2373`) :user:`lmmentel`
* [DOC] In interval_based_classification example notebook, use multivariate dataset for the multivariate examples (:pr:`1822`) :user:`ksachdeva`

Contributors
~~~~~~~~~~~~

:user:`0saurabh0`,
:user:`aiwalter`,
:user:`ciaran-g`,
:user:`danbartl`,
:user:`eenticott-shell`,
:user:`fkiraly`,
:user:`katiebuc`,
:user:`khrapovs`,
:user:`ksachdeva`,
:user:`lmmentel`,
:user:`ltsaprounis`,
:user:`MatthewMiddlehurst`,
:user:`NoaBenAmi`,
:user:`RISHIKESHAVAN`,
:user:`TNTran92`


Version 0.11.0 - 2022-03-26
---------------------------

Highlights
~~~~~~~~~~

* multivariate forecasting, probabilistic forecasting section in forecasting tutorial (:pr:`2041`) :user:`kejsitake`
* hierarchical & global forecasting: forecaster and transformer interfaces are now compatible with hierarchical data, automatically vectorize over hierarchy levels (:pr:`2110`, :pr:`2115`, :pr:`2219`) :user:`danbartl` :user:`fkiraly`
* probabilistic forecasting: ``predict_var`` (variance forecast) and ``predict_proba`` (full distribution forecast) interfaces; performance metrics for interval and quantile forecasts (:pr:`2100`, :pr:`2130`, :pr:`2232`) :user:`eenticott-shell` :user:`fkiraly` :user:`kejsitake`
* dunder methods for transformer and classifier pipelines: write ``my_trafo1 * my_trafo2`` for pipeline, ``my_trafo1 + my_trafo2`` for ``FeatureUnion`` (:pr:`2090`, :pr:`2251`) :user:`fkiraly`
* Frequently requested: ``AutoARIMA`` from ``statsforecast`` package available as ``StatsforecastAutoARIMA`` (:pr:`2251`) :user:`FedericoGarza`
* for extenders: detailed `"creating sktime compatible estimator" guide <https://www.sktime.org/en/stable/developer_guide/add_estimators.html>`_
* for extenders: simplified extension templates for forecasters and transformers (:pr:`2161`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``sktime`` has a new optional dependency set for deep learning, consisting of ``tensorflow`` and ``tensorflow-probability``
* new soft dependency: ``tslearn`` (required for ``tslearn`` clusterers)
* new soft dependency: ``statsforecast`` (required for ``StatsforecastAutoARIMA``)

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* new ``Hierarchical`` scientific type for hierarchical time series data, with mtype format ``pd_multiindex_hier`` (row-multiindexed series)
* new ``Table`` scientific type for "ordinary" tabular (2D data frame like) data which is not time series or sequential
* multiple mtype formats for the ``Table`` scientific type: ``numpy1D``, ``numpy2D``, ``pd_DataFrame_Table``, ``pd_Series_Table``, ``list_of_dict``
* new ``Proba`` scientific type for distributions and distribution like objects (used in probabilistic forecasting)

Forecasting
^^^^^^^^^^^

* forecasters now also accept inputs of ``Panel`` type (panel and global forecasters) and ``Hierarchical`` type (hierarchical forecasters)
* when a forecaster is given ``Panel`` or ``Hierarchical`` input, and only ``Series`` logic is defined, the forecaster will automatically loop over (series) instances
* when a forecaster is given ``Hierarchical`` input, and only ``Panel`` or ``Series`` logic is defined, the forecaster will automatically loop over (panel) instances
* new probabilistic forecasting interface for probabilistic forecasts:

    * new method ``predict_var(fh, X, cov=False)`` for variance forecasts, returns time series of predictive variances
    * new method ``predict_proba(fh, X, marginal=True)`` for distribution forecasts, returns ``tensorflow`` ``Distribution``

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* dunder method for pipelining classifier and transformers: ``my_trafo1 * my_trafo2 * my_clf`` will create a ``ClassifierPipeline`` (``sklearn`` compatible)

Transformations
^^^^^^^^^^^^^^^

* transformers now also accept inputs of ``Panel`` type (panel and global transformers) and ``Hierarchical`` type (hierarchical transformers)
* when a transformer is given ``Panel`` or ``Hierarchical`` input, and only ``Series`` logic is defined, the transformer will automatically loop over (series) instances
* when a transformer is given ``Hierarchical`` input, and only ``Panel`` or ``Series`` logic is defined, the transformer will automatically loop over (panel) instances
* ``Table`` scientific type is used as output of transformers returning "primitives"
* dunder method for pipelining transformers: ``my_trafo1 * my_trafo2 * my_trafo3`` will create a (single) ``TransformerPipeline`` (``sklearn`` compatible)
* dunder method for ``FeatureUnion`` of transformers: ``my_trafo1 + my_trafo2 + my_trafo3`` will create a (single) ``FeatureUnion`` (``sklearn`` compatible)
* transformer dunder pipeline is compatible with ``sklearn`` transformers, automatically wrapped in a ``TabularToSeriesAdaptor``

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* removed: ``check_is``, renamed to ``check_is_mtype`` (:pr:`1692`) :user:`mloning`

Forecasting
^^^^^^^^^^^

* removed: ``return_pred_int`` argument in forecaster ``predict``, ``fit_predict``, ``update_predict_single``. Replaced by ``predict_interval`` and ``predict_quantiles`` interface.
* deprecated: ``fit-in-predict`` tag is deprecated and renamed to ``fit_is_empty``. Old tag ``fit-in-predict`` can be used until 0.12.0 when it will be removed.
* deprecated: forecasting metrics ``symmetric`` argument default will be changed to ``False`` in 0.12.0. Until then the default is ``True``.

Transformations
^^^^^^^^^^^^^^^
* removed: series transformers no longer accept a `Z` argument - use first argument `X` instead (:pr:`1365`, :pr:`1730`)
* deprecated: ``fit-in-transform`` tag is deprecated and renamed to ``fit_is_empty``. Old tag ``fit-in-transform`` can be used until 0.12.0 when it will be removed.
* deprecated: old location in ``series_as_features`` of ``FeatureUnion``, has moved to ``transformations.compose``. Old location is still importable from until 0.12.0.
* deprecated: ``preserve_dataframe`` argument of ``FeatureUnion``, will be removed in 0.12.0.
* deprecated: old location in ``transformations.series.windows_summarizer`` of ``WindowSummarizer``, has moved to ``transformations.series.summarize``. Old location is still importable from until 0.12.0.

Enhancements
~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*  [ENH] cutoff getter for Series, Panel, and Hierarchical mtypes (:pr:`2115`) :user:`fkiraly`
*  [ENH] Gettimeindex to access index of hierarchical data (:pr:`2110`) :user:`danbartl`
*  [ENH] datatypes support for interval and quantile based probabilistic predictions (:pr:`2130`) :user:`fkiraly`
*  [ENH] sklearn typing util (:pr:`2208`) :user:`fkiraly`
*  [ENH] Relaxing `pd-multiindex` mtype to allow string instance index (:pr:`2262`) :user:`fkiraly`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

*  [ENH] hierarchical mtype generator  (:pr:`2093`) :user:`ltsaprounis`


Clustering
^^^^^^^^^^

*  [ENH] ``tslearn`` added as soft dependency and used to add new clusterers. (:pr:`2048`) :user:`chrisholder`
*  [ENH] Add user option to determine return type in single problem clustering/classification problems (:pr:`2139`) :user:`TonyBagnall`


Distances, kernels
^^^^^^^^^^^^^^^^^^

*  [ENH] minor changes to Lcss distance (:pr:`2119`) :user:`TonyBagnall`
*  [ENH] factory to add 3D capability to all distances exported by distances module (:pr:`2051`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

*  [ENH] Add ``AutoARIMA`` from StatsForecast (:pr:`2251`) :user:`FedericoGarza`
*  [ENH] Naive variance prediction estimator/wrapper (:pr:`1865`) :user:`IlyasMoutawwakil`
*  [ENH] ``predict_proba`` for forecasters, `tensorflow-probability` dependency (:pr:`2100`) :user:`fkiraly`
*  [ENH] Probabilistic forecasting metrics (:pr:`2232`) :user:`eenticott-shell`
*  [ENH] ``_predict_fixed_cutoff`` for ``Hierarchical`` data  (:pr:`2094`) :user:`danbartl`
*  [ENH] Change default of percentage error functions to ``symmetric=False`` (:pr:`2069`) :user:`ciaran-g`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

*  [ENH] Add user option to determine return type in single problem clustering/classification problems (:pr:`2139`) :user:`TonyBagnall`
*  [ENH] TEASER early classification implementation (:pr:`2162`) :user:`MatthewMiddlehurst`
*  [ENH] Classifier pipeline and dunder method (:pr:`2164`) :user:`fkiraly`
*  [ENH] Introduce ``classifier_type`` tag (:pr:`2165`) :user:`MatthewMiddlehurst`
*  [ENH] sklearn model selection tests for classification (:pr:`2180`) :user:`MatthewMiddlehurst`
*  [ENH] Rocket transformer: changed precision to float32 (:pr:`2135`) :user:`RafaAyGar`

Transformations
^^^^^^^^^^^^^^^

*  [ENH] Univariate time series bootstrapping (:pr:`2065`) :user:`ltsaprounis`
*  [ENH] changed `FunctionTransformer._fit` to common signature (:pr:`2205`) :user:`fkiraly`
*  [ENH] Upgrade of ``BaseTransformer`` to use vectorization utility, hierarchical mtype compatibility (:pr:`2219`) :user:`fkiraly`
*  [ENH] ``WindowSummarizer`` to deal with hierarchical data (:pr:`2154`) :user:`danbartl`
*  [ENH] Transformer pipeline and dunder method (:pr:`2090`) :user:`fkiraly`
*  [ENH] Tabular transformer adaptor "fit in transform" parameter (:pr:`2209`) :user:`fkiraly`
*  [ENH] dunder pipelines sklearn estimator support (:pr:`2210`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

*  [ENH] test framework: refactor to test classes (:pr:`2142`) :user:`fkiraly`
*  [ENH] one-stop estimator validity checker (:pr:`1993`) :user:`fkiraly`

Governance
^^^^^^^^^^

*  added :user:`danbartl` to core developer list
*  added :user:`ltsaprounis` to core developer list (:pr:`2236`) :user:`ltsaprounis`


Fixed
~~~~~

*  [BUG] fixed state change caused by `ThetaForecaster.predict_quantiles` (:pr:`2108`) :user:`fkiraly`
*  [BUG] ``_make_hierachical`` is renamed to ``_make_hierarchical`` (typo/bug) issue #2195 (:pr:`2196`) :user:`Vasudeva-bit`
*  [BUG] fix wrong output type of ``PaddingTransformer._transform`` (:pr:`2217`) :user:`fkiraly`
*  [BUG] fixing ``nested_dataframe_has_nans`` (:pr:`2216`) :user:`fkiraly`
*  [BUG] Testing vectorization for forecasters, plus various bugfixes (:pr:`2188`) :user:`fkiraly`
*  [BUG] fixed ``ignores-exogeneous-X`` tag for forecasting reducers (:pr:`2230`) :user:`fkiraly`
*  [BUG] fixing ``STLBootstrapTransformer`` error message and docstrings (:pr:`2260`) :user:`fkiraly`
*  [BUG] fix conversion interval->quantiles in `BaseForecaster`, and fix `ARIMA.predict_interval` (:pr:`2281`) :user:`fkiraly`
*  [DOC] fix broken link to CoC (:pr:`2104`) :user:`mikofski`
*  [BUG] Fix windows bug with index freq in ``VectorizedDF.__getitem__`` (:pr:`2279`) :user:`ltsaprounis`
*  [BUG] fixes duplication of Returns section in ``_predict_var`` docstring (:pr:`2306`) :user:`fkiraly`
*  [BUG] Fixed bug with ``check_pdmultiindex_panel`` (:pr:`2092`) :user:`danbartl`
*  [BUG] Fixed crash of kmeans, medoids when empty clusters are generated (:pr:`2060`) :user:`chrisholder`
*  [BUG] Same cutoff typo-fix (:pr:`2193`) :user:`cdahlin`
*  [BUG] Addressing doc build issue due to failed soft dependency imports (:pr:`2170`) :user:`fkiraly`

*  Deprecation handling: sklearn 1.2 deprecation warnings (:pr:`2190`) :user:`hmtbgc`
*  Deprecation handling: Replacing normalize by use of StandardScaler (:pr:`2167`) :user:`KishenSharma6`


Documentation
~~~~~~~~~~~~~

*  [DOC] forecaster tutorial: multivariate forecasting, probabilistic forecasting (:pr:`2041`) :user:`kejsitake`
*  [DOC] New estimator implementation guide (:pr:`2186`) :user:`fkiraly`
*  [DOC] simplified extension templates for transformers and forecasters (:pr:`2161`) :user:`fkiraly`
*  [DOC] contributing page: concrete initial steps (:pr:`2227`) :user:`fkiraly`
*  [DOC] adding "troubleshooting" link in sktime installation instructions (:pr:`2121`) :user:`eenticott-shell`
*  [DOC] enhance distance doc strings (:pr:`2122`) :user:`TonyBagnall`
*  [DOC] updated soft dependency docs with two tier check (:pr:`2182`) :user:`fkiraly`
*  [DOC] replace gitter mentions by appropriate links, references (:pr:`2187`) :user:`TonyBagnall`
*  [DOC] updated the environments doc with python version for sktime, added python 3.9 (:pr:`2199`) :user:`Vasudeva-bit`
*  [DOC] Replaced youtube link with recent PyData Global (:pr:`2191`) :user:`aiwalter`
*  [DOC] extended & cleaned docs on dependency handling (:pr:`2189`) :user:`fkiraly`
*  [DOC] migrating mentoring form to sktime google docs (:pr:`2222`) :user:`fkiraly`
*  [DOC] add scitype/mtype register pointers to docstrings in datatypes (:pr:`2160`) :user:`fkiraly`
*  [DOC] improved docstrings for HIVE-COTE v1.0 (:pr:`2239`) :user:`TonyBagnall`
*  [DOC] typo fix and minor clarification in estimator implementation guide (:pr:`2241`) :user:`fkiraly`
*  [DOC] numpydoc compliance fix of simple forecasting extension template (:pr:`2284`) :user:`fkiraly`
*  [DOC] typos in ``developer_guide.rst`` (:pr:`2131`) :user:`theanorak`
*  [DOC] fix broken link to CoC (:pr:`2104`) :user:`mikofski`
*  [DOC] minor update to tutorials (:pr:`2114`) :user:`ciaran-g`
*  [DOC] various minor doc issues (:pr:`2168`) :user:`aiwalter`

Maintenance
~~~~~~~~~~~

*  [MNT] Update release drafter (:pr:`2096`) :user:`lmmentel`
*  speed up EE tests and ColumnEnsemble example (:pr:`2124`) :user:`TonyBagnall`
*  [MNT] add xfails in `test_plotting` until #2066 is resolved (:pr:`2144`) :user:`fkiraly`
*  [MNT] add skips to entirety of `test_plotting` until #2066 is resolved (:pr:`2147`) :user:`fkiraly`
*  [ENH] improved `deep_equals` return message if `dict`s are discrepant (:pr:`2107`) :user:`fkiraly`
*  [BUG] Addressing doc build issue due to failed soft dependency imports (:pr:`2170`) :user:`fkiraly`
*  [ENH] extending `deep_equals` for `ForecastingHorizon` (:pr:`2225`) :user:`fkiraly`
*  [ENH] unit tests for `deep_equals` utility (:pr:`2226`) :user:`fkiraly`
*  [MNT] Faster docstring examples - `ForecastingGridSearchCV`, `MultiplexForecaster` (:pr:`2229`) :user:`fkiraly`
*  [BUG] remove test for StratifiedGroupKFold (:pr:`2244`) :user:`TonyBagnall`
*  [ENH] Classifier type hints (:pr:`2246`) :user:`MatthewMiddlehurst`
*  Updated pre-commit link and also grammatically updated Coding Style docs (:pr:`2285`) :user:`Tomiiwa`
*  Update .all-contributorsrc (:pr:`2286`) :user:`Tomiiwa`
*  [ENH] Mock estimators and mock estimator generators for testing (:pr:`2197`) :user:`ltsaprounis`
*  [MNT] Deprecation removal 0.11.0 (:pr:`2271`) :user:`fkiraly`
*  [BUG] fixing pyproject and jinja2 CI failures (:pr:`2299`) :user:`fkiraly`
*  [DOC] Update PULL_REQUEST_TEMPLATE.md so PRs should start with [ENH], [DOC] or [BUG] in title (:pr:`2293`) :user:`aiwalter`
*  [MNT] add skips in `test_plotting` until #2066 is resolved (:pr:`2146`) :user:`fkiraly`

Refactored
~~~~~~~~~~

*  [ENH] Clustering experiment save results formatting (:pr:`2156`) :user:`TonyBagnall`
*  [ENH] replace ``np.isnan`` by ``pd.isnull`` in ``datatypes`` (:pr:`2220`) :user:`fkiraly`
*  [ENH] renamed ``fit-in-transform`` and ``fit-in-predict`` to ``fit_is_empty`` (:pr:`2250`) :user:`fkiraly`
*  [ENH] refactoring `test_all_classifiers` to test class architecture (:pr:`2257`) :user:`fkiraly`
*  [ENH] test parameter refactor: all classifiers (:pr:`2288`) :user:`MatthewMiddlehurst`
*  [ENH] test paraneter refactor: ``Arsenal`` (:pr:`2273`) :user:`dionysisbacchus`
*  [ENH] test parameter refactor: ``RocketClassifier`` (:pr:`2166`) :user:`dionysisbacchus`
*  [ENH] test parameter refactor: ``TimeSeriesForestClassifier`` (:pr:`2277`) :user:`lielleravid`
*  [ENH] ``FeatureUnion`` refactor - moved to ``transformations``, tags, dunder method (:pr:`2231`) :user:`fkiraly`
*  [ENH] ``AutoARIMA`` from ``statsforecast`` to ``StatsForecastAutoARIMA`` (:pr:`2272`) :user:`FedericoGarza`

Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`cdahlin`,
:user:`chrisholder`,
:user:`ciaran-g`,
:user:`danbartl`,
:user:`dionysisbacchus`,
:user:`eenticott-shell`,
:user:`FedericoGarza`,
:user:`fkiraly`,
:user:`hmtbgc`,
:user:`IlyasMoutawwakil`,
:user:`kejsitake`,
:user:`KishenSharma6`,
:user:`lielleravid`,
:user:`lmmentel`,
:user:`ltsaprounis`,
:user:`MatthewMiddlehurst`,
:user:`mikofski`,
:user:`RafaAyGar`,
:user:`theanorak`,
:user:`Tomiiwa`,
:user:`TonyBagnall`,
:user:`Vasudeva-bit`,


[0.10.1] - 2022-02-20
---------------------

Highlights
~~~~~~~~~~

* This release is mainly a maintenance patch which upper bounds ``scipy<1.8.0`` to prevent bugs due to interface changes in ``scipy``.
* Once ``sktime`` is compatible with ``scipy 1.8.0``, the upper bound will be relaxed
* New forecaster: ``STLForecaster`` (:pr:`1963`) :user:`aiwalter`
* New transformer: lagged window summarizer transformation (:pr:`1924`) :user:`danbartl`
* Loaders for ``.tsf`` data format (:pr:`1934`) :user:`rakshitha123`

Dependency changes
~~~~~~~~~~~~~~~~~~
* Introduction of bound ``scipy<1.8.0``, to prevent bugs due to interface changes in ``scipy``
* Once ``sktime`` is compatible with ``scipy 1.8.0``, the upper bound will be relaxed

Added
~~~~~

Documentation
^^^^^^^^^^^^^

* [DOC] improvements to the forecasting tutorial (:pr:`1834)` :user:`baggiponte`
* [DOC] Fix wrong conda command to install packages (:pr:`1973`) :user:`schettino72``
* [DOC] Removed gitter from README (:pr:`2025`) :user:`aiwalter`
* [DOC] Fix minor documentation issues (:pr:`2035`) :user:`Saransh-cpp`
* [DOC] Fixed link from README to classification notebook (:pr:`2042`) :user:`Rubiel1`
* [DOC] Added merlion as related software (:pr:`2050`) :user:`aiwalter`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Added loaders for ``.tsf`` data format (:pr:`1934`) :user:`rakshitha123`
* [ENH] Added ``.tsf`` dataset for unit testing (:pr:`1996`) :user:`rakshitha123`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``convert`` store reset/freeze behaviour & fix of bug 1976 (:pr:`1977`) :user:`fkiraly`
* [ENH] new ``Table`` mtypes: ``pd.Series`` based, ``list`` of ``dict`` (as used in bag of words transformers) (:pr:`2076`) :user:`fkiraly``

Forecasting
^^^^^^^^^^^

* [ENH] Added ``STLForecaster`` (:pr:`1963`) :user:`aiwalter`
* [ENH] moving forecaster test params from ``_config`` into classes - all forecasters excluding reduction (:pr:`1902`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Lagged window summarizer transformation (:pr:`1924`) :user:`danbartl`

Maintenance
~~~~~~~~~~~

* [MNT] Update wheels CI/CD workflow after dropping C extensions and Cython (:pr:`1972`) :user:`lmmentel`
* [MNT] Rename classification notebooks (:pr:`1980`) :user:`TonyBagnall`
* [MNT] hotfix: scipy upper bound from ``1.8.0`` (:pr:`1995`) :user:`fkiraly`
* [MNT] replace deprecated ``np.str`` and ``np.float`` (:pr:`1997`) :user:`fkiraly`
* [MNT] Remove ``pytest-xdist`` from CI-CD, un-skip `test_multiprocessing_idempotent` (:pr:`2004`) :user:`fkiraly`
* [MNT] Changing function names in datatypes check to lower_snake_case (:pr:`2014`) :user:`chicken-biryani`
* [MNT] verbose output for ``linux`` and ``mac`` tests (:pr:`2045`) :user:`Saransh-cpp`
* [MNT] GitHub Actions: cancel old but running workflows of a PR when pushing again (:pr:`2063`) :user:`RishiKumarRay`

Fixed
~~~~~
* [BUG] remove MrSEQL notebook in docs (:pr:`1974`) :user:`TonyBagnall`
* [BUG] fix import on clustering extension template (:pr:`1978`) :user:`schettino72`
* [BUG] HC2 component bugfixes (:pr:`2020`) :user:`MatthewMiddlehurst`
* [BUG] Fix a bug in ``PeriodIndex`` arithmetic (:pr:`1981`) :user:`khrapovs`
* [BUG] fix conversion of ``nested_univ`` to ``pd-multiindex`` mtype if series have names (:pr:`2000`) :user:`fkiraly`
* [BUG] ``MiniRocket`` to comply with sklearn init specification, fix ``random_state`` modification in ``__init__`` (:pr:`2027`) :user:`fkiraly`
* [BUG] naive forecaster window error (:pr:`2047`) :user:`eenticott-shell`
* [BUG] Fix silent bug in ``ColumnsEnsembleForecaster._predict`` (:pr:`2083`) :user:`aiwalter`

Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`baggiponte`,
:user:`chicken-biryani`,
:user:`danbartl`,
:user:`eenticott-shell`,
:user:`fkiraly`,
:user:`khrapovs`,
:user:`lmmentel`,
:user:`MatthewMiddlehurst`,
:user:`rakshitha123`,
:user:`RishiKumarRay`,
:user:`Rubiel1`,
:user:`Saransh-cpp`,
:user:`schettino72`,


[0.10.0] - 2022-02-02
---------------------

Highlights
~~~~~~~~~~

* ``sktime`` now supports python 3.7-3.9. Python 3.6 is no longer supported, due to end of life. Last ``sktime`` version to support python 3.6 was 0.9.0.
* ``sktime`` now supports, and requires, ``numpy>=1.21.0`` and ``statsmodels>=0.12.1``
* overhaul of docs for installation and first-time developers (:pr:`1707`) :user:`amrith-shell`
* all probabilistic forecasters now provide ``predict_interval`` and ``predict_quantiles`` interfaces
  (:pr:`1842`, :pr:`1874`, :pr:`1879`, :pr:`1910`, :pr:`1961`) :user:`fkiraly` :user:`k1m190r` :user:`kejsitake`
* new transformation based pipeline classifiers (:pr:`1721`) :user:`MatthewMiddlehurst`
* developer install for ``sktime`` no longer requires C compilers and ``cython`` (:pr:`1761`, :pr:`1847`, :pr:`1932`, :pr:`1927`) :user:`TonyBagnall`
* CI/CD moved completely to GitHub actions (:pr:`1620`, :pr:`1920`) :user:`lmmentel`


Dependency changes
~~~~~~~~~~~~~~~~~~
* ``sktime`` now supports ``python`` 3.7-3.9 on windows, mac, and unix-based systems
* ``sktime`` now supports, and requires, ``numpy>=1.21.0`` and ``statsmodels>=0.12.1``
* ``sktime`` ``Prophet`` interface now uses ``prophet`` instead of deprecated ``fbprophet``
* developer install for ``sktime`` no longer requires C compilers and ``cython``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

New probabilistic forecasting interface for quantiles and predictive intervals:

* for all forecasters with probabilistic forecasting capability, i.e., ``capability:pred_int`` tag
* new method ``predict_interval(fh, X, coverage)`` for interval forecasts
* new method ``predict_quantiles(fh, X, alpha)`` for quantile forecasts
* both vectorized in ``coverage``, ``alpha`` and applicable to multivariate forecasting
* old ``return_pred_int`` interface is deprecated and will be removed in 0.11.0
* see forecaster base API and forecaster extension template

Convenience method to return residuals:

* all forecasters now have a method ``predict_residuals(y, X, fh)``
* if ``fh`` is not passed, in-sample residuals are computed

Transformations
^^^^^^^^^^^^^^^

Base interface refactor rolled out to series transformers (:pr:`1790`, :pr:`1795`):

* ``fit``, ``transform``, ``fit_transform`` now accept both ``Series`` and ``Panel`` as argument
* if ``Panel`` is passed to a series transformer, it is applied to all instances
* all transformers now have signature ``transform(X, y=None)`` and ``inverse_transform(X, y=None)``. This is enforced by the new base interface.
* ``Z`` (former first argument) aliases ``X`` until 0.11.0 in series transformers, will then be removed
* ``X`` (former second argument) was not used in those transformers, was changed to ``y``
* see transformer base API and transformer extension template

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* deprecated, scheduled for removal in 0.11.0: ``check_is`` renamed to ``check_is_mtype``, ``check_is`` to be removed in 0.11.0 (:pr:`1692`) :user:`mloning`

Forecasting
^^^^^^^^^^^

* deprecated, scheduled for removal in 0.11.0: ``return_pred_int`` argument in forecaster ``predict``, ``fit_predict``, ``update_predict_single``. Replaced by ``predict_interval`` and ``predict_quantiles`` interface.


Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Removed: ``MrSEQL`` time series classifier (:pr:`1548`) :user:`TonyBagnall`
* Removed ``RISF`` and shapelet classifier (:pr:`1907`) :user:`TonyBagnall`
* ``data.io`` module moved to `datasets` (:pr:`1907`) :user:`TonyBagnall`

Transformations
^^^^^^^^^^^^^^^

* deprecated, scheduled for removal in 0.11.0: series transformers will no longer accept a `Z` argument - first argument `Z` replaced by `X` (:pr:`1365`, :pr:`1730`)

Added
~~~~~

Documentation
^^^^^^^^^^^^^

* [DOC] updates to forecaster and transformer extension template (:pr:`1774`, :pr:`1853`) :user:`fkiraly`
* [DOC] Update Prophet and ETS docstrings (:pr:`1698`) :user:`mloning`
* [DOC] updated ``get_test_params`` extension template docs regarding imports	(:pr:`1811`) :user:`fkiraly`
* [DOC] reformatted the documentation structure (:pr:`1707`) :user:`amrith-shell`
* [DOC] Added VAR to API docs (:pr:`1964`) :user:`aiwalter`
* [DOC] Updated classification notebook (:pr:`1885`) :user:`TonyBagnall`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``check_is_scitype``, cleaning up dists_kernels input checks/conversions (:pr:`1704`) :user:`fkiraly`
* [ENH] `Table` scitype and refactor of ``convert`` module (:pr:`1745`) :user:`fkiraly`
* [ENH] estimator scitype utility (:pr:`1838`) :user:`fkiraly`
* [ENH] experimental: hierarchical time series scitype	hierarchical_scitype (:pr:`1786`) :user:`fkiraly`
* [ENH] upgraded ``mtype_to_scitype`` to list-like args (:pr:`1807`) :user:`fkiraly`
* [ENH] ``check_is_mtype`` to return scitype (:pr:`1789`) :user:`fkiraly`
* [ENH] vectorization/iteration utility for `sktime` time series formats (:pr:`1806`) :user:`fkiraly`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Update dataset headers (:pr:`1752`) :user:`tonybagnall`
* [ENH] Classification dataset tidy-up (:pr:`1785`) :user:`tonybagnall`
* [ENH] polymorphic data loader in contrib (:pr:`1840`) :user:`tonybagnall`
* [ENH] move functions and tests from `utils/data_io` to `datasets/_data_io` (:pr:`1777`) :user:`tonybagnall`

Clustering
^^^^^^^^^^

* [ENH] Clustering module refactor (:pr:`1864`) :user:`chrisholder`
* [ENH] ``fit`` repeated initialization in Lloyd's algorithm (:pr:`1897`) :user:`chrisholder`


Distances, kernels
^^^^^^^^^^^^^^^^^^

* [ENH] Composable distances interface prototype for numba distance module (:pr:`1858`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] Scaled Logit Transformer (:pr:`1913`, :pr:`1965`) :user:`ltsaprounis`.
* [ENH] add ``fit`` parameters to `statsmodels` Holt-Winters exponential smoothing interface (:pr:`1849`) :user:`fkiraly`
* [ENH] Add ``predict_quantiles`` to FBprophet (:pr:`1910`) :user:`kejsitake`
* [ENH] Add ```predict_quantiles`` to ets, pmdarima adapter (:pr:`1874`) :user:`kejsitake`
* [ENH] Defaults for ``_predict_interval`` and ``_predict_coverage`` (:pr:`1879`, :pr:`1961`) :user:`fkiraly`
* [ENH] refactored column ensemble forecaster (:pr:`1764`) :user:`Aparna-Sakshi`
* [ENH] Forecaster convenience method to return forecast residuals (:pr:`1770`) :user:`fkiraly`
* [ENH] Update extension template for predict_quantiles (:pr:`1780`) :user:`kejsitake`
* [ENH] Prediction intervals refactor: BATS/TBATS; bugfix for #1625; base class updates on ``predict_quantiles`` (:pr:`1842`) :user:`k1m190r`
* [ENH] Change ``_set_fh`` to a ``_check_fh`` that returns `self._fh` (:pr:`1823`) :user:`fkiraly`
* [ENH] Generalize splitters to accept timedeltas (equally spaced) (:pr:`1758`) :user:`khrapovs`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] New transformation based pipeline classifiers (:pr:`1721`) :user:`MatthewMiddlehurst`
* [ENH] ``FreshPRINCE`` params moved from `_config` into estimator (:pr:`1944`) :user:`fkiraly`
* [ENH] user selected return for classification problems data loading functions (:pr:`1799`) :user:`tonybagnall`
* [ENH] TSC refactor: ``compose`` sub-module (:pr:`1852`) :user:`tonybagnall`
* [ENH] TSC refactor: TSC column ensemble (:pr:`1859`) :user:`tonybagnall`
* [ENH] TSC refactor: TSF, RSF (:pr:`1851`) :user:`tonybagnall`
* [ENH] Replace C extensions and Cython with numba based distance calculations (:pr:`1761`, :pr:`1847`, :pr:`1932`, :pr:`1927`) :user:`TonyBagnall`.
* [ENH] introduce msm distance and adapt KNN classifier to use it (:pr:`1926`) :user:`tonybagnall`
* [ENH] Efficiency improvements for HC2	interval_speedup (:pr:`1754`) :user:`MatthewMiddlehurst`
* [ENH] classifier tests: removes replace_X_y, comments, and add contracting tests (:pr:`1800`) :user:`MatthewMiddlehurst`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Transformers module full refactor - part I, ``series`` module	(:pr:`1795`) :user:`fkiraly`
* [ENH] Transformer base class DRY-ing, and ``inverse_transform``	(:pr:`1790`) :user:`fkiraly`
* [ENH] transformer base class to allow multivariate output if input is always univariate (:pr:`1706`) :user:`fkiraly`

Testing module
^^^^^^^^^^^^^^

* [ENH] Test refactor with scenarios (:pr:`1833`) :user:`fkiraly`
* [ENH] Test scenarios for advanced testing	(:pr:`1819`) :user:`fkiraly`
* [ENH] pytest conditional fixtures	(:pr:`1839`) :user:`fkiraly`
* [ENH] Test enhacements documentation (:pr:`1922`) :user:`fkiraly`
* [ENH] split tests in series_as_features into classification and regression (:pr:`1959`) :user:`tonybagnall`
* [ENH] Testing for metadata returns of ``check_is_mtype`` (:pr:`1748`) :user:`fkiraly`
* [ENH] Extended deep_equals, with precise indication of why equality fails	(:pr:`1844`) :user:`fkiraly`
* [ENH] test for ``test_create_test_instances_and_names``	fixture generation method (:pr:`1829`) :user:`fkiraly`
* [ENH] Utils module housekeeping varia	utils-housekeeping (:pr:`1820`) :user:`fkiraly`
* [ENH] Extend testing framework to test multiple instance fixtures per estimator (:pr:`1732`) :user:`fkiraly`

Governance
^^^^^^^^^^

* new CC composition, updated codeowners (:pr:`1796`)
* Add core developer: :user:`lmmentel` (:pr:`1836`)
* updated core developer list (:pr:`1841`) :user:`sumit-158`

Maintenance
^^^^^^^^^^^

* [MNT] Switch the extra dependency from `fbprophet` to `prophet` (:pr:`1958`) :user:`lmmentel`
* [MNT] Updated code dependency version, i.e. `numpy` and `statsmodels` to reduce dependency conflicts (:pr:`1921`) :user:`lmmentel`
* [MNT] Move all the CI/CD worfklows over to github actions and drop azure pipelines and appveyor (:pr:`1620`, :pr:`1920`) :user:`lmemntel`
* [MNT] Refactor legacy test config	(:pr:`1792`) :user:`lmmentel`
* [FIX] Add missing init files (:pr:`1695`) :user:`mloning`
* [MNT] Add shellcheck to pre-commit (:pr:`1703`) :user:`mloning`
* [MNT] Remove assign-contributor workflow (:pr:`1702`) :user:`mloning`
* [MNT] Fail CI on missing init files (:pr:`1699`) :user:`mloning`
* [ENH] replace deprecated ``np.int``, ``np.float`` (:pr:`1734`) :user:`fkiraly`
* [MNT] Correct the bash error propagation for running notebook examples (:pr:`1816`) :user:`lmmentel`

Fixed
~~~~~

* [DOC] Fixed a typo in transformer extension template (:pr:`1901`) :user:`rakshitha123`
* [DOC] Fix typo in Setting up a development environment section (:pr:`1872`) :user:`shubhamkarande13`
* [BUG] Fix incorrect "uses `X`" tag for ARIMA and ``TrendForecaster`` (:pr:`1895`) :user:`ngupta23`
* [BUG] fix error when concatenating train and test (:pr:`1892`) :user:`tonybagnall`
* [BUG] Knn bugfix to allow GridsearchCV and usage with column ensemble (:pr:`1903`) :user:`tonybagnall`
* [BUG] Fixes various bugs in DrCIF, STSF, MUSE, Catch22 (:pr:`1869`) :user:`MatthewMiddlehurst`
* [BUG] fixing mixup of internal variables in detrender	(:pr:`1863`) :user:`fkiraly`
* [BUG] transformer base class changes and bugfixes	(:pr:`1855`) :user:`fkiraly`
* [BUG] fixed erroneous index coercion in ``convert_align_to_align_loc`` (:pr:`1911`) :user:`fkiraly`
* [BUG] bugfixes for various bugs discovered in scenario testing (:pr:`1846`) :user:`fkiraly`
* [BUG] 1523 fixing ``ForecastHorizon.to_absolute`` for freqs with anchorings	(:pr:`1830`) :user:`eenticott-shell`
* [BUG] remove duplicated input checks from ``BaseClassifier.score`` (:pr:`1813`) :user:`fkiraly`
* [BUG] fixed mtype return field in ``check_is_scitype`` (:pr:`1805`) :user:`fkiraly`
* [BUG] fix fh -> self.fh in ``predict_interval`` and ``predict_quantiles``	(:pr:`1775`) :user:`fkiraly`
* [BUG] fix incorrect docstrings and resolving confusion unequal length/spaced in panel metadata inference (:pr:`1768`) :user:`fkiraly`
* [BUG] hotfix for bug when passing multivariate `y` to boxcox transformer (:pr:`1724`) :user:`fkiraly`
* [BUG] fixes CIF breaking with CIT, added preventative test (:pr:`1709`) :user:`MatthewMiddlehurst`
* [BUG] Correct the `examples/catch22.ipynb` call to ``transform_single_feature``	(:pr:`1793`) :user:`lmmentel`
* [BUG] Fixes prophet bug concerning the internal change of exogenous X	 (:pr:`1711`) :user:`kejsitake`
* [BUG] Fix DeprecationWarning of ``pd.Series`` in sktime/utils/tests/test_datetime.py:21	(:pr:`1743`) :user:`khrapovs`
* [BUG] bugfixes in ``BaseClassifier``, updated base class docstrings (:pr:`1804`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`amrith-shell`,
:user:`Aparna-Sakshi`,
:user:`AreloTanoh`,
:user:`chrisholder`,
:user:`eenticott-shell`,
:user:`fkiraly`,
:user:`k1m190r`,
:user:`kejsitake`,
:user:`khrapovs`,
:user:`lmmentel`,
:user:`ltsaprounis`,
:user:`MatthewMiddlehurst`,
:user:`MrPr3ntice`,
:user:`mloning`,
:user:`ngupta23`,
:user:`rakshitha123`,
:user:`RNKuhns`,
:user:`shubhamkarande13`,
:user:`sumit-158`,
:user:`TonyBagnall`,

[0.9.0] - 2021-12-08
--------------------

Highlights
~~~~~~~~~~

* frequently requested: AutoARIMA ``get_fitted_params`` access for fitted order and seasonal order (:pr:`1641`) :user:`AngelPone`
* Numba distance module - efficient time series distances (:pr:`1574`) :user:`chrisholder`
* Transformers base interface refactor - default vectorization to panel data :user:`fkiraly`
* new experimental module: Time series alignment, dtw-python interface (:pr:`1264`) :user:`fkiraly`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``check_is`` renamed to ``check_is_mtype``, ``check_is`` to be deprecated in 0.10.0 (:pr:`1692`) :user:`mloning`


Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* time series classifiers now accept 2D ``np.ndarray`` by conversion to 3D rather than throwing exception (:pr:`1604`) :user:`TonyBagnall`

Transformations
^^^^^^^^^^^^^^^

Base interface refactor (:pr:`1365`, :pr:`1663`, :pr:`1706`):

* ``fit``, ``transform``, ``fit_transform`` now accept both ``Series`` and ``Panel`` as argument
* if ``Panel`` is passed to a series transformer, it is applied to all instances
* all transformers now use `X` as their first argument, `y` as their second argument. This is enforced by the new base interface.
* This was inconsistent previously between types of transformers: the series-to-series transformers were using `Z` as first argument, `X` as second argument.
* `Z` (former first argument) aliases `X` until 0.10.0 in series transformers, will then be deprecated
* `X` (former second argument) was not used in those transformers where it changed to `y`
* see new transformer extension template
* these changes will gradually be rolled out to all transformers through 0.9.X versions


New deprecations for 0.10.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``check_is`` renamed to ``check_is_mtype``, ``check_is`` to be deprecated in 0.10.0 (:pr:`1692`) :user:`mloning`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* MrSEQL time series classifier (:pr:`1548`) :user:`TonyBagnall`

Transformations
^^^^^^^^^^^^^^^

* series transformers will no longer accept a `Z` argument - first argument `Z` replaced by `X` (:pr:`1365`)

Added
~~~~~

Documentation
^^^^^^^^^^^^^

* [DOC] Windows installation guide for sktime development with Anaconda and PyCharm by (:pr:`1640`) :user:`jasonlines`
* [DOC] Update installation.rst (:pr:`1636`) :user:`MrPr3ntice`
* [DOC] additions to forecaster extension template (:pr:`1535`) :user:`fkiraly`
* [DOC] Add missing classes to API reference (:pr:`1571`) :user:`RNKuhns`
* [DOC] Add toggle button to make examples easy to copy (:pr:`1572`) :user:`RNKuhns`
* [DOC] Update docs from roadmap planning sessions (:pr:`1527`) :user:`mloning`
* [DOC] STLTransformer docstring and attribute (:pr:`1611`) :user:`aiwalter`
* [DOC] typos in user documentation (:pr:`1671`) :user:`marcio55afr`
* [DOC] Add links to estimator overview to README (:pr:`1691`) :user:`mloning`
* [DOC] Update Time series forest regression docstring (:pr:`800`) :user:`thayeylolu`
* [DOC] fix docstring in Feature Union (:pr:`1470`) :user:`AreloTanoh`
* [DOC] Update Prophet and ETS docstrings (:pr:`1698`) :user:`mloning`
* [DOC] Added new contributors (:pr:`1602` :pr:`1559`) :user:`Carlosbogo` :user:`freddyaboulton`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] added ``check_is_scitype`` for scitype checks, cleaning up dists_kernels input checks/conversions (:pr:`1704`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] Auto-ETS checks models to select from based on non-negativity of data (:pr:`1615`) :user:`chernika158`
* [DOC] meta-tuning examples for docstring of ``ForecastingGridSearchCV`` (:pr:`1656`) :user:`aiwalter`

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [ENH] new module: time series alignment; alignment distances (:pr:`1264`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Classifier test speed ups (:pr:`1599`) :user:`MatthewMiddlehurst`
* [ENH] Experiments tidy-up by (:pr:`1619`) :user:`TonyBagnall`
* [ENH] MiniRocket and MultiRocket as options for RocketClassifier (:pr:`1637`) :user:`MatthewMiddlehurst`
* [ENH] Updated classification base class typing (:pr:`1633`) :user:`chrisholder`
* [ENH] Integrate multi-rocket (:pr:`1567`) :user:`fstinner`
* TSC refactor: Interval based classification package(:pr:`1583`) :user:`MatthewMiddlehurst`
* TSC refactor: Distance based classification package (:pr:`1584`) :user:`MatthewMiddlehurst`
* TSC refactor: Feature based classification package (:pr:`1545`) :user:`MatthewMiddlehurst`


Time series distances
^^^^^^^^^^^^^^^^^^^^^

* [ENH] Numba distance module - efficient time series distances (:pr:`1574`) :user:`chrisholder`
* [ENH] Distance metric refactor (:pr:`1664`) :user:`chrisholder`

Governance
^^^^^^^^^^

* eligibility and end of tenure clarification (:pr:`1573`) :user:`fkiraly`

Maintenance
^^^^^^^^^^^

* [MNT] Update release script (:pr:`1562`) :user:`mloning`
* [MNT] Delete release-drafter.yml (:pr:`1561`) :user:`mloning`
* [MNT] Fail CI on missing init files (:pr:`1699`) :user:`mloning`


Fixed
~~~~~

Estimator registry
^^^^^^^^^^^^^^^^^^

* [BUG] Fixes to registry look-up, test suite for registry look-up (:pr:`1648`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] Facebook prophet side effects on exogenous data X (:pr:`1711`) :user:`kejsitake`
* [BUG] fixing bug for ``_split``, accidental removal of `pandas.Index` support (:pr:`1582`) :user:`fkiraly`
* [BUG] Fix ``convert`` and ``_split`` for Numpy 1D input (:pr:`1650`) :user:`fkiraly`
* [BUG] issue with update_y_X when we refit forecaster by (:pr:`1595`) :user:`ltsaprounis`

Performance metrics, evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] missing clone in `evaluate` by (:pr:`1670`) :user:`ltsaprounis`
* [BUG] fixing display via `repr` (:pr:`1566`) :user:`RNKuhns`
* [BUG] Fix `test_wilcoxon` compatibility between pandas versions (:pr:`1653`) :user:`lmmentel`


Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [BUG] missing alignment fixtures (:pr:`1661`) :user:`fkiraly`


Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fixes :issue:`1234` (:pr:`1600`) :user:`Carlosbogo`
* [BUG] load from UCR fix (:pr:`1610`) :user:`TonyBagnall`
* [BUG] TimeSeriesForest Classifier Fix (:pr:`1588`) :user:`OliverMatthews`
* [BUG] fix parameter mismatch in ShapeDTW by (:pr:`1638`) :user:`TonyBagnall`

Transformations
^^^^^^^^^^^^^^^

* [BUG] Fix Imputer. Added Imputer tests (:pr:`1666`) :user:`aiwalter`
* [BUG] Fix `ColumnwiseTransformer` example (:pr:`1681`) :user:`mloning`
* [BUG] Fix `FeatureUnion` test failure (:pr:`1665`) :user:`lmmentel`
* [BUG] Refactor the `_diff_transform` function to be compatible with pandas 1.3.4 (:pr:`1644`) :user:`lmmentel`


Maintenance
^^^^^^^^^^^

* [MNT] fixing version clask between Numba and numpy (:pr:`1623`) :user:`TonyBagnall`
* [MNT] Fix appveyor (:pr:`1669`) :user:`mloning`
* [MNT] testing framework: replace `time.time` with time.perf_counter (:pr:`1680`) :user:`mloning`
* [MNT] Add missing init files (:pr:`1695`) :user:`mloning`


Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`AngelPone`,
:user:`AreloTanoh`,
:user:`Carlosbogo`,
:user:`chernika158`,
:user:`chrisholder`,
:user:`fstinner`,
:user:`fkiraly`,
:user:`freddyaboulton`,
:user:`kejsitake`,
:user:`lmmentel`,
:user:`ltsaprounis`,
:user:`MatthewMiddlehurst`,
:user:`marcio55afr`,
:user:`MrPr3ntice`,
:user:`mloning`,
:user:`OliverMatthews`,
:user:`RNKuhns`,
:user:`thayeylolu`,
:user:`TonyBagnall`,


Full changelog
~~~~~~~~~~~~~~
https://github.com/alan-turing-institute/sktime/compare/v0.8.1...v0.9.0


[0.8.1] - 2021-10-28
--------------------

Highlights
~~~~~~~~~~

* main forecasting pipelines now support multivariate forecasting - tuning, pipelines, imputers (:pr:`1376`) :user:`aiwalter`
* collection of new transformers - date-time dummies, statistical summaries, STL transform, transformer from function (:pr:`1329` :pr:`1356` :pr:`1463` :pr:`1498`) :user:`boukepostma` :user:`eyalshafran` :user:`danbartl` :user:`RNKuhns`
* new interface points for probabilistic forecasting, :code:`predict_interval` and :code:`predict_quantiles` (:pr:`1421`) :user:`SveaMeyer13`
* experimental interface for time series segmentation (:pr:`1352`) :user:`patrickzib`


New deprecations for 0.10.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* current prediction intervals interface in :code:`predict` via :code:`return_pred_int` will be deprecated and replaced by the new interface points :code:`predict_interval` and :code:`predict_quantiles`


Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* new interface points for probabilistic forecasting, :code:`predict_interval` and :code:`predict_quantiles` (:pr:`1421`) :user:`SveaMeyer13`
* changed forecasting :code:`univariate-only` tag to :code:`ignores-exogeneous-X` (:pr:`1358`) :user:`fkiraly`


Added
~~~~~

BaseEstimator/BaseObject
^^^^^^^^^^^^^^^^^^^^^^^^

* Error handling for `get_tag` (:pr:`1450`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* statsmodels VAR interface (:pr:`1083`, :pr:`1491`) :user:`thayeylolu` :user:`fkiraly`
* multivariate :code:`TransformedTargetForecaster`, :code:`ForecastingPipeline`, :code:`BaseGridSearch`, :code:`MultiplexForecaster` (:pr:`1376`) :user:`aiwalter`
* prediction intervals for statsmodels interface :code:`_StatsModelsAdapter` (:pr:`1489`) :user:`eyalshafran`
* geometric mean based forecasting metrics  (:pr:`1472`, :pr:`837`) :user:`RNKuhns`

* new multivariate forecasting dataset, statsmodels macroeconomic data (:pr:`1553`) :user:`aiwalter` :user:`SinghShreya05`


Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* HIVE-COTE 2.0 Classifier (:pr:`1504`) :user:`MatthewMiddlehurst`
* Auto-generate d classifier capabilities summary :pr:`997` (:pr:`1229`) :user:`BINAYKUMAR943`

Transformers
^^^^^^^^^^^^

* date-time dummy feature transformer :code:`DateTimeFeatures` (:pr:`1356`) :user:`danbartl`
* statistical summary transformer, :code:`SummaryTransformer` (:pr:`1329`) :user:`RNKuhns`
* transformer factory from function, :code:`FunctionTransformer` (:pr:`1498`) :user:`boukepostma`
* STL transformation, :code:`STLTransformer` (:pr:`1463`) :user:`eyalshafran`
* Multivariate imputer (:pr:`1461`) :user:`aiwalter`

Annotation: change-points, segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Clasp for time series segmentation (CIKM'21 publication) (:pr:`1352`) :user:`patrickzib`

Documentation
^^^^^^^^^^^^^

* Add badge to track pypi downloads to README (:pr:`1506`) :user:`RNKuhns`
* [DOC] Add deprecation guide (:pr:`1552`) :user:`mloning`
* [DOC] Add coverage consideration to reviewer guide (:pr:`1403`) :user:`mloning`
* [DOC] Update to TSC extension template (:pr:`1525`) :user:`TonyBagnall`

Governance
^^^^^^^^^^

* Governance change: clearer timelines and conditions for decision making (:pr:`1110`) :user:`fkiraly`
* :user:`aiwalter` joined community council (:pr:`1532`)
* :user:`SveaMeyer13`, :user:`GuzalBulatova`, and :user:`freddyaboulton` joined core devs (:pr:`1444`)

Testing framework
^^^^^^^^^^^^^^^^^

* Tests refactor: using `pytest_generate_tests` instead of loops (:pr:`1407`) :user:`fkiraly`
* Tests refactor: Adding get_test_params method to extension template (:pr:`1395`) :user:`Aparna-Sakshi`
* Changed defaults in `make_forecasting_problem` (:pr:`1477`) :user:`aiwalter`

Fixed
~~~~~

* Refactor TSC: base class (:pr:`1517`) :user:`TonyBagnall`
* Refactor TSC: Hybrid/kernel based classification package (:pr:`1557`) :user:`MatthewMiddlehurst`
* Refactor TSC: Dictionary based classification package (:pr:`1544`) :user:`MatthewMiddlehurst`
* Refactor TSC: Time series classifiers refactor/Shape_DTW (:pr:`1554`) :user:`Piyush1729`
* Refactor TSC: :code:`_muse` classifier (:pr:`1359`) :user:`BINAYKUMAR943`
* Refactor TSC: :code:`ShapeletTransformClassifier`, documentation for HC2 (:pr:`1490`) :user:`MatthewMiddlehurst`
* Refactor TSC: catch22 (:pr:`1487`) :user:`RavenRudi`
* Refactor TSC: tsfresh classifier (:pr:`1473`) :user:`kejsitake`

* Refactor forecasting: forecaster x/y checks (:pr:`1436`) :user:`fkiraly`

* [MNT] Fix appveyor failure (:pr:`1541`) :user:`freddyaboulton`
* [MNT] Fix macOS CI (:pr:`1511`) :user:`mloning`
* [MNT] Depcrecate manylinux2010 (:pr:`1379`) :user:`mloning`
* [MNT] Added pre-commit hook to sort imports (:pr:`1465`) :user:`aiwalter`
* [MNT] add :code:`max_requirements`, bound statsmodels (:pr:`1479`) :user:`fkiraly`
* [MNT] Hotfix tag scitype:y typo (:pr:`1449`) :user:`aiwalter`
* [MNT] Add :code:`pydocstyle` to precommit (:pr:`890`) :user:`mloning`

* [BUG] incorrect/missing weighted geometric mean in forecasting ensemble (:pr:`1370`) :user:`fkiraly`
* [BUG] :pr:`1469`: stripping names of index X and y  (:pr:`1493`) :user:`boukepostma`
* [BUG] W-XXX frequency bug from :pr:`866` (:pr:`1409`) :user:`xiaobenbenecho`
* [BUG] Pandas.NA for unpredictible insample forecasts in AutoARIMA (:pr:`1442`) :user:`IlyasMoutawwakil`
* [BUG] missing :code:`extract_path` in :code:`_data_io` (:pr:`1475`) :user:`yairbeer`
* [BUG] Refactor sktime/.../_panels/_examples.py for tsai compatibility (:pr:`1453`) :user:`bobbys-dev`
* [BUG] Grid/random search tag fix (:pr:`1455`) :user:`fkiraly`
* [BUG] model_selection/split passed the entire DataFrame as index if DataFrame was provided (:pr:`1456`) :user:`fkiraly`
* [BUG] multivariate :code:`NaiveForecaster` was missing :code:`update` (:pr:`1457`) :user:`fkiraly`

* [DOC] docstring fixes in :code:`_proximity_forest.py` (:pr:`1531`) :user:`TonyBagnall`
* [DOC] fixes to landing page links (:pr:`1429`) :user:`Aparna-Sakshi`
* [DOC] Add DataChef blog post to community showcase (:pr:`1464`) :user:`myprogrammerpersonality`
* [DOC] Fixes broken links/estimator overview (:pr:`1445`) :user:`afzal442`
* [DOC] Remove license info from docstrings (:pr:`1437`) :user:`ronnie-llamado`


All contributors: :user:`Aparna-Sakshi`, :user:`BINAYKUMAR943`, :user:`IlyasMoutawwakil`, :user:`MatthewMiddlehurst`, :user:`Piyush1729`, :user:`RNKuhns`, :user:`RavenRudi`, :user:`SveaMeyer13`, :user:`TonyBagnall`, :user:`afzal442`, :user:`aiwalter`, :user:`bobbys-dev`, :user:`boukepostma`, :user:`danbartl`, :user:`eyalshafran`, :user:`fkiraly`, :user:`freddyaboulton`, :user:`kejsitake`, :user:`mloning`, :user:`myprogrammerpersonality`, :user:`patrickzib`, :user:`ronnie-llamado`, :user:`xiaobenbenecho`, :user:`SinghShreya05`, and :user:`yairbeer`



[0.8.0] - 2021-09-17
--------------------

Highlights
~~~~~~~~~~

* Python 3.9 support for linux/osx (:pr:`1255`) :user:`freddyaboulton`
* :code:`conda-forge` metapackage for installing `sktime` with all extras :user:`freddyaboulton`
* framework support for multivariate forecasting (:pr:`980` :pr:`1195` :pr:`1286` :pr:`1301` :pr:`1306` :pr:`1311` :pr:`1401` :pr:`1410`) :user:`aiwalter` :user:`fkiraly` :user:`thayeylolu`
* consolidated lookup of estimators and tags using :code:`registry.all_estimators` and :code:`registry.all_tags` (:pr:`1196`) :user:`fkiraly`
* [DOC] major overhaul of :code:`sktime`'s `online documentation <https://www.sktime.org/en/latest/>`_
* [DOC] `searchable, auto-updating estimators register <https://www.sktime.org/en/latest/estimator_overview.html>`_ in online documentation (:pr:`930` :pr:`1138`) :user:`afzal442` :user:`mloning`
* [MNT] working Binder in-browser notebook showcase (:pr:`1266`) :user:`corvusrabus`
* [DOC] tutorial notebook for in-memory data format conventions, validation, and conversion (:pr:`1232`) :user:`fkiraly`
* easy conversion functionality for estimator inputs, series and panel data (:pr:`1061` :pr:`1187` :pr:`1201` :pr:`1225`) :user:`fkiraly`
* consolidated tags system, dynamic tagging (:pr:`1091` :pr:`1134`) :user:`fkiraly`


Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseEstimator/BaseObject
^^^^^^^^^^^^^^^^^^^^^^^^

* estimator (class and object) capabilities are inspectable by :code:`get_tag` and :code:`get_tags` interface
* list all tags applying to an estimator type by :code:`registry/all_tags`
* list all estimators of a specific type, with certain tags, by :code:`registry/all_estimators`

In-memory data types
^^^^^^^^^^^^^^^^^^^^

* introduction of m(achine)types and scitypes for defining in-memory format conventions across all modules, see `in-memory data types tutorial <https://github.com/alan-turing-institute/sktime/blob/main/examples/AA_datatypes_and_datasets.ipynb>`_
* loose conversion methods now in :code:`_convert` files in :code:`datatypes` will no longer be publicly accessible in 0.10.0

Forecasting
^^^^^^^^^^^

* Forecasters can now be passed :code:`pd.DataFrame`, :code:`pd.Series`, :code:`np.ndarray` as :code:`X` or :code:`y`, and return forecasts of the same type as passed for :code:`y`
* :code:`sktime` now supports multivariate forecasters, with all core interface methods returning sensible return types in that case
* whether forecaster can deal with multivariate series can be inspected via :code:`get_tag("scitype:y")`, which can return :code:`"univariate"`, :code:`"multivariate"`, or :code:`"both"`
* further tags have been introduced, see :code:`registry/all_tags`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* tags have been introduced, see :code:`registry/all_tags`


Added
~~~~~

Forecasting
^^^^^^^^^^^

* Multivariate :code:`ColumnEnsembleForecaster` (:pr:`1082` :pr:`1349`) :user:`fkiraly` :user:`GuzalBulatova`
* Multivariate :code:`NaiveForecaster` (:pr:`1401`) :user:`aiwalter`
* :code:`UnobservedComponents` :code:`statsmodels` wrapper (:pr:`1394`) :user:`juanitorduz`
* :code:`AutoEnsembleForecaster` (:pr:`1220`) :user:`aiwalter`
* :code:`TrendForecaster` (using :code:`sklearn` regressor for value vs time index) (:pr:`1209`) :user:`tensorflow-as-tf`
* Multivariate moving cutoff formatting (:pr:`1213`) :user:`fkiraly`
* Prophet custom seasonalities (:pr:`1378`) :user:`IlyasMoutawwakil`
* Extend aggregation functionality in :code:`EnsembleForecaster` (:pr:`1190`) :user:`GuzalBulatova`
* :code:`plot_lags` to plot series against its lags (:pr:`1330`) :user:`RNKuhns`
* Added :code:`n_best_forecasters` summary to grid searches (:pr:`1139`) :user:`aiwalter`
* Forecasting grid search: cloning more tags (:pr:`1360`) :user:`fkiraly`
* :code:`ForecastingHorizon` supporting more input types, :code:`is_relative` detection on construction from index type (:pr:`1169`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Rotation forest time series classifier (:pr:`1391`) :user:`MatthewMiddlehurst`
* Transform classifiers (:pr:`1180`) :user:`MatthewMiddlehurst`
* New Proximity forest version (:pr:`733`) :user:`moradabaz`
* Enhancement on RISE (:pr:`975`) :user:`whackteachers`


Transformers
^^^^^^^^^^^^

* :code:`ColumnwiseTransformer` (multivariate transformer compositor) (:pr:`1044`) :user:`SveaMeyer13`
* :code:`Differencer` transformer (:pr:`945`) :user:`RNKuhns`
* :code:`FeatureSelection` transformer (:pr:`1347`) :user:`aiwalter`
* :code:`ExponentTransformer` and :code:`SqrtTransformer` (:pr:`1127`) :user:`RNKuhns`


Benchmarking and evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Critical Difference Diagrams (:pr:`1277`) :user:`SveaMeyer13`
* Classification experiments (:pr:`1260`) :user:`TonyBagnall`
* Clustering experiments (:pr:`1221`) :user:`TonyBagnall`
* change to classification experiments (:pr:`1137`) :user:`TonyBagnall`

Documentation
^^^^^^^^^^^^^

* Update documentation backend and reduce warnings in doc creation (:pr:`1199`) (:pr:`1205`) :user:`mloning`
* [DOC] Development community showcase page (:pr:`1337`) :user:`afzal442`
* [DOC] additional clarifying details to documentation guide (in developer's guide) (:pr:`1315`) :user:`RNKuhns`
* [DOC] Add annotation ext template (:pr:`1151`) :user:`mloning`
* [DOC] roadmap document (:pr:`1145`) :user:`mloning`

Testing framework
^^^^^^^^^^^^^^^^^

* unit test for absence of side effects in estimator methods (:pr:`1078`) :user:`fkiraly`


Fixed
~~~~~

* Refactor forecasting: :code:`StackingForecaster` (:pr:`1220`) :user:`aiwalter`

* Refactor TSC: DrCIF and CIF to new interface (:pr:`1269`) :user:`MatthewMiddlehurst`
* Refactor TSC: TDE additions and documentation for HC2 (:pr:`1357`) :user:`MatthewMiddlehurst`
* Refactor TSC: Arsenal additions and documentation for HC2 (:pr:`1305`) :user:`MatthewMiddlehurst`
* Refactor TSC: _cboss (:pr:`1295`) :user:`BINAYKUMAR943`
* Refactor TSC: rocket classifier (:pr:`1239`) :user:`victordremov`
* Refactor TSC: Dictionary based classifiers (:pr:`1084`) :user:`MatthewMiddlehurst`

* Refactor tests: estimator test parameters with the estimator (:pr:`1361`) :user:`Aparna-Sakshi`

* Update _data_io.py (:pr:`1308`) :user:`TonyBagnall`
* Data io (:pr:`1248`) :user:`TonyBagnall`

* [BUG] checking of input types in plotting (:pr:`1197`) :user:`fkiraly`
* [BUG] :code:`NaiveForecaster` behaviour fix for trailing NaN values (:pr:`1130`) :user:`Flix6x`
* [BUG] Fix :code:`all_estimators` when extras are missing. (:pr:`1259`) :user:`xloem`
* [BUG] Contract test fix (:pr:`1392`) :user:`MatthewMiddlehurst`
* [BUG] Data writing updates and JapaneseVowels dataset fix (:pr:`1278`) :user:`MatthewMiddlehurst`
* [BUG] Fixed ESTIMATOR_TEST_PARAMS reference in :code:`test_all_estimators` (:pr:`1406`) :user:`fkiraly`
* [BUG] remove incorrect exogeneous and return_pred_int errors (:pr:`1368`) :user:`fkiraly`
* [BUG] - broken binder and test_examples check (:pr:`1343`) :user:`fkiraly`
* [BUG] Fix minor silent issues in :code:`TransformedTargetForecaster` (:pr:`845`) :user:`aiwalter`
* [BUG] Troubleshooting for C compiler after pytest failed (:pr:`1262`) :user:`tensorflow-as-tf`
* [BUG] bugfix in tutorial documentation of univariate time series classification. (:pr:`1140`) :user:`BINAYKUMAR943`
* [BUG] removed format check from index test (:pr:`1193`) :user:`fkiraly`
* [BUG] bugfix - convertIO broken references to np.ndarray (:pr:`1191`) :user:`fkiraly`
* [BUG] STSF test fix (:pr:`1170`) :user:`MatthewMiddlehurst`
* [BUG] :code:`set_tags` call in :code:`BaseObject.clone_tags` used incorrect signature (:pr:`1179`) :user:`fkiraly`

* [DOC] Update transformer docstrings Boss (:pr:`1320`) :user:`thayeylolu`
* [DOC] Updated docstring of exp_smoothing.py (:pr:`1339`) :user:`mathco-wf`
* [DOC] updated the link in CONTRIBUTING.md (:pr:`1428`) :user:`Aparna-Sakshi`
* [DOC] Correct typo in contributing guidelines (:pr:`1398`) :user:`juanitorduz`
* [DOC] Fix community repo link (:pr:`1400`) :user:`mloning`
* [DOC] Fix minor typo in README (:pr:`1416`) :user:`justinshenk`
* [DOC] Fixed a typo in citation page (:pr:`1310`) :user:`AreloTanoh`
* [DOC] EnsembleForecaster and AutoEnsembleForecaster docstring example (:pr:`1382`) :user:`aiwalter`
* [DOC] multiple minor fixes to docs (:pr:`1328`) :user:`mloning`
* [DOC] Docstring improvements for bats, tbats, arima, croston (:pr:`1309`) :user:`Lovkush-A`
* [DOC] Update detrend module docstrings (:pr:`1335`) :user:`SveaMeyer13`
* [DOC] updated extension templates - object tags (:pr:`1340`) :user:`fkiraly`
* [DOC] Update ThetaLinesTransformer's docstring (:pr:`1312`) :user:`GuzalBulatova`
* [DOC] Update ColumnwiseTransformer and TabularToSeriesAdaptor docstrings (:pr:`1322`) :user:`GuzalBulatova`
* [DOC] Update transformer docstrings (:pr:`1314`) :user:`RNKuhns`
* [DOC] Description and link to cosine added (:pr:`1326`) :user:`AreloTanoh`
* [DOC] naive forcasting docstring edits (:pr:`1333`) :user:`AreloTanoh`
* [DOC] Update .all-contributorsrc (:pr:`1336`) :user:`pul95`
* [DOC] Typo in transformations.rst fixed (:pr:`1324`) :user:`AreloTanoh`
* [DOC] Add content to documentation guide for use in docsprint (:pr:`1297`) :user:`RNKuhns`
* [DOC] Added slack and google calendar to README (:pr:`1283`) :user:`aiwalter`
* [DOC] Add binder badge to README (:pr:`1285`) :user:`mloning`
* [DOC] docstring fix for distances/series extension templates (:pr:`1256`) :user:`fkiraly`
* [DOC] adding binder link to readme (landing page) (:pr:`1282`) :user:`fkiraly`
* [DOC] Update contributors (:pr:`1243`) :user:`mloning`
* [DOC] add conda-forge max dependency recipe to installation and readme (:pr:`1226`) :user:`fkiraly`
* [DOC] Adding table of content in the forecasting tutorial (:pr:`1200`) :user:`bilal-196`
* [DOC] Complete docstring of EnsembleForecaster  (:pr:`1165`) :user:`GuzalBulatova`
* [DOC] Add annotation to docs (:pr:`1156`) :user:`mloning`
* [DOC] Add funding (:pr:`1173`) :user:`mloning`
* [DOC] Minor update to See Also of BOSS Docstrings (:pr:`1172`) :user:`RNKuhns`
* [DOC] Refine the Docstrings for BOSS Classifiers (:pr:`1166`) :user:`RNKuhns`
* [DOC] add examples in docstrings in classification (:pr:`1164`) :user:`ltoniazzi`
* [DOC] adding example in docstring of KNeighborsTimeSeriesClassifier (:pr:`1155`) :user:`ltoniazzi`
* [DOC] Update README  (:pr:`1024`) :user:`fkiraly`
* [DOC] rework of installation guidelines (:pr:`1103`) :user:`fkiraly`

* [MNT] Update codecov config (:pr:`1396`) :user:`mloning`
* [MNT] removing tests for data downloader dependent on third party website, change in test dataset for test_time_series_neighbors (:pr:`1258`) :user:`TonyBagnall`
* [MNT] Fix appveyor CI (:pr:`1253`) :user:`mloning`
* [MNT] Update feature_request.md (:pr:`1242`) :user:`aiwalter`
* [MNT] Format setup files (:pr:`1236`) :user:`TonyBagnall`
* [MNT] Fix pydocstyle config (:pr:`1149`) :user:`mloning`
* [MNT] Update release script (:pr:`1135`) :user:`mloning`

All contributors: :user:`Aparna-Sakshi`, :user:`AreloTanoh`, :user:`BINAYKUMAR943`, :user:`Flix6x`, :user:`GuzalBulatova`, :user:`IlyasMoutawwakil`, :user:`Lovkush-A`, :user:`MatthewMiddlehurst`, :user:`RNKuhns`, :user:`SveaMeyer13`, :user:`TonyBagnall`, :user:`afzal442`, :user:`aiwalter`, :user:`bilal-196`, :user:`corvusrabus`, :user:`fkiraly`, :user:`freddyaboulton`, :user:`juanitorduz`, :user:`justinshenk`, :user:`ltoniazzi`, :user:`mathco-wf`, :user:`mloning`, :user:`moradabaz`, :user:`pul95`, :user:`tensorflow-as-tf`, :user:`thayeylolu`, :user:`victordremov`, :user:`whackteachers` and :user:`xloem`


[0.7.0] - 2021-07-12
--------------------

Added
~~~~~
* new module (experimental): Time Series Clustering (:pr:`1049`) :user:`TonyBagnall`
* new module (experimental): Pairwise transformers, kernels/distances on tabular data and panel data - base class, examples, extension templates (:pr:`1071`) :user:`fkiraly` :user:`chrisholder`
* new module (experimental): Series annotation and PyOD adapter (:pr:`1021`) :user:`fkiraly` :user:`satya-pattnaik`
* Clustering extension templates, docstrings & get_fitted_params (:pr:`1100`) :user:`fkiraly`
* New Classifier: Implementation of signature based methods.  (:pr:`714`) :user:`jambo6`
* New Forecaster: Croston's method (:pr:`730`) :user:`Riyabelle25`
* New Forecaster: ForecastingPipeline for pipelining with exog data (:pr:`967`) :user:`aiwalter`
* New Transformer: Multivariate Detrending (:pr:`1042`) :user:`SveaMeyer13`
* New Transformer: ThetaLines transformer (:pr:`923`) :user:`GuzalBulatova`
* sktime registry (:pr:`1067`) :user:`fkiraly`
* Feature/information criteria get_fitted_params (:pr:`942`) :user:`ltsaprounis`
* Add plot_correlations() to plot series and acf/pacf (:pr:`850`) :user:`RNKuhns`
* Add doc-quality tests on changed files (:pr:`752`) :user:`mloning`
* Docs: Create add_dataset.rst (:pr:`970`) :user:`Riyabelle25`
* Added two new related software packages (:pr:`1019`) :user:`aiwalter`
* Added orbit as related software (:pr:`1128`) :user:`aiwalter`
* adding fkiraly as codeowner for forecasting base classes (:pr:`989`) :user:`fkiraly`
* added mloning and aiwalter as forecasting/base code owners (:pr:`1108`) :user:`fkiraly`

Changed
~~~~~~~
* Update metric to handle y_train (:pr:`858`) :user:`RNKuhns`
* TSC base template refactor (:pr:`1026`) :user:`fkiraly`
* Forecasting refactor: base class refactor and extension template (:pr:`912`) :user:`fkiraly`
* Forecasting refactor: base/template docstring fixes, added fit_predict method (:pr:`1109`) :user:`fkiraly`
* Forecasters refactor: NaiveForecaster (:pr:`953`) :user:`fkiraly`
* Forecasters refactor: BaseGridSearch, ForecastingGridSearchCV, ForecastingRandomizedSearchCV (:pr:`1034`) :user:`GuzalBulatova`
* Forecasting refactor: polynomial trend forecaster (:pr:`1003`) :user:`thayeylolu`
* Forecasting refactor: Stacking, Multiplexer, Ensembler and TransformedTarget Forecasters (:pr:`977`) :user:`thayeylolu`
* Forecasting refactor: statsmodels and  theta forecaster (:pr:`1029`) :user:`thayeylolu`
* Forecasting refactor: reducer (:pr:`1031`) :user:`Lovkush-A`
* Forecasting refactor: ensembler, online-ensembler-forecaster and descendants (:pr:`1015`) :user:`thayeylolu`
* Forecasting refactor: TbatAdapter (:pr:`1017`) :user:`thayeylolu`
* Forecasting refactor: PmdArimaAdapter (:pr:`1016`) :user:`thayeylolu`
* Forecasting refactor: Prophet (:pr:`1005`) :user:`thayeylolu`
* Forecasting refactor: CrystallBall Forecaster (:pr:`1004`) :user:`thayeylolu`
* Forecasting refactor: default tags in BaseForecaster; added some new tags (:pr:`1013`) :user:`fkiraly`
* Forecasting refactor: removing _SktimeForecaster and horizon mixins (:pr:`1088`) :user:`fkiraly`
* Forecasting tutorial rework (:pr:`972`) :user:`fkiraly`
* Added tuning tutorial to forecasting example notebook - fkiraly suggestions on top of :pr:`1047` (:pr:`1053`) :user:`fkiraly`
* Classification: Kernel based refactor (:pr:`875`) :user:`MatthewMiddlehurst`
* Classification: catch22 Remake (:pr:`864`) :user:`MatthewMiddlehurst`
* Forecasting: Remove step_length hyper-parameter from reduction classes (:pr:`900`) :user:`mloning`
* Transformers: Make OptionalPassthrough to support multivariate input (:pr:`1112`) :user:`aiwalter`
* Transformers: Improvement to Multivariate-Detrending (:pr:`1077`) :user:`SveaMeyer13`
* Update plot_series to handle pd.Int64 and pd.Range index uniformly (:pr:`892`) :user:`Dbhasin1`
* Including floating numbers as a window length (:pr:`827`) :user:`thayeylolu`
* update docs on loading data (:pr:`885`) :user:`SveaMeyer13`
* Update docs (:pr:`887`) :user:`mloning`
* [DOC] Updated docstrings to inform that methods accept ForecastingHorizon (:pr:`872`) :user:`julramos`

Fixed
~~~~~
* Fix use of seasonal periodicity in naive model with mean strategy (from PR :pr:`917`) (:pr:`1124`) :user:`mloning`
* Fix ForecastingPipeline import (:pr:`1118`) :user:`mloning`
* Bugfix - forecasters should use internal interface _all_tags for self-inspection, not _has_tag (:pr:`1068`) :user:`fkiraly`
* bugfix: Prophet adapter fails to clone after setting parameters (:pr:`911`) :user:`Yard1`
* Fix seeding issue in Minirocket Classifier (:pr:`1094`) :user:`Lovkush-A`
* fixing soft dependencies link (:pr:`1035`) :user:`fkiraly`
* Fix minor typos in docstrings (:pr:`889`) :user:`GuzalBulatova`
* Fix manylinux CI (:pr:`914`) :user:`mloning`
* Add limits.h to ensure pip install on certain OS's (:pr:`915`) :user:`tombh`
* Fix side effect on input for Imputer and HampelFilter (:pr:`1089`) :user:`aiwalter`
* BaseCluster class issues resolved (:pr:`1075`) :user:`chrisholder`
* Cleanup metric docstrings and fix bug in _RelativeLossMixin (:pr:`999`) :user:`RNKuhns`
* minor clarifications in forecasting extension template preamble (:pr:`1069`) :user:`fkiraly`
* Fix fh in imputer method based on in-sample forecasts (:pr:`861`) :user:`julramos`
* Arsenal fix, extended capabilities and HC1 unit tests (:pr:`902`) :user:`MatthewMiddlehurst`
* minor bugfix - setting _is_fitted to False before input checks in forecasters (:pr:`941`) :user:`fkiraly`
* Properly process random_state when fitting Time Series Forest ensemble in parallel (:pr:`819`) :user:`kachayev`
* bump nbqa (:pr:`998`) :user:`MarcoGorelli`
* datetime: Construct Timedelta from parsed pandas frequency (:pr:`873`) :user:`ckastner`

All contributors: :user:`Dbhasin1`, :user:`GuzalBulatova`, :user:`Lovkush-A`, :user:`MarcoGorelli`, :user:`MatthewMiddlehurst`, :user:`RNKuhns`, :user:`Riyabelle25`, :user:`SveaMeyer13`, :user:`TonyBagnall`, :user:`Yard1`, :user:`aiwalter`, :user:`chrisholder`, :user:`ckastner`, :user:`fkiraly`, :user:`jambo6`, :user:`julramos`, :user:`kachayev`, :user:`ltsaprounis`, :user:`mloning`, :user:`thayeylolu` and :user:`tombh`


[0.6.1] - 2021-05-14
--------------------

Fixed
~~~~~
* Exclude Python 3.10 from manylinux CI (:pr:`870`) :user:`mloning`
* Fix AutoETS handling of infinite information criteria (:pr:`848`) :user:`ltsaprounis`
* Fix smape import (:pr:`851`) :user:`mloning`

Changed
~~~~~~~
* ThetaForecaster now works with initial_level (:pr:`769`) :user:`yashlamba`
* Use joblib to parallelize ensemble fitting for Rocket classifier (:pr:`796`) :user:`kachayev`
* Update maintenance tools (:pr:`829`) :user:`mloning`
* Undo pmdarima hotfix and avoid pmdarima 1.8.1 (:pr:`831`) :user:`aaronreidsmith`
* Hotfix pmdarima version (:pr:`828`) :user:`aiwalter`

Added
~~~~~
* Added Guerrero method for lambda estimation to BoxCoxTransformer (:pr:`778`) (:pr:`791`) :user:`GuzalBulatova`
* New forecasting metrics (:pr:`801`) :user:`RNKuhns`
* Implementation of DirRec reduction strategy (:pr:`779`) :user:`luiszugasti`
* Added cutoff to BaseGridSearch to use any grid search inside evaluate… (:pr:`825`) :user:`aiwalter`
* Added pd.DataFrame transformation for Imputer and HampelFilter (:pr:`830`) :user:`aiwalter`
* Added default params for some transformers (:pr:`834`) :user:`aiwalter`
* Added several docstring examples (:pr:`835`) :user:`aiwalter`
* Added skip-inverse-transform tag for Imputer and HampelFilter (:pr:`788`) :user:`aiwalter`
* Added a reference to alibi-detect (:pr:`815`) :user:`satya-pattnaik`

All contributors: :user:`GuzalBulatova`, :user:`RNKuhns`, :user:`aaronreidsmith`, :user:`aiwalter`, :user:`kachayev`, :user:`ltsaprounis`, :user:`luiszugasti`, :user:`mloning`, :user:`satya-pattnaik` and :user:`yashlamba`


[0.6.0] - 2021-04-15
--------------------

Fixed
~~~~~
* Fix counting for Github's automatic language discovery (:pr:`812`) :user:`xuyxu`
* Fix counting for Github's automatic language discovery (:pr:`811`) :user:`xuyxu`
* Fix examples CI checks (:pr:`793`) :user:`mloning`
* Fix TimeSeriesForestRegressor (:pr:`777`) :user:`mloning`
* Fix Deseasonalizer docstring (:pr:`737`) :user:`mloning`
* SettingWithCopyWarning in Prophet with exogenous data (:pr:`735`) :user:`jschemm`
* Correct docstrings for check_X and related functions (:pr:`701`) :user:`Lovkush-A`
* Fixed bugs mentioned in :pr:`694`  (:pr:`697`) :user:`AidenRushbrooke`
* fix typo in CONTRIBUTING.md (:pr:`688`) :user:`luiszugasti`
* Fix duplicacy in the contribution's list (:pr:`685`) :user:`afzal442`
* HIVE-COTE 1.0 fix (:pr:`678`) :user:`MatthewMiddlehurst`

Changed
~~~~~~~
* Update sklearn version (:pr:`810`) :user:`mloning`
* Remove soft dependency check for numba (:pr:`808`) :user:`mloning`
* Modify tests for forecasting reductions (:pr:`756`) :user:`Lovkush-A`
* Upgrade nbqa (:pr:`794`) :user:`MarcoGorelli`
* Enhanced exception message of splitters (:pr:`771`) :user:`aiwalter`
* Enhance forecasting model selection/evaluation (:pr:`739`) :user:`mloning`
* Pin PyStan version (:pr:`751`) :user:`mloning`
* master to main conversion in docs folder closes :pr:`644` (:pr:`667`) :user:`ayan-biswas0412`
* Update governance (:pr:`686`) :user:`mloning`
* remove MSM from unit tests for now (:pr:`698`) :user:`TonyBagnall`
* Make update_params=true by default (:pr:`660`) :user:`pabworks`
* update dataset names (:pr:`676`) :user:`TonyBagnall`

Added
~~~~~
* Add support for exogenous variables to forecasting reduction (:pr:`757`) :user:`mloning`
* Added forecasting docstring examples (:pr:`772`) :user:`aiwalter`
* Added the agg argument to EnsembleForecaster (:pr:`774`) :user:`Ifeanyi30`
* Added OptionalPassthrough transformer (:pr:`762`) :user:`aiwalter`
* Add doctests (:pr:`766`) :user:`mloning`
* Multiplexer forecaster (:pr:`715`) :user:`koralturkk`
* Upload source tarball to PyPI during releases (:pr:`749`) :user:`dsherry`
* Create developer guide (:pr:`734`) :user:`mloning`
* Refactor TSF classifier into TSF regressor (:pr:`693`) :user:`luiszugasti`
* Outlier detection with HampelFilter (:pr:`708`) :user:`aiwalter`
* changes to contributing.md to include directions to installation (:pr:`695`) :user:`kanand77`
* Evaluate (example and fix) (:pr:`690`) :user:`aiwalter`
* Knn unit tests (:pr:`705`) :user:`TonyBagnall`
* Knn transpose fix (:pr:`689`) :user:`TonyBagnall`
* Evaluate forecaster function (:pr:`657`) :user:`aiwalter`
* Multioutput reduction strategy for forecasting (:pr:`659`) :user:`Lovkush-A`

All contributors: :user:`AidenRushbrooke`, :user:`Ifeanyi30`, :user:`Lovkush-A`, :user:`MarcoGorelli`, :user:`MatthewMiddlehurst`, :user:`TonyBagnall`, :user:`afzal442`, :user:`aiwalter`, :user:`ayan-biswas0412`, :user:`dsherry`, :user:`jschemm`, :user:`kanand77`, :user:`koralturkk`, :user:`luiszugasti`, :user:`mloning`, :user:`pabworks` and :user:`xuyxu`


[0.5.3] - 2021-02-06
--------------------

Fixed
~~~~~
* Fix reduced regression forecaster reference (:pr:`658`) :user:`mloning`
* Address Bug :pr:`640` (:pr:`642`) :user:`patrickzib`
* Ed knn (:pr:`638`) :user:`TonyBagnall`
* Euclidean distance for KNNs (:pr:`636`) :user:`goastler`

Changed
~~~~~~~
* Pin NumPy 1.19 (:pr:`643`) :user:`mloning`
* Update CoC committee (:pr:`614`) :user:`mloning`
* Benchmarking issue141 (:pr:`492`) :user:`ViktorKaz`
* Catch22 Refactor & Multithreading (:pr:`615`) :user:`MatthewMiddlehurst`

Added
~~~~~
* Create new factory method for forecasting via reduction (:pr:`635`) :user:`Lovkush-A`
* Feature ForecastingRandomizedSearchCV (:pr:`634`) :user:`pabworks`
* Added Imputer for missing values (:pr:`637`) :user:`aiwalter`
* Add expanding window splitter (:pr:`627`) :user:`koralturkk`
* Forecasting User Guide (:pr:`595`) :user:`Lovkush-A`
* Add data processing functionality to convert between data formats (:pr:`553`) :user:`RNKuhns`
* Add basic parallel support for `ElasticEnsemble` (:pr:`546`) :user:`xuyxu`

All contributors: :user:`Lovkush-A`, :user:`MatthewMiddlehurst`, :user:`RNKuhns`, :user:`TonyBagnall`, :user:`ViktorKaz`, :user:`aiwalter`, :user:`goastler`, :user:`koralturkk`, :user:`mloning`, :user:`pabworks`, :user:`patrickzib` and :user:`xuyxu`

[0.5.2] - 2021-01-13
--------------------

Fixed
~~~~~
* Fix ModuleNotFoundError issue (:pr:`613`) :user:`Hephaest`
* Fixes _fit(X) in KNN (:pr:`610`) :user:`TonyBagnall`
* UEA TSC module improvements 2 (:pr:`599`) :user:`TonyBagnall`
* Fix sktime.classification.frequency_based not found error (:pr:`606`) :user:`Hephaest`
* UEA TSC module improvements 1 (:pr:`579`) :user:`TonyBagnall`
* Relax numba pinning (:pr:`593`) :user:`dhirschfeld`
* Fix fh.to_relative() bug for DatetimeIndex (:pr:`582`) :user:`aiwalter`

All contributors: :user:`Hephaest`, :user:`MatthewMiddlehurst`, :user:`TonyBagnall`, :user:`aiwalter` and :user:`dhirschfeld`

[0.5.1] - 2020-12-29
--------------------

Added
~~~~~
* Add ARIMA (:pr:`559`) :user:`HYang1996`
* Add fbprophet wrapper (:pr:`515`) :user:`aiwalter`
* Add MiniRocket and MiniRocketMultivariate (:pr:`542`) :user:`angus924`
* Add Cosine, ACF and PACF transformers (:pr:`509`) :user:`afzal442`
* Add example notebook Window Splitters (:pr:`555`) :user:`juanitorduz`
* Add SlidingWindowSplitter visualization on doctrings (:pr:`554`) :user:`juanitorduz`

Fixed
~~~~~
* Pin pandas version to fix pandas-related AutoETS error on Linux  (:pr:`581`) :user:`mloning`
* Fixed default argument in docstring in SlidingWindowSplitter (:pr:`556`) :user:`ngupta23`

All contributors: :user:`HYang1996`, :user:`TonyBagnall`, :user:`afzal442`, :user:`aiwalter`, :user:`angus924`, :user:`juanitorduz`, :user:`mloning` and :user:`ngupta23`

[0.5.0] - 2020-12-19
--------------------

Added
~~~~~
* Add tests for forecasting with exogenous variables (:pr:`547`) :user:`mloning`
* Add HCrystalBall wrapper (:pr:`485`) :user:`MichalChromcak`
* Tbats (:pr:`527`) :user:`aiwalter`
* Added matrix profile using stumpy  (:pr:`471`) :user:`utsavcoding`
* User guide (:pr:`377`) :user:`mloning`
* Add GitHub workflow for building and testing on macOS (:pr:`505`) :user:`mloning`
* [DOC] Add dtaidistance (:pr:`502`) :user:`mloning`
* Implement the `feature_importances_` property for RISE (:pr:`497`) :user:`AaronX121`
* Add scikit-fda to the list of related software (:pr:`495`) :user:`vnmabus`
* [DOC] Add roadmap to docs (:pr:`467`) :user:`mloning`
* Add parallelization for `RandomIntervalSpectralForest` (:pr:`482`) :user:`AaronX121`
* New Ensemble Forecasting Methods  (:pr:`333`) :user:`magittan`
* CI run black formatter on notebooks as well as Python scripts (:pr:`437`) :user:`MarcoGorelli`
* Implementation of catch22 transformer, CIF classifier and dictionary based clean-up (:pr:`453`) :user:`MatthewMiddlehurst`
* Added write dataset to ts file functionality (:pr:`438`) :user:`whackteachers`
* Added ability to load from csv containing long-formatted data (:pr:`442`) :user:`AidenRushbrooke`
* Transform typing (:pr:`420`) :user:`mloning`

Changed
~~~~~~~
* Refactoring utils and transformer module (:pr:`538`) :user:`mloning`
* Update README (:pr:`454`) :user:`mloning`
* Clean up example notebooks (:pr:`548`) :user:`mloning`
* Update README.rst (:pr:`536`) :user:`aiwalter`
* [Doc]Updated load_data.py (:pr:`496`) :user:`Afzal-Ind`
* Update forecasting.py (:pr:`487`) :user:`raishubham1`
* update basic motion description (:pr:`475`) :user:`vollmersj`
* [DOC] Update docs in benchmarking/data.py (:pr:`489`) :user:`Afzal-Ind`
* Edit Jupyter Notebook 01_forecasting (:pr:`486`) :user:`bmurdata`
* Feature & Performance improvements of SFA/WEASEL (:pr:`457`) :user:`patrickzib`
* Moved related software from wiki to docs (:pr:`439`) :user:`mloning`

Fixed
~~~~~
* Fixed issue outlined in issue 522 (:pr:`537`) :user:`ngupta23`
* Fix plot-series (:pr:`533`) :user:`gracewgao`
* added mape_loss and cosmetic fixes to notebooks (removed kernel) (:pr:`500`) :user:`tch`
* Fix azure pipelines (:pr:`506`) :user:`mloning`
* [DOC] Fix broken docstrings of `RandomIntervalSpectralForest` (:pr:`473`) :user:`AaronX121`
* Add back missing bibtex reference to classifiers (:pr:`468`) :user:`whackteachers`
* Avoid seaborn warning (:pr:`472`) :user:`davidbp`
* Bump pre-commit versions, run again on notebooks (:pr:`469`) :user:`MarcoGorelli`
* Fix series validation (:pr:`463`) :user:`mloning`
* Fix soft dependency imports (:pr:`446`) :user:`mloning`
* Fix bug in AutoETS (:pr:`445`) :user:`HYang1996`
* Add ForecastingHorizon class to docs (:pr:`444`) :user:`mloning`

Removed
~~~~~~~
* Remove manylinux1 (:pr:`458`) :user:`mloning`

All contributors: :user:`AaronX121`, :user:`Afzal-Ind`, :user:`AidenRushbrooke`, :user:`HYang1996`, :user:`MarcoGorelli`, :user:`MatthewMiddlehurst`, :user:`MichalChromcak`, :user:`TonyBagnall`, :user:`aiwalter`, :user:`bmurdata`, :user:`davidbp`, :user:`gracewgao`, :user:`magittan`, :user:`mloning`, :user:`ngupta23`, :user:`patrickzib`, :user:`raishubham1`, :user:`tch`, :user:`utsavcoding`, :user:`vnmabus`, :user:`vollmersj` and :user:`whackteachers`

[0.4.3] - 2020-10-20
--------------------

Added
~~~~~
* Support for 3d numpy array (:pr:`405`) :user:`mloning`
* Support for downloading dataset from UCR UEA time series classification data set repository (:pr:`430`) :user:`Emiliathewolf`
* Univariate time series regression example to TSFresh notebook (:pr:`428`) :user:`evanmiller29`
* Parallelized TimeSeriesForest using joblib. (:pr:`408`) :user:`kkoziara`
* Unit test for multi-processing (:pr:`414`) :user:`kkoziara`
* Add date-time support for forecasting framework (:pr:`392`) :user:`mloning`

Changed
~~~~~~~
* Performance improvements of dictionary classifiers (:pr:`398`) :user:`patrickzib`

Fixed
~~~~~
* Fix links in Readthedocs and Binder launch button (:pr:`416`) :user:`mloning`
* Fixed small bug in performance metrics (:pr:`422`) :user:`krumeto`
* Resolved warnings in notebook examples (:pr:`418`) :user:`alwinw`
* Resolves :pr:`325` ModuleNotFoundError for soft dependencies (:pr:`410`) :user:`alwinw`

All contributors: :user:`Emiliathewolf`, :user:`alwinw`, :user:`evanmiller29`, :user:`kkoziara`, :user:`krumeto`, :user:`mloning` and :user:`patrickzib`


[0.4.2] - 2020-10-01
--------------------

Added
~~~~~
* ETSModel with auto-fitting capability (:pr:`393`) :user:`HYang1996`
* WEASEL classifier (:pr:`391`) :user:`patrickzib`
* Full support for exogenous data in forecasting framework (:pr:`382`) :user:`mloning`, (:pr:`380`) :user:`mloning`
* Multivariate dataset for US consumption over time (:pr:`385`) :user:`SebasKoel`
* Governance document (:pr:`324`) :user:`mloning`, :user:`fkiraly`

Fixed
~~~~~
* Documentation fixes (:pr:`400`) :user:`brettkoonce`, (:pr:`399`) :user:`akanz1`, (:pr:`404`) :user:`alwinw`

Changed
~~~~~~~
* Move documentation to ReadTheDocs with support for versioned documentation (:pr:`395`) :user:`mloning`
* Refactored SFA implementation (additional features and speed improvements) (:pr:`389`) :user:`patrickzib`
* Move prediction interval API to base classes in forecasting framework (:pr:`387`) :user:`big-o`
* Documentation improvements (:pr:`364`) :user:`mloning`
* Update CI and maintenance tools (:pr:`394`) :user:`mloning`

All contributors: :user:`HYang1996`, :user:`SebasKoel`, :user:`fkiraly`, :user:`akanz1`, :user:`alwinw`, :user:`big-o`, :user:`brettkoonce`, :user:`mloning`, :user:`patrickzib`


[0.4.1] - 2020-07-09
--------------------

Added
~~~~~
- New sktime logo :user:`mloning`
- TemporalDictionaryEnsemble (:pr:`292`) :user:`MatthewMiddlehurst`
- ShapeDTW (:pr:`287`) :user:`Multivin12`
- Updated sktime artwork (logo) :user:`mloning`
- Truncation transformer (:pr:`315`) :user:`ABostrom`
- Padding transformer (:pr:`316`) :user:`ABostrom`
- Example notebook with feature importance graph for time series forest (:pr:`319`) :user:`HYang1996`
- ACSF1 data set (:pr:`314`) :user:`BandaSaiTejaReddy`
- Data conversion function from 3d numpy array to nested pandas dataframe (:pr:`304`) :user:`vedazeren`

Changed
~~~~~~~
- Replaced gunpoint dataset in tutorials, added OSULeaf dataset (:pr:`295`) :user:`marielledado`
- Updated macOS advanced install instructions (:pr:`306`) (:pr:`308`) :user:`sophijka`
- Updated contributing guidelines (:pr:`301`) :user:`Ayushmaanseth`

Fixed
~~~~~
- Typos (:pr:`293`) :user:`Mo-Saif`, (:pr:`285`) :user:`Pangoraw`, (:pr:`305`) :user:`hiqbal2`
- Manylinux wheel building (:pr:`286`) :user:`mloning`
- KNN compatibility with sklearn (:pr:`310`) :user:`Cheukting`
- Docstrings for AutoARIMA (:pr:`307`) :user:`btrtts`

All contributors: :user:`Ayushmaanseth`, :user:`Mo-Saif`, :user:`Pangoraw`, :user:`marielledado`,
:user:`mloning`, :user:`sophijka`, :user:`Cheukting`, :user:`MatthewMiddlehurst`, :user:`Multivin12`,
:user:`ABostrom`, :user:`HYang1996`, :user:`BandaSaiTejaReddy`, :user:`vedazeren`, :user:`hiqbal2`, :user:`btrtts`


[0.4.0] - 2020-06-05
--------------------

Added
~~~~~
- Forecasting framework, including: forecasting algorithms (forecasters),
  tools for composite model building (meta-forecasters), tuning and model
  evaluation
- Consistent unit testing of all estimators
- Consistent input checks
- Enforced PEP8 linting via flake8
- Changelog
- Support for Python 3.8
- Support for manylinux wheels


Changed
~~~~~~~
- Revised all estimators to comply with common interface and to ensure scikit-learn compatibility

Removed
~~~~~~~
- A few redundant classes for the series-as-features setting in favour of scikit-learn's implementations: :code:`Pipeline` and :code:`GridSearchCV`
- :code:`HomogeneousColumnEnsembleClassifier` in favour of more flexible :code:`ColumnEnsembleClassifier`

Fixed
~~~~~
- Deprecation and future warnings from scikit-learn
- User warnings from statsmodels
