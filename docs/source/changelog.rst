.. _changelog:

Changelog
=========

All notable changes to this project will be documented in this file. We keep track of changes in this file since v0.4.0. The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and we adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_. The source code for all `releases <https://github.com/alan-turing-institute/sktime/releases>`_ is available on GitHub.

.. note::

    To stay up-to-date with sktime releases, subscribe to sktime `here
    <https://libraries.io/pypi/sktime>`_ or follow us on `Twitter <https://twitter.com/sktime_toolbox>`_.

For upcoming changes and next releases, see our `milestones <https://github.com/alan-turing-institute/sktime/milestones?direction=asc&sort=due_date&state=open>`_.
For our long-term plan, see our :ref:`roadmap`.


[0.9.0] - 2021-12-03
--------------------

Documentation
^^^^^^^^^^^^^

* [DOC] Added myself as contributor (:pr:`1602`) :user:`Carlosbogo`
* [DOC] Add missing classes to API reference (:pr:`1571`) :user:`RNKuhns`
* [DOC] additions to forecaster extension template (:pr:`1535`) :user:`fkiraly`
* [DOC] Add toggle button to make examples easy to copy (:pr:`1572`) :user:`RNKuhns`
* [DOC] Update docs from roadmap planning sessions (:pr:`1527`) :user:`mloning`
* [DOC] Add freddyaboulton to core developers list (:pr:`1559`) :user:`freddyaboulton`

Governance
^^^^^^^^^^

* Governance: eligibility and end of tenure clarification (:pr:`1573`) :user:`fkiraly`

Performance metrics
^^^^^^^^^^^^^^^^^^^

* Fix Performance metric bug that affects repr (:pr:`1566`) :user:`RNKuhns`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Fixes :issue:`1234` (:pr:`1600`) :user:`Carlosbogo`
* convert 2D to 3D rather than throw exception (:pr:`1604`) :user:`TonyBagnall`
* load from UCR fix (:pr:`1610`) :user:`TonyBagnall`
* Classifier test speed ups (:pr:`1599`) :user:`MatthewMiddlehurst`
* TimeSeriesForest Classifier Fix (:pr:`1588`) :user:`OliverMatthews`
* Interval based classification package refactor (:pr:`1583`) :user:`MatthewMiddlehurst`
* Distance based classification package refactor (:pr:`1584`) :user:`MatthewMiddlehurst`
* Feature based classification package refactor (:pr:`1545`) :user:`MatthewMiddlehurst`
* Deprecate various (:pr:`1548`) :user:`TonyBagnall`

Maintenance
^^^^^^^^^^^

* [MNT] Update release script (:pr:`1562`) :user:`mloning`

All contributors: :user:`Carlosbogo`, :user:`MatthewMiddlehurst`, :user:`OliverMatthews`, :user:`RNKuhns`, :user:`TonyBagnall`, :user:`fkiraly`, :user:`freddyaboulton` and :user:`mloning`


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
