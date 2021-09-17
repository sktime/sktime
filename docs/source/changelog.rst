.. _changelog:

Changelog
=========

All notable changes to this project will be documented in this file. We keep track of changes in this file since v0.4.0. The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and we adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_. The source code for all `releases <https://github.com/alan-turing-institute/sktime/releases>`_ is available on GitHub.

.. note::

    To stay up-to-date with sktime releases, subscribe to sktime `here
    <https://libraries.io/pypi/sktime>`_ or follow us on `Twitter <https://twitter.com/sktime_toolbox>`_.



[0.8.0] - 2021-09-17
--------------------

Highlights
~~~~~~~~~~

* python 3.9 support for linux/osx (#1255) @freddyaboulton
* `conda-forge` metapackage for installing `sktime` with all extras @freddyaboulton
* framework support for multivariate forecasting (#980 #1195 #1286 #1301 #1306 #1311 #1401 #1410) @aiwalter @fkiraly @thayeylolu
* consolidated lookup of estimators and tags using `registry.all_estimators` and `registry.all_tags` (#1196) @fkiraly
* [DOC] major overhaul of `sktime`'s [online documentation](https://www.sktime.org/en/latest/)
* [DOC] [searchable, auto-updating estimators register](https://www.sktime.org/en/latest/estimator_overview.html) in online documentation (#930 #1138) @afzal442 @mloning
* [MNT] working Binder in-browser notebook showcase (#1266) @corvusrabus
* [DOC] tutorial notebook for in-memory data format conventions, validation, and conversion (#1232) @fkiraly
* easy conversion functionality for estimator inputs, series and panel data (#1061 #1187 #1201 #1225) @fkiraly
* consolidated tags system, dynamic tagging (#1091 #1134) @fkiraly


Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseEstimator/BaseObject
^^^^^^^^^^^^^^^^^^^^^^^^

* estimator (class and object) capabilities are inspectable by `get_tag` and `get_tags` interface
* list all tags applying to an estimator type by `registry/all_tags`
* list all estimators of a specific type, with certain tags, by `registry/all_estimators`

In-memory data types
^^^^^^^^^^^^^^^^^^^^

* introduction of m(achine)types and scitypes for defining in-memory format conventions across all modules, see [in-memory data types tutorial](https://github.com/alan-turing-institute/sktime/blob/main/examples/AA_datatypes_and_datasets.ipynb)
* loose conversion methods now in `_convert` files in `datatypes` will be deprecated in 0.10.0

Forecasting
^^^^^^^^^^^

* Forecasters can now be passed `pd.DataFrame`, `pd.Series`, `np.ndarray` as `X`, `y` and return forecasts of the same type as passed for `y`
* whether forecaster can deal with multivariate series can be inspected via `get_tag("scitype:y")`, which can return `"univariate"`, `"multivariate"`, or `"both"`
* further tags have been introduced, see `registry/all_tags`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* tags have been introduced, see `registry/all_tags`


Added
~~~~~

Forecasting
^^^^^^^^^^^

* Multivariate `ColumnEnsembleForecaster` (#1082 #1349) @fkiraly @GuzalBulatova
* Multivariate `NaiveForecaster` (#1401) @aiwalter
* `UnobservedComponents` `statsmodels` wrapper (#1394) @juanitorduz
* `AutoEnsembleForecaster` (#1220) @aiwalter
* `TrendForecaster` (using `sklearn` regressor for value vs time index) (#1209) @tensorflow-as-tf
* Multivariate moving cutoff formatting (#1213) @fkiraly
* Prophet custom seasonalities (#1378) @IlyasMoutawwakil
* Extend aggregation functionality in `EnsembleForecaster` (#1190) @GuzalBulatova
* `plot_lags` to plot series against its lags (#1330) @RNKuhns
* Added `n_best_forecasters` summary to grid searches (#1139) @aiwalter
* Forecasting grid search: cloning more tags (#1360) @fkiraly
* `ForecastingHorizon` supporting more input types, `is_relative` detection on construction from index type (#1169) @fkiraly

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Rotation forest time series classifier (#1391) @MatthewMiddlehurst
* Transform classifiers (#1180) @MatthewMiddlehurst
* New Proximity forest version (#733) @moradabaz
* Enhancement on RISE (#975) @whackteachers


Transformers
^^^^^^^^^^^^

* `ColumnwiseTransformer` (multivariate transformer compositor) (#1044) @SveaMeyer13
* `Differencer` transformaer (#945) @RNKuhns
* `FeatureSelection` transformer (#1347) @aiwalter
* `ExponentTransformer` and `SqrtTransformer` (#1127) @RNKuhns


Benchmarking and evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Critical Difference Diagrams (#1277) @SveaMeyer13
* Classification experiments (#1260) @TonyBagnall
* Clustering experiments (#1221) @TonyBagnall
* change to classification experiments (#1137) @TonyBagnall

Documentation
^^^^^^^^^^^^^

* Update documentation backend and reduce warnings in doc creation (#1199) (#1205) @mloning
* [DOC] Development community showcase page (#1337) @afzal442
* [DOC] additional clarifying details to documentation guide (in developer's guide) (#1315) @RNKuhns
* [DOC] Add annotation ext template (#1151) @mloning
* [DOC] roadmap document (#1145) @mloning

Testing framework
^^^^^^^^^^^^^^^^^

* unit test for absence of side effects in estimator methods (#1078) @fkiraly


Fixed
~~~~~

* Refactor forecasting: `StackingForecaster` (#1220) @aiwalter

* Refactor TSC: DrCIF and CIF to new interface (#1269) @MatthewMiddlehurst
* Refactor TSC: TDE additions and documentation for HC2 (#1357) @MatthewMiddlehurst
* Refactor TSC: Arsenal additions and documentation for HC2 (#1305) @MatthewMiddlehurst
* Refactor TSC: _cboss (#1295) @BINAYKUMAR943
* Refactor TSC: rocket classifier (#1239) @victordremov
* Refactor TSC: Dictionary based classifiers (#1084) @MatthewMiddlehurst

* Refactor tests: estimator test parameters with the estimator (#1361) @Aparna-Sakshi

* Update _data_io.py (#1308) @TonyBagnall
* Data io (#1248) @TonyBagnall

* [BUG] checking of input types in plotting (#1197) @fkiraly
* [BUG] `NaiveForecaster` behaviour fix for trailing NaN values (#1130) @Flix6x
* [BUG] Fix `all_estimators` when extras are missing. (#1259) @xloem
* [BUG] Contract test fix (#1392) @MatthewMiddlehurst
* [BUG] Data writing updates and JapaneseVowels dataset fix (#1278) @MatthewMiddlehurst
* [BUG] Fixed ESTIMATOR_TEST_PARAMS reference in `test_all_estimators` (#1406) @fkiraly
* [BUG] remove incorrect exogeneous and return_pred_int errors (#1368) @fkiraly
* [BUG] - broken binder and test_examples check (#1343) @fkiraly
* [BUG] Fix minor silent issues in `TransformedTargetForecaster` (#845) @aiwalter
* [BUG] Troubleshooting for C compiler after pytest failed (#1262) @tensorflow-as-tf
* [BUG] bugfix in tutorial documentation of univariate time series classification. (#1140) @BINAYKUMAR943
* [BUG] removed format check from index test (#1193) @fkiraly
* [BUG] bugfix - convertIO broken references to np.ndarray (#1191) @fkiraly
* [BUG] STSF test fix (#1170) @MatthewMiddlehurst
* [BUG] `set_tags` call in `BaseObject.clone_tags` used incorrect signature (#1179) @fkiraly

* [DOC] Update transformer docstrings Boss (#1320) @thayeylolu
* [DOC] Updated docstring of exp_smoothing.py (#1339) @mathco-wf
* [DOC] updated the link in CONTRIBUTING.md (#1428) @Aparna-Sakshi
* [DOC] Correct typo in contributing guidelines (#1398) @juanitorduz
* [DOC] Fix community repo link (#1400) @mloning
* [DOC] Fix minor typo in README (#1416) @justinshenk
* [DOC] Fixed a typo in citation page (#1310) @AreloTanoh
* [DOC] EnsembleForecaster and AutoEnsembleForecaster docstring example (#1382) @aiwalter
* [DOC] multiple minor fixes to docs (#1328) @mloning
* [DOC] Docstring improvements for bats, tbats, arima, croston (#1309) @Lovkush-A
* [DOC] Update detrend module docstrings (#1335) @SveaMeyer13
* [DOC] updated extension templates - object tags (#1340) @fkiraly
* [DOC] Update ThetaLinesTransformer's docstring (#1312) @GuzalBulatova
* [DOC] Update ColumnwiseTransformer and TabularToSeriesAdaptor docstrings (#1322) @GuzalBulatova
* [DOC] Update transformer docstrings (#1314) @RNKuhns
* [DOC] Description and link to cosine added (#1326) @AreloTanoh
* [DOC] naive forcasting docstring edits (#1333) @AreloTanoh
* [DOC] Update .all-contributorsrc (#1336) @pul95
* [DOC] Typo in transformations.rst fixed (#1324) @AreloTanoh
* [DOC] Add content to documentation guide for use in docsprint (#1297) @RNKuhns
* [DOC] Added slack and google calendar to README (#1283) @aiwalter
* [DOC] Add binder badge to README (#1285) @mloning
* [DOC] docstring fix for distances/series extension templates (#1256) @fkiraly
* [DOC] adding binder link to readme (landing page) (#1282) @fkiraly
* [DOC] Update contributors (#1243) @mloning
* [DOC] add conda-forge max dependency recipe to installation and readme (#1226) @fkiraly
* [DOC] Adding table of content in the forecasting tutorial (#1200) @bilal-196
* [DOC] Complete docstring of EnsembleForecaster  (#1165) @GuzalBulatova
* [DOC] Add annotation to docs (#1156) @mloning
* [DOC] Add funding (#1173) @mloning
* [DOC] Minor update to See Also of BOSS Docstrings (#1172) @RNKuhns
* [DOC] Refine the Docstrings for BOSS Classifiers (#1166) @RNKuhns
* [DOC] add examples in docstrings in classification (#1164) @ltoniazzi
* [DOC] adding example in docstring of KNeighborsTimeSeriesClassifier (#1155) @ltoniazzi
* [DOC] Update README  (#1024) @fkiraly
* [DOC] rework of installation guidelines (#1103) @fkiraly

* [MNT] Update codecov config (#1396) @mloning
* [MNT] removing tests for data downloader dependent on third party website, change in test dataset for test_time_series_neighbors (#1258) @TonyBagnall
* [MNT] Fix appveyor CI (#1253) @mloning
* [MNT] Update feature_request.md (#1242) @aiwalter
* [MNT] Format setup files (#1236) @TonyBagnall
* [MNT] Fix pydocstyle config (#1149) @mloning
* [MNT] Update release script (#1135) @mloning

All contributors: @Aparna-Sakshi, @AreloTanoh, @BINAYKUMAR943, @Flix6x, @GuzalBulatova, @IlyasMoutawwakil, @Lovkush-A, @MatthewMiddlehurst, @RNKuhns, @SveaMeyer13, @TonyBagnall, @afzal442, @aiwalter, @bilal-196, @corvusrabus, @fkiraly, @freddyaboulton, @juanitorduz, @justinshenk, @ltoniazzi, @mathco-wf, @mloning, @moradabaz, @pul95, @tensorflow-as-tf, @thayeylolu, @victordremov, @whackteachers and @xloem


[0.7.0] - 2021-07-12
--------------------

Added
~~~~~
* new module (experimental): Time Series Clustering (#1049) @TonyBagnall
* new module (experimental): Pairwise transformers, kernels/distances on tabular data and panel data - base class, examples, extension templates (#1071) @fkiraly @chrisholder
* new module (experimental): Series annotation and PyOD adapter (#1021) @fkiraly @satya-pattnaik
* Clustering extension templates, docstrings & get_fitted_params (#1100) @fkiraly
* New Classifier: Implementation of signature based methods.  (#714) @jambo6
* New Forecaster: Croston's method (#730) @Riyabelle25
* New Forecaster: ForecastingPipeline for pipelining with exog data (#967) @aiwalter
* New Transformer: Multivariate Detrending (#1042) @SveaMeyer13
* New Transformer: ThetaLines transformer (#923) @GuzalBulatova
* sktime registry (#1067) @fkiraly
* Feature/information criteria get_fitted_params (#942) @ltsaprounis
* Add plot_correlations() to plot series and acf/pacf (#850) @RNKuhns
* Add doc-quality tests on changed files (#752) @mloning
* Docs: Create add_dataset.rst (#970) @Riyabelle25
* Added two new related software packages (#1019) @aiwalter
* Added orbit as related software (#1128) @aiwalter
* adding fkiraly as codeowner for forecasting base classes (#989) @fkiraly
* added mloning and aiwalter as forecasting/base code owners (#1108) @fkiraly

Changed
~~~~~~~
* Update metric to handle y_train (#858) @RNKuhns
* TSC base template refactor (#1026) @fkiraly
* Forecasting refactor: base class refactor and extension template (#912) @fkiraly
* Forecasting refactor: base/template docstring fixes, added fit_predict method (#1109) @fkiraly
* Forecasters refactor: NaiveForecaster (#953) @fkiraly
* Forecasters refactor: BaseGridSearch, ForecastingGridSearchCV, ForecastingRandomizedSearchCV (#1034) @GuzalBulatova
* Forecasting refactor: polynomial trend forecaster (#1003) @thayeylolu
* Forecasting refactor: Stacking, Multiplexer, Ensembler and TransformedTarget Forecasters (#977) @thayeylolu
* Forecasting refactor: statsmodels and  theta forecaster (#1029) @thayeylolu
* Forecasting refactor: reducer (#1031) @Lovkush-A
* Forecasting refactor: ensembler, online-ensembler-forecaster and descendants (#1015) @thayeylolu
* Forecasting refactor: TbatAdapter (#1017) @thayeylolu
* Forecasting refactor: PmdArimaAdapter (#1016) @thayeylolu
* Forecasting refactor: Prophet (#1005) @thayeylolu
* Forecasting refactor: CrystallBall Forecaster (#1004) @thayeylolu
* Forecasting refactor: default tags in BaseForecaster; added some new tags (#1013) @fkiraly
* Forecasting refactor: removing _SktimeForecaster and horizon mixins (#1088) @fkiraly
* Forecasting tutorial rework (#972) @fkiraly
* Added tuning tutorial to forecasting example notebook - fkiraly suggestions on top of #1047 (#1053) @fkiraly
* Classification: Kernel based refactor (#875) @MatthewMiddlehurst
* Classification: catch22 Remake (#864) @MatthewMiddlehurst
* Forecasting: Remove step_length hyper-parameter from reduction classes (#900) @mloning
* Transformers: Make OptionalPassthrough to support multivariate input (#1112) @aiwalter
* Transformers: Improvement to Multivariate-Detrending (#1077) @SveaMeyer13
* Update plot_series to handle pd.Int64 and pd.Range index uniformly (#892) @Dbhasin1
* Including floating numbers as a window length (#827) @thayeylolu
* update docs on loading data (#885) @SveaMeyer13
* Update docs (#887) @mloning
* [DOC] Updated docstrings to inform that methods accept ForecastingHorizon (#872) @julramos

Fixed
~~~~~
* Fix use of seasonal periodicity in naive model with mean strategy (from PR #917) (#1124) @mloning
* Fix ForecastingPipeline import (#1118) @mloning
* Bugfix - forecasters should use internal interface _all_tags for self-inspection, not _has_tag (#1068) @fkiraly
* bugfix: Prophet adapter fails to clone after setting parameters (#911) @Yard1
* Fix seeding issue in Minirocket Classifier (#1094) @Lovkush-A
* fixing soft dependencies link (#1035) @fkiraly
* Fix minor typos in docstrings (#889) @GuzalBulatova
* Fix manylinux CI (#914) @mloning
* Add limits.h to ensure pip install on certain OS's (#915) @tombh
* Fix side effect on input for Imputer and HampelFilter (#1089) @aiwalter
* BaseCluster class issues resolved (#1075) @chrisholder
* Cleanup metric docstrings and fix bug in _RelativeLossMixin (#999) @RNKuhns
* minor clarifications in forecasting extension template preamble (#1069) @fkiraly
* Fix fh in imputer method based on in-sample forecasts (#861) @julramos
* Arsenal fix, extended capabilities and HC1 unit tests (#902) @MatthewMiddlehurst
* minor bugfix - setting _is_fitted to False before input checks in forecasters (#941) @fkiraly
* Properly process random_state when fitting Time Series Forest ensemble in parallel (#819) @kachayev
* bump nbqa (#998) @MarcoGorelli
* datetime: Construct Timedelta from parsed pandas frequency (#873) @ckastner

All contributors: @Dbhasin1, @GuzalBulatova, @Lovkush-A, @MarcoGorelli, @MatthewMiddlehurst, @RNKuhns, @Riyabelle25, @SveaMeyer13, @TonyBagnall, @Yard1, @aiwalter, @chrisholder, @ckastner, @fkiraly, @jambo6, @julramos, @kachayev, @ltsaprounis, @mloning, @thayeylolu and @tombh


[0.6.1] - 2021-05-14
--------------------

Fixed
~~~~~
* Exclude Python 3.10 from manylinux CI (#870) @mloning
* Fix AutoETS handling of infinite information criteria (#848) @ltsaprounis
* Fix smape import (#851) @mloning

Changed
~~~~~~~
* ThetaForecaster now works with initial_level (#769) @yashlamba
* Use joblib to parallelize ensemble fitting for Rocket classifier (#796) @kachayev
* Update maintenance tools (#829) @mloning
* Undo pmdarima hotfix and avoid pmdarima 1.8.1 (#831) @aaronreidsmith
* Hotfix pmdarima version (#828) @aiwalter

Added
~~~~~
* Added Guerrero method for lambda estimation to BoxCoxTransformer (#778) (#791) @GuzalBulatova
* New forecasting metrics (#801) @RNKuhns
* Implementation of DirRec reduction strategy (#779) @luiszugasti
* Added cutoff to BaseGridSearch to use any grid search inside evaluate… (#825) @aiwalter
* Added pd.DataFrame transformation for Imputer and HampelFilter (#830) @aiwalter
* Added default params for some transformers (#834) @aiwalter
* Added several docstring examples (#835) @aiwalter
* Added skip-inverse-transform tag for Imputer and HampelFilter (#788) @aiwalter
* Added a reference to alibi-detect (#815) @satya-pattnaik

All contributors: @GuzalBulatova, @RNKuhns, @aaronreidsmith, @aiwalter, @kachayev, @ltsaprounis, @luiszugasti, @mloning, @satya-pattnaik and @yashlamba


[0.6.0] - 2021-04-15
--------------------

Fixed
~~~~~
* Fix counting for Github's automatic language discovery (#812) @xuyxu
* Fix counting for Github's automatic language discovery (#811) @xuyxu
* Fix examples CI checks (#793) @mloning
* Fix TimeSeriesForestRegressor (#777) @mloning
* Fix Deseasonalizer docstring (#737) @mloning
* SettingWithCopyWarning in Prophet with exogenous data (#735) @jschemm
* Correct docstrings for check_X and related functions (#701) @Lovkush-A
* Fixed bugs mentioned in #694  (#697) @AidenRushbrooke
* fix typo in CONTRIBUTING.md (#688) @luiszugasti
* Fix duplicacy in the contribution's list (#685) @afzal442
* HIVE-COTE 1.0 fix (#678) @MatthewMiddlehurst

Changed
~~~~~~~
* Update sklearn version (#810) @mloning
* Remove soft dependency check for numba (#808) @mloning
* Modify tests for forecasting reductions (#756) @Lovkush-A
* Upgrade nbqa (#794) @MarcoGorelli
* Enhanced exception message of splitters (#771) @aiwalter
* Enhance forecasting model selection/evaluation (#739) @mloning
* Pin PyStan version (#751) @mloning
* master to main conversion in docs folder closes #644 (#667) @ayan-biswas0412
* Update governance (#686) @mloning
* remove MSM from unit tests for now (#698) @TonyBagnall
* Make update_params=true by default (#660) @pabworks
* update dataset names (#676) @TonyBagnall

Added
~~~~~
* Add support for exogenous variables to forecasting reduction (#757) @mloning
* Added forecasting docstring examples (#772) @aiwalter
* Added the agg argument to EnsembleForecaster (#774) @Ifeanyi30
* Added OptionalPassthrough transformer (#762) @aiwalter
* Add doctests (#766) @mloning
* Multiplexer forecaster (#715) @koralturkk
* Upload source tarball to PyPI during releases (#749) @dsherry
* Create developer guide (#734) @mloning
* Refactor TSF classifier into TSF regressor (#693) @luiszugasti
* Outlier detection with HampelFilter (#708) @aiwalter
* changes to contributing.md to include directions to installation (#695) @kanand77
* Evaluate (example and fix) (#690) @aiwalter
* Knn unit tests (#705) @TonyBagnall
* Knn transpose fix (#689) @TonyBagnall
* Evaluate forecaster function (#657) @aiwalter
* Multioutput reduction strategy for forecasting (#659) @Lovkush-A

All contributors: @AidenRushbrooke, @Ifeanyi30, @Lovkush-A, @MarcoGorelli, @MatthewMiddlehurst, @TonyBagnall, @afzal442, @aiwalter, @ayan-biswas0412, @dsherry, @jschemm, @kanand77, @koralturkk, @luiszugasti, @mloning, @pabworks and @xuyxu


[0.5.3] - 2021-02-06
--------------------

Fixed
~~~~~
* Fix reduced regression forecaster reference (#658) @mloning
* Address Bug #640 (#642) @patrickzib
* Ed knn (#638) @TonyBagnall
* Euclidean distance for KNNs (#636) @goastler

Changed
~~~~~~~
* Pin NumPy 1.19 (#643) @mloning
* Update CoC committee (#614) @mloning
* Benchmarking issue141 (#492) @ViktorKaz
* Catch22 Refactor & Multithreading (#615) @MatthewMiddlehurst

Added
~~~~~
* Create new factory method for forecasting via reduction (#635) @Lovkush-A
* Feature ForecastingRandomizedSearchCV (#634) @pabworks
* Added Imputer for missing values (#637) @aiwalter
* Add expanding window splitter (#627) @koralturkk
* Forecasting User Guide (#595) @Lovkush-A
* Add data processing functionality to convert between data formats (#553) @RNKuhns
* Add basic parallel support for `ElasticEnsemble` (#546) @xuyxu

All contributors: @Lovkush-A, @MatthewMiddlehurst, @RNKuhns, @TonyBagnall, @ViktorKaz, @aiwalter, @goastler, @koralturkk, @mloning, @pabworks, @patrickzib and @xuyxu

[0.5.2] - 2021-01-13
--------------------

Fixed
~~~~~
* Fix ModuleNotFoundError issue (#613) @Hephaest
* Fixes _fit(X) in KNN (#610) @TonyBagnall
* UEA TSC module improvements 2 (#599) @TonyBagnall
* Fix sktime.classification.frequency_based not found error (#606) @Hephaest
* UEA TSC module improvements 1 (#579) @TonyBagnall
* Relax numba pinning (#593) @dhirschfeld
* Fix fh.to_relative() bug for DatetimeIndex (#582) @aiwalter

All contributors: @Hephaest, @MatthewMiddlehurst, @TonyBagnall, @aiwalter and @dhirschfeld

[0.5.1] - 2020-12-29
--------------------

Added
~~~~~
* Add ARIMA (#559) @HYang1996
* Add fbprophet wrapper (#515) @aiwalter
* Add MiniRocket and MiniRocketMultivariate (#542) @angus924
* Add Cosine, ACF and PACF transformers (#509) @afzal442
* Add example notebook Window Splitters (#555) @juanitorduz
* Add SlidingWindowSplitter visualization on doctrings (#554) @juanitorduz

Fixed
~~~~~
* Pin pandas version to fix pandas-related AutoETS error on Linux  (#581) @mloning
* Fixed default argument in docstring in SlidingWindowSplitter (#556) @ngupta23

All contributors: @HYang1996, @TonyBagnall, @afzal442, @aiwalter, @angus924, @juanitorduz, @mloning and @ngupta23

[0.5.0] - 2020-12-19
--------------------

Added
~~~~~
* Add tests for forecasting with exogenous variables (#547) @mloning
* Add HCrystalBall wrapper (#485) @MichalChromcak
* Tbats (#527) @aiwalter
* Added matrix profile using stumpy  (#471) @utsavcoding
* User guide (#377) @mloning
* Add GitHub workflow for building and testing on macOS (#505) @mloning
* [DOC] Add dtaidistance (#502) @mloning
* Implement the `feature_importances_` property for RISE (#497) @AaronX121
* Add scikit-fda to the list of related software (#495) @vnmabus
* [DOC] Add roadmap to docs (#467) @mloning
* Add parallelization for `RandomIntervalSpectralForest` (#482) @AaronX121
* New Ensemble Forecasting Methods  (#333) @magittan
* CI run black formatter on notebooks as well as Python scripts (#437) @MarcoGorelli
* Implementation of catch22 transformer, CIF classifier and dictionary based clean-up (#453) @MatthewMiddlehurst
* Added write dataset to ts file functionality (#438) @whackteachers
* Added ability to load from csv containing long-formatted data (#442) @AidenRushbrooke
* Transform typing (#420) @mloning

Changed
~~~~~~~
* Refactoring utils and transformer module (#538) @mloning
* Update README (#454) @mloning
* Clean up example notebooks (#548) @mloning
* Update README.rst (#536) @aiwalter
* [Doc]Updated load_data.py (#496) @Afzal-Ind
* Update forecasting.py (#487) @raishubham1
* update basic motion description (#475) @vollmersj
* [DOC] Update docs in benchmarking/data.py (#489) @Afzal-Ind
* Edit Jupyter Notebook 01_forecasting (#486) @bmurdata
* Feature & Performance improvements of SFA/WEASEL (#457) @patrickzib
* Moved related software from wiki to docs (#439) @mloning

Fixed
~~~~~
* Fixed issue outlined in issue 522 (#537) @ngupta23
* Fix plot-series (#533) @gracewgao
* added mape_loss and cosmetic fixes to notebooks (removed kernel) (#500) @tch
* Fix azure pipelines (#506) @mloning
* [DOC] Fix broken docstrings of `RandomIntervalSpectralForest` (#473) @AaronX121
* Add back missing bibtex reference to classifiers (#468) @whackteachers
* Avoid seaborn warning (#472) @davidbp
* Bump pre-commit versions, run again on notebooks (#469) @MarcoGorelli
* Fix series validation (#463) @mloning
* Fix soft dependency imports (#446) @mloning
* Fix bug in AutoETS (#445) @HYang1996
* Add ForecastingHorizon class to docs (#444) @mloning

Removed
~~~~~~~
* Remove manylinux1 (#458) @mloning

All contributors: @AaronX121, @Afzal-Ind, @AidenRushbrooke, @HYang1996, @MarcoGorelli, @MatthewMiddlehurst, @MichalChromcak, @TonyBagnall, @aiwalter, @bmurdata, @davidbp, @gracewgao, @magittan, @mloning, @ngupta23, @patrickzib, @raishubham1, @tch, @utsavcoding, @vnmabus, @vollmersj and @whackteachers

[0.4.3] - 2020-10-20
--------------------

Added
~~~~~
* Support for 3d numpy array (#405) @mloning
* Support for downloading dataset from UCR UEA time series classification data set repository (#430) @Emiliathewolf
* Univariate time series regression example to TSFresh notebook (#428) @evanmiller29
* Parallelized TimeSeriesForest using joblib. (#408) @kkoziara
* Unit test for multi-processing (#414) @kkoziara
* Add date-time support for forecasting framework (#392) @mloning

Changed
~~~~~~~
* Performance improvements of dictionary classifiers (#398) @patrickzib

Fixed
~~~~~
* Fix links in Readthedocs and Binder launch button (#416) @mloning
* Fixed small bug in performance metrics (#422) @krumeto
* Resolved warnings in notebook examples (#418) @alwinw
* Resolves #325 ModuleNotFoundError for soft dependencies (#410) @alwinw

All contributors: @Emiliathewolf, @alwinw, @evanmiller29, @kkoziara, @krumeto, @mloning and @patrickzib


[0.4.2] - 2020-10-01
--------------------

Added
~~~~~
* ETSModel with auto-fitting capability (#393) @HYang1996
* WEASEL classifier (#391) @patrickzib
* Full support for exogenous data in forecasting framework (#382) @mloning, (#380) @mloning
* Multivariate dataset for US consumption over time (#385) @SebasKoel
* Governance document (#324) @mloning, @fkiraly

Fixed
~~~~~
* Documentation fixes (#400) @brettkoonce, (#399) @akanz1, (#404) @alwinw

Changed
~~~~~~~
* Move documentation to ReadTheDocs with support for versioned documentation (#395) @mloning
* Refactored SFA implementation (additional features and speed improvements) (#389) @patrickzib
* Move prediction interval API to base classes in forecasting framework (#387) @big-o
* Documentation improvements (#364) @mloning
* Update CI and maintenance tools (#394) @mloning

All contributors: @HYang1996, @SebasKoel, @fkiraly, @akanz1, @alwinw, @big-o, @brettkoonce, @mloning, @patrickzib


[0.4.1] - 2020-07-09
--------------------

Added
~~~~~
- New sktime logo @mloning
- TemporalDictionaryEnsemble (#292) @MatthewMiddlehurst
- ShapeDTW (#287) @Multivin12
- Updated sktime artwork (logo) @mloning
- Truncation transformer (#315) @ABostrom
- Padding transformer (#316) @ABostrom
- Example notebook with feature importance graph for time series forest (#319) @HYang1996
- ACSF1 data set (#314) @BandaSaiTejaReddy
- Data conversion function from 3d numpy array to nested pandas dataframe (#304) @vedazeren

Changed
~~~~~~~
- Replaced gunpoint dataset in tutorials, added OSULeaf dataset (#295) @marielledado
- Updated macOS advanced install instructions (#306) (#308) @sophijka
- Updated contributing guidelines (#301) @Ayushmaanseth

Fixed
~~~~~
- Typos (#293) @Mo-Saif, (#285) @Pangoraw, (#305) @hiqbal2
- Manylinux wheel building (#286) @mloning
- KNN compatibility with sklearn (#310) @Cheukting
- Docstrings for AutoARIMA (#307) @btrtts

All contributors: @Ayushmaanseth, @Mo-Saif, @Pangoraw, @marielledado,
@mloning, @sophijka, @Cheukting, @MatthewMiddlehurst, @Multivin12,
@ABostrom, @HYang1996, @BandaSaiTejaReddy, @vedazeren, @hiqbal2, @btrtts


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
