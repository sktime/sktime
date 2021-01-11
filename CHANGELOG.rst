Changelog
=========

All notable changes to this project will be documented in this file. We keep track of changes in this file since v0.4.0. The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and we adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_. The source code for all `releases <https://github.com/alan-turing-institute/sktime/releases>`_ is available on GitHub.

[0.5.2] - 2021-xx-xx
--------------------


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
* Fix links in Readthedocs and Binder launch button (#416)
@mloning
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
