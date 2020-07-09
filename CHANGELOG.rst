Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and we adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

We keep track of changes in this file since v0.4.0.

[0.4.2] - 2020-xx-xx
--------------------


[0.4.1] - 2020-07-09
--------------------

Added
~~~~~
- TemporalDictionaryEnsemble (#292) @MatthewMiddlehurst
- ShapeDTW (#287) @Multivin12
- Updated sktime artwork (logo) @mloning
- Truncation transformer (#315) @ABostrom
- Padding transformer (#316) @ABostrom
- Example notebook with feature importance graph for time series forest (#319) @HYang1996
- ACSF1 data set (#314) @BandaSaiTejaReddy
- Data conversion function from 3d numpy array to nested pandas dataframe
(#304) @vedazeren

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
