.. _changelog:

Changelog
=========

All notable changes to this project will be documented in this file.
We keep track of changes in this file since v0.4.0.
The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and
we adhere to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.
The source code for all `releases <https://github.com/sktime/sktime/releases>`_ is
available on GitHub.

.. note::

    To stay up-to-date with sktime releases, subscribe to sktime `here
    <https://libraries.io/pypi/sktime>`_ or follow us on `LinkedIn <https://www.linkedin.com/company/scikit-time/>`_.

For upcoming changes and next releases, see our `milestones <https://github.com/sktime/sktime/milestones?direction=asc&sort=due_date&state=open>`_.
For our long-term plan, see our :ref:`roadmap`.


Version 0.32.3 - 2024-08-27
---------------------------

Hotfix release with bugfix for html representation of forecasting pipelines.

For last non-maintenance content updates, see 0.32.2.

Contents
~~~~~~~~

* [BUG] fix html display for ``TransformedTargetForecaster`` and ``ForecastingPipeline``

Version 0.32.2 - 2024-08-26
---------------------------

Highlights
~~~~~~~~~~

* ``HierarchicalProphet`` forecaster from ``prophetverse`` (:pr:`7028`) :user:`felipeangelimvieira`
* Regularized VAR reduction forecaster, ``VARReduce`` (:pr:`6725`) :user:`meraldoantonio`
* Interface to TimesFM Forecaster (:pr:`6571`) :user:`geetu040`
* Subsequence Extraction Transformer (:pr:`6967`) :user:`wirrywoo`
* Framework support for categorical data has been extended to transformers and pipelines (:pr:`6924`) :user:`Abhay-Lejith`
* Clusterer tags for capability to assign cluster centers (:pr:`7018`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``holiday`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.56``
* ``dask`` (data container and parallelization back-end) bounds have been updated to ``<2024.8.2``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

New tags for clusterers have been added to characterize capabilities
to assign cluster centers. The following boolean tags have been added:

* ``capability:predict``, whether the clusterer can assign cluster labels via ``predict``
* ``capability:predict_proba``, for probabilistic cluster assignment
* ``capability: out_of_sample``, for out-of-sample cluster assignment.
  If False, the clusterer can only assign clusters to data points seen during fitting.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] placeholder record decorator (:pr:`7029`) :user:`fkiraly`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Hierarchical sales toydata generator from workshops (:pr:`6953`) :user:`marrov`
* [ENH] Convert the date column to a period with daily frequency in ``load_m5`` (:pr:`6990`) :user:`SaiRevanth25`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Polars ``Series`` scitype supports  (:pr:`6485`) :user:`pranavvp16`
* [ENH] Polars ``Panel`` scitype support  (:pr:`6552`) :user:`pranavvp16`
* [ENH] Addition of ``feature_kind`` metadata attribute to ``gluonts`` datatypes (:pr:`6871`) :user:`shlok191`

Forecasting
^^^^^^^^^^^

* [ENH] interface to TimesFM Forecaster (:pr:`6571`) :user:`geetu040`
* [ENH] New regularized VAR reduction forecaster, ``VARReduce`` (:pr:`6725`) :user:`meraldoantonio`
* [ENH] Add ``HierarchicalProphet`` estimator to ``prophetverse`` module (:pr:`7028`) :user:`felipeangelimvieira`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] clusterer tags for capability to assign cluster centers (:pr:`7018`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Extending categorical support in X to transformers and pipelines (:pr:`6924`) :user:`Abhay-Lejith`
* [ENH] Subsequence Extraction Transformer (:pr:`6967`) :user:`wirrywoo`

Documentation
~~~~~~~~~~~~~

* [DOC] minor improvements to docstring of ``Bollinger`` (bands) (:pr:`6978`) :user:`fkiraly`
* [DOC] Update ``.all-contributorsrc`` with council roles (:pr:`6962`) :user:`fkiraly`
* [DOC] update soft dependency handling guide for estimators (:pr:`7000`) :user:`fkiraly`
* [DOC] improvements to docstrings for panel tasks - time series classification, regression, clustering (:pr:`6991`) :user:`fkiraly`
* [DOC] update XinyuWuu's user name (:pr:`7030`) :user:`fkiraly`
* [DOC] fixes to ``TransformedTargetForecaster`` docstring (:pr:`7002`) :user:`fkiraly`
* [DOC] update intro notebook with material from ISF and EuroSciPy 2024 (:pr:`7013`) :user:`fkiraly`
* [DOC] Fix docstring for ``ExpandingCutoffSplitter`` (:pr:`7033`) :user:`ninedigits`
* [DOC] fix incorrect import in ``EnbPIForecaster`` docstring (:pr:`7015`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] Refactor ``show_versions`` to use ``dependencies`` module (:pr:`6883`) :user:`fkiraly`
* [MNT] sync changelog with hotfix branch ``anirban-sktime-0.31.2`` (:pr:`6963`) :user:`yarnabrina`
* [MNT] add ``numpy 2`` incompatibility flag to ``pmdarima`` dependency (:pr:`6974`) :user:`fkiraly`
* [MNT] decorate ``test_auto_arima`` with ``numpy 2`` skip until final fix/diagnosis (:pr:`6973`) :user:`fkiraly`
* [MNT] remove ``tsbootstrap`` dependency from public dependency sets (:pr:`6966`) :user:`fkiraly`
* [MNT] rename base class ``TimeSeriesLloyds`` to ``BaseTimeSeriesLloyds`` (:pr:`6992`) :user:`fkiraly`
* [MNT] remove module level ``numba`` import warnings (:pr:`6999`) :user:`fkiraly`
* [MNT] ``esig`` based estimators: add ``numpy<2`` bound (:pr:`7036`) :user:`fkiraly`
* [MNT] [Dependabot](deps): Bump ``tj-actions/changed-files`` from 44 to 45 (:pr:`7019`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``holidays`` requirement from ``<0.55,>=0.29`` to ``>=0.29,<0.56`` (:pr:`7006`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``dask`` requirement from ``<2024.8.1`` to ``<2024.8.2`` (:pr:`7005`) :user:`dependabot[bot]`

Fixes
~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``test_softdep_error`` dependency handling check if environment marker tag is not satisfied (:pr:`6961`) :user:`fkiraly`
* [BUG] fix dependency checkers in case of multiple distributions available in environment, e.g., on databricks (:pr:`6986`) :user:`fkiraly`, :user:`toandaominh1997`

Benchmarking and Metrics
^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix ``ForecastingBenchmark`` giving an error when the dataloader returns the tuple (y, X) (:pr:`6971`) :user:`SaiRevanth25`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix ``nested_univ`` converter inconsistent handling of index level names (:pr:`7026`) :user:`pranavvp16`

Forecasting
^^^^^^^^^^^

* [BUG] ``TinyTimeMixerForecaster``: fix truncating index and update ``test_params`` (:pr:`6965`) :user:`geetu040`
* [BUG] Do not add season condition names as extra regressors in Prophet (:pr:`6988`) :user:`wpdonders`
* [BUG] Fix ``Prophet`` ``_get_fitted_params ``error when the timeseries is constant (:pr:`7011`) :user:`felipeangelimvieira`

Contributors
~~~~~~~~~~~~

:user:`Abhay-Lejith`,
:user:`felipeangelimvieira`,
:user:`fkiraly`,
:user:`geetu040`,
:user:`marrov`,
:user:`meraldoantonio`,
:user:`ninedigits`,
:user:`pranavvp16`,
:user:`SaiRevanth25`,
:user:`shlok191`,
:user:`toandaominh1997`,
:user:`wirrywoo`,
:user:`wpdonders`,
:user:`yarnabrina`


Version 0.32.1 - 2024-08-12
---------------------------

Hotfix release for using ``make_reduction`` with not fully ``sklearn`` compliant
tabular regressors such as from ``catboost``.

For last non-maintenance content updates, see 0.31.1.

Contents
~~~~~~~~

* [BUG] fix ``make_reduction`` type inference for non-sklearn estimators


Version 0.32.0 - 2024-08-11
---------------------------

Maintenance release, with scheduled deprecations and change actions.

For last non-maintenance content updates, see 0.31.1.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``skpro`` (soft dependency) bounds have been updated to ``>=2,<2.6.0``
* ``skforecast`` (forecasting soft dependency) bounds have been updated to ``<0.14.0``.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

* all ``sktime`` estimators and objects are now required to have at least
  two test parameter sets in
  ``get_test_params`` to be compliant with ``check_estimator`` contract tests.
  This requirement was previously stated in the extension template but not enforced.
  It is now also included in the automated tests via ``check_estimator``.
  Estimators without (unreserved) parameters, i.e., where two
  distinct parameter sets are not possible, are excepted from this.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

* From ``sktime 0.38.0``, forecasters' ``predict_proba`` will
  require ``skpro`` to be present in the python environment,
  for distribution objects to represent distributional forecasts.
  Until ``sktime 0.35.0``, ``predict_proba`` will continue working without ``skpro``,
  defaulting to return objects in ``sktime.proba`` if ``skpro`` is not present.
  From ``sktime 0.35.0``, an error will be raised upon call of
  forecaster ``predict_proba`` if ``skpro`` is not present
  in the environment.
  Users of forecasters' ``predict_proba`` should ensure
  that ``skpro`` is installed in the environment.

* The probability distributions module ``sktime.proba`` deprecated and will
  be fully replaced by ``skpro`` in ``sktime 0.38.0``.
  Until ``sktime 0.38.0``, imports from ``sktime.proba`` will continue working,
  defaulting to ``sktime.proba`` if ``skpro`` is not present,
  otherwise redirecting imports to ``skpro`` objects.
  From ``sktime 0.35.0``, an error will be raised if ``skpro`` is not present
  in the environment, otherwise imports are redirected to ``skpro``.
  Direct or indirect users of ``sktime.proba`` should ensure ``skpro`` is
  installed in the environment.
  Direct users of the ``sktime.proba`` module should,
  in addition, replace any imports from
  ``sktime.proba`` with imports from ``skpro.distributions``.

Contents
~~~~~~~~

* [MNT] 0.32.0 deprecations and change actions (:pr:`6916`) :user:`fkiraly`
* [MNT] [Dependabot](deps): Update ``skpro`` requirement from ``<2.5.0,>=2`` to ``>=2,<2.6.0`` (:pr:`6897`) :user:`dependabot[bot]`
* [MNT] remove ``numpy 2`` incompatibility flag from ``numba`` based estimators (:pr:`6915`) :user:`fkiraly`
* [MNT] isolate ``joblib`` (:pr:`6385`) :user:`fkiraly`
* [MNT] handle more ``pandas`` deprecations (:pr:`6941`) :user:`fkiraly`
* [MNT] deprecation of ``proba`` module in favour of ``skpro`` soft dependency (:pr:`6940`) :user:`fkiraly`
* [MNT] update versions of ``pre-commit`` hooks (:pr:`6947`) :user:`yarnabrina`
* [MNT] 0.32.0 release action - revert temporary skip ``get_test_params`` number check for 0.21.1 and 0.22.0 release (:pr:`5114`) :user:`fkiraly`
* [MNT] Bump ``skforecast`` to ``0.13`` version allowing support for python ``3.12`` (:pr:`6946`) :user:`yarnabrina`
* [BUG] Fix ``Xt_msg`` type in ``tranformations.base`` (:pr:`6944`) :user:`hliebert`

Contributors
~~~~~~~~~~~~

:user:`fkiraly`,
:user:`hliebert`,
:user:`yarnabrina`


Version 0.31.2 - 2024-08-13
---------------------------

Hotfix release, released after hotfix release 0.32.1,
to apply the same hotfix to 0.31.X versions as well.

Hotfix for using ``make_reduction`` with not fully ``sklearn`` compliant
tabular regressors such as from ``catboost``.

For last non-maintenance content updates, see 0.31.1.

Contents
~~~~~~~~

* [BUG] fix ``make_reduction`` type inference for non-sklearn estimators

Notes
^^^^^

This is a hotfix for 0.31.1 release, fixing a regression. This release is not contained
in the 0.32.0 or 0.32.1 releases.


Version 0.31.1 - 2024-08-10
---------------------------

Highlights
~~~~~~~~~~

* html representation of objects now has a button linking to documentation page (:pr:`6876`) :user:`mateuszkasprowicz`
* interface to TinyTimeMixer foundation model (:pr:`6712`) :user:`geetu040`
* interface to ``autots`` ensemble (:pr:`5948`) :user:`MBristle`
* interface to  ``darts`` reduction models (:pr:`6712`) :user:`fnhirwa`, :user:`yarnabrina`
* ``LTSFTransformer`` based on ``cure-lab`` research code base (:pr:`6202`) :user:`geetu040`
* MVTS transformer classifier (:pr:`6791`) :user:`geetu040`
* forecasters can now support categorical ``X``, as per tag (:pr:`6704`, :pr:`6732`) :user:`Abhay-Lejith`
* ``DirectReductionForecaster`` now has a ``windows_identical`` option (:pr:`6650`) :user:`hliebert`
* ``ForecastingOptunaSearchCV`` can now be passed custom samplers and "higher is better" scores (:pr:`6823`, :pr:`6846`) :user:`bastisar`, :user:`gareth-brown-86`, :user:`mk406`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``holiday`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.54``
* ``dask`` (data container and parallelization back-end) bounds have been updated to ``<2024.8.1``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* implementers no longer need to set the ``package_import_alias`` tag
  when estimator dependencies have a different import name than the PEP 440 package name.
  All internal logic now only uses the PEP 440 package name.
  There is no need to remove the tag if already set, but it is no longer required.
* estimators now have a tag ``capability:categorical_in_X: bool`` to indicate
  that the estimator can handle categorical features in the input data ``X``.
  Such estimator can be used with categorical and string-valued features
  if ``X`` is passed in one of the ``pandas`` based mtypes.
* the html representation of all objects now includes a link to the documentation
  of the object, and is now in line with the ``sklearn`` html representation.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] improved environment package version check (:pr:`6776`) :user:`fkiraly`
* [ENH] Remove package import alias related internal logic and tags (:pr:`6821`) :user:`fkiraly`
* [ENH] Adding tag for categorical support in ``X`` (:pr:`6704`) :user:`Abhay-Lejith`
* [ENH] Adding categorical support: Raising error in yes/no case (:pr:`6732`) :user:`Abhay-Lejith`
* [ENH] Link to docs in object's html repr (:pr:`6876`) :user:`mateuszkasprowicz`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Data Loader for M5 dataset (:pr:`6731`) :user:`SaiRevanth25`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``check_pdmultiindex_panel`` to return names of invalid ``object`` columns if there are any (:pr:`6797`) :user:`SaiRevanth25`
* [ENH] Allow object dtype in series (:pr:`5886`) :user:`yarnabrina`
* [ENH] converter framework tests in ``datatypes`` to cover all types, including those requiring soft dependencies (:pr:`6838`) :user:`fkiraly`
* [ENH] add missing ``feature_kind`` metadata fields to ``gluonts`` based data container checkers (:pr:`6861`) :user:`fkiraly`
* [ENH] added ``feature_kind`` metadata in datatype checks (:pr:`6490`) :user:`Abhay-Lejith`
* [ENH] Adding support for ``gluonts`` ``PandasDataset`` object (:pr:`6668`) :user:`shlok191`
* [ENH] Added support for ``gluonts`` ``PandasDataset`` as a ``Series`` scitype (:pr:`6837`) :user:`shlok191`

Forecasting
^^^^^^^^^^^

* [ENH] interface to ``autots`` ensemble (:pr:`5948`) :user:`MBristle`
* [ENH] ``darts`` Reduction Models adapter (:pr:`6712`) :user:`fnhirwa`, :user:`yarnabrina`
* [ENH] Extension Template For Global Forecasting API (:pr:`6699`) :user:`XinyuWuu`
* [ENH] enable multivariate data passed to ``autots`` interface (:pr:`6805`) :user:`fkiraly`
* [ENH] Add Sampler to ``ForecastingOptunaSearchCV`` (:pr:`6823`) :user:`bastisar`
* [ENH] Improve ``TestAllGlobalForecasters`` (:pr:`6845`) :user:`XinyuWuu`
* [ENH] Add scoring direction to ``ForecastingOptunaSearchCV`` (:pr:`6846`) :user:`gareth-brown-86`, :user:`mk406`
* [ENH] de-novo implementation of ``LTSFTransformer`` based on ``cure-lab`` research code base (:pr:`6202`) :user:`geetu040`
* [ENH] Add ``windows_identical`` to ``DirectReductionForecaster`` (:pr:`6650`) :user:`hliebert`
* [ENH] updates type inference in ``make_reduction`` to use central scitype inference and allow proba tabular regressors (:pr:`6893`) :user:`fkiraly`
* [ENH] DeepAR and  NHiTS and refinements for ``pytorch-forecasting`` interface (:pr:`6551`) :user:`XinyuWuu`
* [ENH] Interface to TinyTimeMixer foundation model (:pr:`6712`) :user:`geetu040`
* [ENH] remove now superfluous try-excepts in forecasting API test suite (:pr:`6906`) :user:`fkiraly`
* [ENH] improve ``test_global_forecasting_tag`` (:pr:`6929`) :user:`geetu040`

Registry and search
^^^^^^^^^^^^^^^^^^^

* [ENH] in estimator html repr, make version retrieval safer and more flexible (:pr:`6923`) :user:`fkiraly`

Time series anomalies, changepoints, segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] time series annotation (outliers, changepoints) - test class and full ``check_estimator`` integration (:pr:`6843`) :user:`fkiraly`
* [ENH] Add Windowed Local Outlier Factor Anomaly Detector (:pr:`6524`) :user:`Alex-JG3`
* [ENH] Add binary segmentation annotator for change point detection (:pr:`6723`) :user:`Alex-JG3`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Pytorch Classifier intermediate base class for TSC (:pr:`6791`) :user:`geetu040`
* [ENH] MVTS transformer classifier (:pr:`6791`) :user:`geetu040`

Transformations
^^^^^^^^^^^^^^^

* [ENH] add second test params dict to ``Aggregator`` (:pr:`6759`) :user:`fr1ll`
* [ENH] ``pandas`` inner type and global pooling for ``TabularToSeriesAdaptor`` (:pr:`6752`) :user:`fkiraly`
* [ENH] alternative returns for ``VmdTransformer`` - mode spectra and central frequencies (:pr:`6857`) :user:`fkiraly`
* [ENH] simplify dictionaries and alias handling in ``Catch22`` (:pr:`6104`) :user:`fkiraly`
* [ENH] making ``self._is_vectorized`` access more defensive in ``BaseTransformer`` (:pr:`6863`) :user:`fkiraly`

Test framework
^^^^^^^^^^^^^^

* [ENH] make ``pyproject.toml`` parsing for differential testing more robust against non-package relevant changes (:pr:`6882`) :user:`fkiraly`

Vendor and onboard libraries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Vendor fracdiff library (:pr:`6777`) :user:`DinoBektesevic`
* [ENH] improvements to vendored ``fracdiff`` library (:pr:`6912`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] Notebook and Template For Global Forecasting API (:pr:`6699`) :user:`XinyuWuu`
* [DOC] Add authorship credits to ``MatrixProfileTransformer`` for Stumpy authors (:pr:`6762`) :user:`alexander-lakocy`
* [DOC] add examples to ``StatsForecastGARCH`` and ``StatsForecastARCH`` docstrings (:pr:`6761`) :user:`melinny`
* [DOC] Add alignment notebook example (:pr:`6768`) :user:`alexander-lakocy`
* [DOC] fix transformers type table in API reference in accordance with sphinx guidelines (:pr:`6771`) :user:`alexander-lakocy`
* [DOC] Modify editable install to make cross-platform (:pr:`6758`) :user:`fr1ll`
* [DOC] ``TruncationTransformer`` docstring example (:pr:`6765`) :user:`ceroper`
* [DOC] De-duplicate User Guide and Examples (closes #6767) (:pr:`6770`) :user:`alexander-lakocy`
* [DOC] improved docstring of ``DWTTransformer`` (:pr:`6764`) :user:`Mitchjkjkjk`
* [DOC] various improvements to user journey on documentation page (:pr:`6760`) :user:`fkiraly`
* [DOC] Time series k means max iter parameter docstring (:pr:`6726`) :user:`AlexeyOm`
* [DOC] cross-reference estimator search from tags API reference (:pr:`6816`) :user:`fkiraly`, :user:`yarnabrina`
* [DOC] updated docstring for ``check_is_mtype`` to match skpro ``check_is_mtype`` function (:pr:`6835`) :user:`julian-fong`
* [DOC] example & tutorial notebooks: normalize execution counts, indentation, execute all cells (:pr:`6847`) :user:`fkiraly`
* [DOC] clarify column handling in docstring of ``FourierFeatures`` (:pr:`6834`) :user:`fkiraly`
* [DOC] added fork usage recommendations (:pr:`6827`) :user:`yarnabrina`
* [DOC] change links in documentation to refer to same version (:pr:`6841`) :user:`yarnabrina`
* [DOC] minor improvements to ``check_scoring`` docstring (:pr:`6877`) :user:`fkiraly`
* [DOC] add proper author credits to 1:1 interface classes - aligners, distances, forecasters, parameter estimators (:pr:`6850`) :user:`fkiraly`
* [DOC] fix docstring formatting of ``evaluate`` (:pr:`6864`) :user:`fkiraly`
* [DOC] Add documentation for benchmarking module (:pr:`6792`) :user:`benHeid`
* [DOC] add elections link on landing page (:pr:`6910`) :user:`fkiraly`
* [DOC] Add example notebook for the graphical pipeline (:pr:`5175`) :user:`benHeid`
* [DOC] git workflow guide - chained branches, fixing header fonts (:pr:`6913`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] Remove ``tbats`` python version constraint (:pr:`6769`) :user:`fr1ll`
* [MNT] Update ``Callable`` import from ``typing`` to ``collections.abc`` (:pr:`6798`) :user:`yarnabrina`
* [MNT] Fix spellings using ``codespell`` and ``typos`` (:pr:`6799`) :user:`yarnabrina`
* [MNT] improved environment package version check (:pr:`6776`) :user:`fkiraly`
* [MNT] downgrade pykan version to ``<0.2.2`` (:pr:`6853`) :user:`geetu040`
* [MNT] add non-unicode characters check to the linter (:pr:`6807`) :user:`fnhirwa`
* [MNT] updates and fixes to type hints (:pr:`6743`) :user:`ZhipengXue97`
* [MNT] Resolve the issue with diacritics failing to be decoded on Windows (:pr:`6862`) :user:`fnhirwa`
* [MNT] sync docstring and code formatting of dependency checker module with ``skbase`` (:pr:`6873`) :user:`fkiraly`
* [MNT] Remove package import alias related internal logic and tags (:pr:`6821`) :user:`fkiraly`
* [MNT] restrict failing Mr-SEQL version (:pr:`6879`) :user:`fkiraly`
* [MNT] release workflow: Upgrade deprecated pypa action parameter (:pr:`6878`) :user:`szepeviktor`
* [MNT] Fix ``pykan`` import and dependency checks (:pr:`6881`) :user:`fkiraly`
* [MNT] temporarily pin ``matplotlib`` below ``3.9.1`` (:pr:`6890`) :user:`yarnabrina`
* [MNT] make ``pyproject.toml`` parsing for differential testing more robust against non-package relevant changes (:pr:`6882`) :user:`fkiraly`
* [MNT] formatter for jupyter notebook json in build tools (:pr:`6849`) :user:`fkiraly`
* [MNT] sync differential testing utilities with ``skpro`` (:pr:`6840`) :user:`fkiraly`
* [MNT] Handle deprecations from ``pandas`` (:pr:`6855`) :user:`fkiraly`
* [MNT] sync docstring and code formatting of dependency checker module with ``skbase`` (:pr:`6873`) :user:`fkiraly`
* [MNT] fix ``.all-contributorsrc`` syntax (:pr:`6918`) :user:`fkiraly`
* [MNT] Resolve the issue with diacritics failing to be decoded on Windows (:pr:`6862`) :user:`fnhirwa`
* [MNT] changelog utility: fix termination condition to retrieve merged PR (:pr:`6920`) :user:`fkiraly`
* [MNT] restore ``holidays`` lower bound to ``0.29`` (:pr:`6921`) :user:`fkiraly`
* [MNT] Updating the GHA dependencies to install OSX dependencies and setting the compiler flags (:pr:`6926`) :user:`fnhirwa`
* [MNT] revert an erroneous instance of ``pandas`` deprecation fix (:pr:`6925`) :user:`fkiraly`
* [MNT] Update the path to script to fix #6926 (:pr:`6933`) :user:`fnhirwa`
* [MNT] [Dependabot](deps): Update ``pytest`` requirement from ``<8.3,>=7.4`` to ``>=7.4,<8.4`` (:pr:`6819`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``dask`` requirement from ``<2024.6.3`` to ``<2024.7.2`` (:pr:`6818`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``sphinx-gallery`` requirement from ``<0.17.0`` to ``<0.18.0`` (:pr:`6820`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``holidays`` requirement from ``<0.53,>=0.52`` to ``>=0.52,<0.54`` (:pr:`6780`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update sphinx requirement from ``!=7.2.0,<8.0.0`` to ``!=7.2.0,<9.0.0`` (:pr:`6865`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``holidays`` requirement from ``<0.54,>=0.52`` to ``>=0.52,<0.55`` (:pr:`6898`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``dask`` requirement from ``<2024.7.2`` to ``<2024.8.1`` (:pr:`6907`) :user:`dependabot[bot]`

Fixes
~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``_check_soft_dependencies`` for post and pre versions of patch versions (:pr:`6909`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix type inconsistency in conversion ``pandas`` to ``xarray`` based ``Series`` (:pr:`6856`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] Fix ``pykan`` dependency and set lower bound (:pr:`6789`) :user:`benHeid`
* [BUG] correct dependency tag for ``pytorch-forecasting`` forecasters: rename ``pytorch_forecasting`` to correct package name ``pytorch-forecasting`` (:pr:`6830`) :user:`XinyuWuu`

Registry and search
^^^^^^^^^^^^^^^^^^^

* [BUG] fix polymorphic estimators missing in estimator overview, e.g., ``pytorch-forecasting`` forecasters (:pr:`6803`) :user:`fkiraly`

Time series anomalies, changepoints, segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix bug when predicting segments from clasp change point annotator (:pr:`6756`) :user:`Alex-JG3`

Transformations
^^^^^^^^^^^^^^^

* [BUG] Refactor ``ADICVTransformer`` and fix CV calculation (:pr:`6757`) :user:`sbhobbes`
* [BUG] fix ``BaseTransformer`` broadcasting condition in ``inverse_transform`` for decomposers (:pr:`6824`) :user:`fkiraly`
* [BUG] fix ``MSTL`` inverse transform and use in forecasting pipeline (:pr:`6825`) :user:`fkiraly`
* [BUG] fix handling of ``numpy`` integers in refactored ``Catch22`` transformation (:pr:`6934`) :user:`fkiraly`

Visualization
^^^^^^^^^^^^^

* [BUG] In ``plot_series``, trim unused levels when verifying dataframe formatting (:pr:`6754`) :user:`SultanOrazbayev`

Contributors
~~~~~~~~~~~~

:user:`Abhay-Lejith`,
:user:`Alex-JG3`,
:user:`alexander-lakocy`,
:user:`AlexeyOm`,
:user:`bastisar`,
:user:`benHeid`,
:user:`ceroper`,
:user:`DinoBektesevic`,
:user:`fkiraly`,
:user:`fnhirwa`,
:user:`fr1ll`,
:user:`gareth-brown-86`,
:user:`geetu040`,
:user:`hliebert`,
:user:`julian-fong`,
:user:`mateuszkasprowicz`,
:user:`MBristle`,
:user:`melinny`,
:user:`Mitchjkjkjk`,
:user:`mk406`,
:user:`SaiRevanth25`,
:user:`sbhobbes`,
:user:`shlok191`,
:user:`SultanOrazbayev`,
:user:`szepeviktor`,
:user:`XinyuWuu`,
:user:`yarnabrina`,
:user:`ZhipengXue97`


Version 0.31.0 - 2024-07-11
---------------------------

Maintenance release:

* scheduled deprecations and change actions
* ``numpy 2`` compatibility
* code style and pre-commit updates, using ``ruff`` for linting

For last non-maintenance content updates, see 0.30.2.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``numpy`` (core dependency) bounds have been updated to ``<2.1,>=1.21``
* ``skpro`` (soft dependency) bounds have been updated to ``>=2,<2.5.0``

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Time series anomalies, changepoints, segmentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The ``fmt`` argument in time series annotators is now deprecated.
  Users should use the ``predict`` and ``transform`` methods instead,
  ``predict`` instead of ``fmt="sparse"``, and ``transform`` instead of
  ``fmt="dense"``.

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* The ``convert_y_to_keras`` method in deep learning classifiers has been removed.
  Users who have been using this method should
  instead use ``OneHotEncoder`` from ``sklearn`` directly, as ``convert_y_to_keras``
  is a simple wrapper around ``OneHotEncoder`` with default settings.

Contents
~~~~~~~~

* [MNT] raise ``numpy`` bound to ``numpy < 2.1``, ``numpy 2`` compatibility (:pr:`6624`) :user:`fkiraly`
* [MNT] [Dependabot](deps): Update skpro requirement from ``<2.4.0,>=2`` to ``>=2,<2.5.0`` (:pr:`6663`) :user:`dependabot[bot]`
* [MNT] bound ``prophet`` based forecasters to ``numpy<2`` due to incompatibility of ``prophet`` (:pr:`6721`) :user:`fkiraly`
* [MNT] further ``numpy 2`` compatibility fixes in estimators (:pr:`6729`) :user:`fkiraly`
* [MNT] handle ``numpy 2`` incompatible soft deps (:pr:`6728`) :user:`fkiraly`
* [MNT] Upgrade code style beyond ``python 3.8`` (:pr:`6330`) :user:`yarnabrina`
* [MNT] Update pre commit hooks post dropping ``python 3.8`` support (:pr:`6331`) :user:`yarnabrina`
* [MNT] suppress aggressive ``freq`` related warnings from ``pandas 2.2`` (:pr:`6733`) :user:`fkiraly`
* [MNT] 0.31.0 deprecations and change actions (:pr:`6716`) :user:`fkiraly`
* [MNT] switch to ``ruff`` as linting tool (:pr:`6676`) :user:`fnhirwa`
* [ENH] refactor and bugfixes for environment checker utilities (:pr:`6719`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`fkiraly`,
:user:`fnhirwa`,
:user:`yarnabrina`


Version 0.30.2 - 2024-07-04
---------------------------

Highlights
~~~~~~~~~~

* new `estimator overview table and estimator search page <https://www.sktime.net/en/stable/estimator_overview.html>`_ (:pr:`6147`) :user:`duydl`
* ``HFTransformersForecaster`` (hugging face transformers connector) now has a user friendly interface for applying PEFT methods (:pr:`6457`) :user:`geetu040`
* ``ForecastingOptunaSearchCV`` for hyper-parameter tuning of forecasters via ``optuna`` (:pr:`6630`) :user:`mk406`, :user:`gareth-brown-86`
* ``prophetverse`` package forecasters are now indexed by ``sktime`` (:pr:`6614`) :user:`felipeangelimvieira`
* ``pytorch-forecasting`` adapter, experimental global forecasting API (:pr:`6228`) :user:`XinyuWuu`
* ``skforecast`` adapter for reduction strategies (:pr:`6531`) :user:`Abhay-Lejith`, :user:`yarnabrina`
* EnbPI based forecaster with components from ``aws-fortuna`` (:pr:`6449`) :user:`benHeid`
* DTW distances and aligners from ``dtaidistance`` (:pr:`6578`) :user:`fkiraly`
* ``parametrize_with_checks`` utility for granular API compliance test setup in 2nd/3rd party libraries (:pr:`6588`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``holidays`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.53``
* ``dask`` (data container and parallelization back-end) bounds have been updated to ``<2024.5.3``
* ``optuna`` is now a soft dependency, via the ``ForecastingOptunaSearchCV`` estimator, in the ``all_extras`` soft dependency set,
  with bounds ``<3.7``
* ``pytorch-forecasting`` is now a soft dependency, in the ``dl`` (deep learning) soft dependency set
* ``skforecast`` is now a soft dependency, in the ``all_extras`` soft dependency set and the ``forecasting`` soft dependency set,
  with bounds ``<0.13,>=0.12.1``
* ``dtaidistance`` is now a soft dependency, in the ``all_extras`` soft dependency set and the ``alignment`` soft dependency set,
  with bounds ``<2.4``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

The base forecaster interface now has a dedicated interface point for
global forecasting or fine-tuning: in forecasters supporting global forecast,
an ``y`` argument may be passed in ``predict``, indicating new time series instances
for a global forecast, or a context for foundation models.
Forecasters capable of global forecasting or fine-tuning (this is the same interface
point) are tagged with the tag ``capability:global_forecasting``, value ``True``.

The global forecasting and fine-tuning interfaces are currently experimental,
and may undergo changes.

Users are invited to give feedback, and test the feature with the new
``pytorch-forecasting`` adapter.

Test framework
^^^^^^^^^^^^^^

* 2nd and 3rd party extension packages can now use the ``parametrize_with_checks``
  utility to set up granular API compliance tests. For detailed usage notes,
  consult the extender guide: :ref:`developer_guide_add_estimators`.
* various quality-of-life improvements have been made to facilitate
  indexing an estimator in the estimator overview and estimator search for
  developers of API compatible 2nd and 3rd party packages,
  without adding it directly to the main ``sktime`` repository.
  For detailed usage notes, consult the extender guide:
  :ref:`developer_guide_add_estimators`, or inspect the ``Prophetverse`` forecaster
  as a worked example.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] prevent imports caused by ``_check_soft_dependencies``, speed up dependency check and test collection time (:pr:`6355`) :user:`fkiraly`, :user:`yarnabrina`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Parallelization option for ``ForecastingBenchmark`` (:pr:`6568`) :user:`benHeid`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Added GluonTS datasets as ``sktime`` mtypes (:pr:`6530`) :user:`shlok191`

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [ENH] DTW distances from ``dtaidistance`` (:pr:`6578`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] ``pytorch-forecasting`` adapter with Global Forecasting API (:pr:`6228`) :user:`XinyuWuu`
* [ENH] fitted parameter forwarding utility, forward ``statsforecast`` estimators' fitted parameters (:pr:`6349`) :user:`fkiraly`
* [ENH] EnbPI based forecaster with components from ``aws-fortuna`` (:pr:`6449`) :user:`benHeid`
* [ENH] ``skforecast`` ForecasterAutoreg adapter  (:pr:`6531`) :user:`Abhay-Lejith`, :user:`yarnabrina`
* [ENH] Extend ``HFTransformersForecaster`` for PEFT methods (:pr:`6457`) :user:`geetu040`
* [ENH] in ``BaseForecaster``, move check for ``capability:insample`` to ``_check_fh`` boilerplate (:pr:`6593`) :user:`XinyuWuu`
* [ENH] indexing ``prophetverse`` forecaster (:pr:`6614`) :user:`fkiraly`
* [ENH] ``ForecastingOptunaSearchCV`` for hyper-parameter tuning of forecasters via ``optuna`` (:pr:`6630`) :user:`mk406`, :user:`gareth-brown-86`

Registry and search
^^^^^^^^^^^^^^^^^^^

* [ENH] enhanced estimator overview table - tag display and search (:pr:`6147`) :user:`duydl`, :user:`fkiraly`

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [ENH] DTW aligners from ``dtaidistance`` (:pr:`6578`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] resolve duplication in KNeighborsClassifier and KNeighborsRegressor (:pr:`6504`) :user:`Z-Fran`
* [ENH] added two test params sets to ``FCNNetwork`` (:pr:`6562`) :user:`TheoWeih`
* [ENH] further refactor of knn classifier and regressor (:pr:`6615`) :user:`fkiraly`
* [ENH] update ``tests._config`` to skip various sporadically failing tests for Proximity Forest and Proximity Tree until fixed (:pr:`6638`) :user:`julian-fong`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Time Series Regression grid search (:pr:`6118`) :user:`ksharma6`
* [ENH] test parameters for ``RocketRegressor`` (:pr:`6149`) :user:`iaryangoyal`
* [ENH] resolve duplication in KNeighborsClassifier and KNeighborsRegressor (:pr:`6504`) :user:`Z-Fran`
* [ENH] further refactor of knn classifier and regressor (:pr:`6615`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] refactor ``WindowSummarizer`` tests (:pr:`6564`) :user:`fkiraly`

Test framework
^^^^^^^^^^^^^^

* [ENH] differential testing for base functionality in various modules (:pr:`6534`) :user:`fkiraly`
* [ENH] further differential testing for the ``transformations`` module (:pr:`6533`) :user:`fkiraly`
* [ENH] differential testing in ``dist_kernels`` and ``clustering`` modules (:pr:`6543`) :user:`fkiraly`
* [ENH] simplify and add differential testing to ``forecasting.compose.tests`` module (:pr:`6563`) :user:`fkiraly`
* [ENH] simplify and add differential testing to ``sktime.pipeline`` module (:pr:`6565`) :user:`fkiraly`
* [ENH] differential testing in ``benchmarking`` module (:pr:`6566`) :user:`fkiraly`
* [ENH] move doctests to main test suite to ensure conditional execution (:pr:`6536`) :user:`fkiraly`
* [ENH] minor improvements to test efficiency (:pr:`6586`) :user:`fkiraly`
* [ENH] ``parametrize_with_checks`` utility for granular API compliance test setup in 2nd/3rd party libraries (:pr:`6588`) :user:`fkiraly`
* [ENH] differential testing to ``utils`` module (:pr:`6620`) :user:`fkiraly`
* [ENH] differential testing and minor improvements to ``forecasting.base`` tests (:pr:`6619`) :user:`fkiraly`
* [ENH] differential testing for ``performance_metrics`` module (:pr:`6616`) :user:`fkiraly`
* [ENH] fixes and improvements to ``pytest`` ``doctest`` integration (:pr:`6621`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] fix broken links on webpage docs (:pr:`6339`) :user:`duydl`
* [DOC] document more tags (:pr:`6496`) :user:`fkiraly`
* [DOC] fix minor typos in tags API reference (:pr:`6631`) :user:`fkiraly`
* [DOC] update dependencies reference (:pr:`6655`) :user:`emmanuel-ferdman`
* [DOC] fix minor typo in developer comment in ``BaseTransformer`` (:pr:`6689`) :user:`Spinachboul`
* [DOC] rst roadmap documentation page stale since 2021 - replace by correct links to recent roadmaps (:pr:`6556`) :user:`fkiraly`
* [DOC] clarify docs on ARIMA estimators, add author credits for upstream (:pr:`6705`) :user:`fkiraly`
* [DOC] credit :user:`doberbauer`` for ``pykalman`` python 3.11 compatibility fix (:pr:`6662`) :user:`doberbauer`

Maintenance
~~~~~~~~~~~

* [MNT] [Dependabot](deps): Update holidays requirement from ``<0.51,>=0.29`` to ``>=0.29,<0.52`` (:pr:`6634`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update holidays requirement from ``<0.52,>=0.29`` to ``>=0.52,<0.53`` (:pr:`6702`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update dask requirement from ``<2024.6.1`` to ``<2024.6.2`` (:pr:`6643`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update numba requirement from ``<0.60,>=0.53`` to ``>=0.53,<0.61`` (:pr:`6590`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update dask requirement from ``<2024.6.2`` to ``<2024.6.3`` (:pr:`6647`) :user:`dependabot[bot]`
* [MNT] remove coverage reporting and ``pytest-cov`` from PR CI and ``setup.cfg`` (:pr:`6363`) :user:`fkiraly`
* [MNT] ``numpy 2`` compatibility fixes - estimators (:pr:`6626`) :user:`fkiraly`
* [MNT] ``scipy`` ``1.14.0`` compatibility for ``deep_equals`` plugin for ``csr_matrix`` (:pr:`6664`) :user:`fkiraly`
* [MNT] deprecate unused ``_check_soft_dependencies`` argument ``suppress_import_stdout`` (:pr:`6691`) :user:`fkiraly`

Fixes
~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``AUCalibration`` probabilistic metric for ``multivariate`` case (:pr:`6617`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [BUG] fix bug 4076: ``PerformanceWarning`` in ``load_from_tsfile_to_dataframe`` (:pr:`6632`) :user:`ericjb`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] patch over ``pandas 2.2.X`` issue in ``freq`` timestamp/period round trip conversion for period start timestamps such as ``"MonthBegin"`` (:pr:`6574`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] fix passing of ``y`` in ``ForecastingPipeline`` (:pr:`6706`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix bug in fitted parameter override in ``pyts`` and ``tslearn`` adapters (:pr:`6707`) :user:`fkiraly`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix bug in fitted parameter override in ``pyts`` and ``tslearn`` adapters (:pr:`6707`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] in ``TimeSeriesForestRegressor``, fix failure: ``self.criterion`` does not exist (:pr:`6573`) :user:`ksharma6`

Test framework
^^^^^^^^^^^^^^

* [BUG] partially revert ``pytest.skip`` change from #6233 due to side effects in downstream test suites (:pr:`6508`) :user:`fkiraly`
* [BUG] fix test failures introduced by differential testing refactor (:pr:`6585`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix ``HolidayFeatures`` crashes if dataframe doesn't contain specified date  (:pr:`6550`) :user:`fnhirwa`
* [BUG] in ``Differencer``, make explicit clone to avoid ``SettingWithCopyWarning`` (:pr:`6567`) :user:`benHeid`
* [BUG] minirocket: fix zero division errors #5174 (:pr:`6612`) :user:`benshaw2`
* [BUG] ensure correct setting of ``requires_X`` and ``requires_y`` tag for ``FeatureUnion`` (:pr:`6695`) :user:`fkiraly`
* [BUG] ensure correct setting of ``requires_X`` and ``requires_y`` tag for ``TransformerPipeline`` (:pr:`6692`) :user:`fkiraly`
* [BUG] partial fix for dropped column names in ``PaddingTransformer`` (:pr:`6693`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`Abhay-Lejith`,
:user:`benHeid`,
:user:`benshaw2`,
:user:`doberbauer`,
:user:`emmanuel-ferdman`,
:user:`ericjb`,
:user:`felipeangelimvieira`,
:user:`fkiraly`,
:user:`fnhirwa`,
:user:`gareth-brown-86`,
:user:`geetu040`,
:user:`iaryangoyal`,
:user:`julian-fong`,
:user:`ksharma6`,
:user:`mk406`,
:user:`shlok191`,
:user:`Spinachboul`,
:user:`TheoWeih`,
:user:`XinyuWuu`,
:user:`yarnabrina`,
:user:`Z-Fran`


Version 0.30.1 - 2024-06-04
---------------------------

Minimal maintenance update with actions consolidating onboard packages.

For last major feature update, see 0.29.1.

Contents
~~~~~~~~

* [MNT] reorganization of onboard libs - ``pykalman``, ``vmdpy`` (:pr:`6535`) :user:`fkiraly`
* [MNT] differential testing for ``split`` module (:pr:`6532`) :user:`fkiraly`


Version 0.30.0 - 2024-06-03
---------------------------

Major upgrade to the time series anomaly, changepoints, segmentation API (:user:`Alex-JG3`).
Users should review the section in the release notes.

Kindly also note the python 3.8 End-of-life warning below.

Also includes scheduled deprecations and change actions.

For last major feature update, see 0.29.1.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``joblib`` is now an explicit core dependency, with bounds ``<1.5,>=1.2.0``.
  Previously, ``joblib`` was an indirect core dependency, via ``scikit-learn``.
  Due to direct imports, this was changed to an explicit dependency.

* ``scikit-learn`` (core dependency) bounds have been updated to ``>=0.24,<1.6.0``

* ``scikit-base`` (core dependency) bounds have been updated to ``>=0.6.1,<0.9.0``

* ``skpro`` (soft dependency) bounds have been updated to ``>=2,<2.4.0``

* ``kotsu`` is not longer a soft dependency required by the forecasting benchmarking
  framework. The ``kotsu`` package is no longer maintained,
  and its necessary imports have beend moved
  to ``sktime`` as private utilities until refactor. See :pr:`6514`.

* ``pykalman`` (transformations soft dependency) has been forked into ``sktime``,
  as ``sktime.libs.pykalman``, as the original package is no longer maintained,
  see ``sktime`` issue 5414 or ``pykalman`` issue 109.

  * The package fork will be maintained in ``sktime``.
  * Direct users of ``pykalman`` can replace imports ``from pykalman import x``
    with equivalent imports ``from sktime.libs.pykalman import x``.
  * Indirect users via the transformer ``KalmanFilterTransformerPK`` will not be
    impacted as APIs do not change, except that they no longer require
    the original ``pykalman`` package in their python environment.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

The time series annotation, anomalies, changepoints, segmentation API has been
fully reworked to be in line with ``scikit-base`` patterns, ``sktime`` tags,
and to provide a more consistent and flexible interface.

* the API provides ``predict`` methods for annotation labels, e.g., segments,
  outlier points, and a ``transform`` method for indicator series, for instance
  1/0 indicator whether an anomaly is present at the time stamp.
* the ``fmt`` argument used in some estimators is now deprecated,
  in favour of using ``predict`` or ``transform``.
* The type of annotation, e.g., change points or segmentation, is
  encoded by the new tag ``task`` used in time series annotators,
  with values ``anomaly_detection``, ``segmentation``, ``changepoint_detection``.
* Low-level methods allow polymorphic use of annotators, e.g., a changepoint detector
  to be used for segmentation, via ``predict_points`` or ``predict_segments``.
  The ``predict`` method defaults to the type of annotation defined by ``task``.

A full tutorial with examples will be created over the next release cycles,
and further enhancements are planned.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.8 End-of-life
^^^^^^^^^^^^^^^^^^^^^^

``sktime`` now requires Python version ``>=3.9``.
No errors will be raised on Python 3.8, but test coverage and support for
Python 3.8 has been dropped.

Kindly note for context: python 3.8 will reach end of life
in October 2024, and multiple ``sktime`` core dependencies,
including ``scikit-learn``, have already dropped support for 3.8.

Forecasting
^^^^^^^^^^^

``cINNForecaster`` has been renamed to ``CINNForecaster``.
The estimator is no longer available under its old name,
after the deprecation period.
Users should replace any imports of ``cINNForecaster``
with imports of ``CINNForecaster``.

Enhancements
~~~~~~~~~~~~

* [ENH] Rework of base series annotator API (:pr:`6265`) :user:`Alex-JG3`
* [ENH] upgrade ``is_module_changed`` test utility for paths (:pr:`6518`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] updated ``all_estimators`` docstring for ``re.Pattern`` support (:pr:`6478`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] [Dependabot](deps): Update skpro requirement from ``<2.3.0,>=2`` to ``>=2,<2.4.0`` (:pr:`6443`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update scikit-learn requirement from ``<1.5.0,>=0.24`` to ``>=0.24,<1.6.0`` (:pr:`6462`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update scikit-base requirement from ``<0.8.0,>=0.6.1`` to ``>=0.6.1,<0.9.0`` (:pr:`6488`) :user:`dependabot[bot]`
* [MNT] drop test coverage on python 3.8 in CI (:pr:`6329`) :user:`yarnabrina`
* [MNT] final change cycle (0.30.0) for renaming ``cINNForecaster`` to ``CINNForecaster`` (:pr:`6367`) :user:`geetu040`
* [MNT] added ``joblib`` as core dependency (:pr:`6384`) :user:`yarnabrina`
* [MNT] 0.30.0 deprecations and change actions (:pr:`6468`) :user:`fkiraly`
* [MNT] modified CRLF line endings to LF line endings (:pr:`6512`) :user:`yarnabrina`
* [MNT] Move dependency checkers to separate module in ``utils`` (:pr:`6354`) :user:`fkiraly`
* [MNT] resolution to ``pykalman`` issue - ``sktime`` local pykalman fork (:pr:`6188`) :user:`fkiraly`
* [MNT] add systematic differential test switch to low-level tests (:pr:`6511`) :user:`fkiraly`
* [MNT] isolate ``utils`` module init and ``sktime`` init from external imports (:pr:`6516`) :user:`fkiraly`
* [MNT] preparing refactor of benchmark framework: folding minimal ``kotsu`` library into ``sktime`` (:pr:`6514`) :user:`fkiraly`
* [MNT] run tests in ``distances`` module only if it has changed (:pr:`6517`) :user:`fkiraly`
* [MNT] refactor ``pykalman`` tests to ``pytest`` and conditional execution (:pr:`6519`) :user:`fkiraly`
* [MNT] conditional execution of tests in ``datatypes`` module (:pr:`6520`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`Alex-JG3`,
:user:`dependabot[bot]`,
:user:`fkiraly`,
:user:`geetu040`,
:user:`yarnabrina`


Version 0.29.1 - 2024-05-30
---------------------------

Highlights
~~~~~~~~~~

* ``TransformSelectForecaster`` to apply different forecasters depending on series type (e.g., intermittent, lumpy) (:pr:`6453`) :user:`shlok191`
* Kolmogorov-Arnold Network (KAN) forecaster (:pr:`6386`) :user:`benHeid`
* New probabilistic forecast metrics: interval width (sharpness), area under the
  calibration curve (:pr:`6437`, :pr:`6460`) :user:`fkiraly`
* Data loader for fpp3 (Forecasting, Princniples and Practice) datasets via ``rdata`` package, in ``sktime`` data formats (:pr:`6477`) :user:`ericjb`
* Bollinger Bands transformation (:pr:`6473`) :user:`ishanpai`
* ADI/CV2 (Syntetos/Boylan) feature extractor (:pr:`6336`) :user:`shlok191`
* ``ExpandingCutoffSplitter`` - splitter by moving cutoff (:pr:`6360`) :user:`ninedigits`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``holidays`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.50``
* ``pycatch22`` (transformations soft dependency) bounds have been updated to ``<0.4.6``
* ``dtw-python`` (distances and alignment soft dependency) bounds have been updated to ``>=1.3,<1.6``
* ``dask`` (data container and parallelization back-end) bounds have been updated to ``<2024.5.2``
* ``transformers`` (forecasting soft dependency) bounds have been updated to ``<4.41.0``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* all metrics for point forecasts now support weighting, via the ``sample_weight`` parameter.
  If passed, the metric will be weighted by the sample weights.
  For hierarchical data, the weights are applied to the series level,
  in this case all series need to have same length.
  Probabilistic metrics do not support weighting yet, this will be added in a future release.

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* all time series aligners now possess the ``capability:unequal_length`` tag,
  which is ``True`` if the aligner can handle time series of unequal length,
  and ``False`` otherwise. An informative error message, based on the tag,
  is now raised if an aligner not supporting unequal length time series is used on such data.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* The ``convert_y_to_keras`` method in deep learning classifiers has been deprecated and
  will be removed in 0.31.0. Users who have been using this method should
  instead use ``OneHotEncoder`` from ``sklearn`` directly, as ``convert_y_to_keras``
  is a simple wrapper around ``OneHotEncoder`` with default settings.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``ExpandingCutoffSplitter`` - splitter by moving cutoff (:pr:`6360`) :user:`ninedigits`
* [ENH] Interval width (sharpness) metric (:pr:`6437`) :user:`fkiraly`
* [ENH] unsigned area under the calibration curve metric for distribution forecasts (:pr:`6460`) :user:`fkiraly`
* [ENH] forecasting metrics: ensure uniform support and testing for ``sample_weight`` parameter (:pr:`6495`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [ENH] data loader for fpp3 datasets from CRAN via ``rdata`` package, to ``sktime`` data formats (:pr:`6477`) :user:`ericjb`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Polars conversion utilities (:pr:`6455`) :user:`pranavvp16`

Forecasting
^^^^^^^^^^^

* [ENH] Kolmogorov-Arnold Network (KAN) forecaster (:pr:`6386`) :user:`benHeid`
* [ENH] Compositor to apply forecasters depending on series type (e.g., intermittent) (:pr:`6453`) :user:`shlok191`
* [ENH] compatibility of ``ForecastingHorizon`` with ``pandas`` ``freq`` ``2Y`` on ``pandas 2.2.0`` and above (:pr:`6500`) :user:`fkiraly`
* [ENH] add test case for ``ForecastingHorizon``  ``pandas 2.2.X`` compatibility, failure case #6499 (:pr:`6503`) :user:`fkiraly`
* [ENH] remove ``Prophet`` from ``test_differencer_cutoff`` (:pr:`6492`) :user:`fkiraly`
* [ENH] address deprecation and raise error in ``test_differencer_cutoff`` (:pr:`6493`) :user:`fkiraly`

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [ENH] time series aligners capability check at input, tag for unequal length capability (:pr:`6486`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Make deep classifier's ``convert_y_to_keras`` private (:pr:`6373`) :user:`cedricdonie`
* [ENH] classification test scenario with three classes and ``pd-multiindex`` mtype (:pr:`6374`) :user:`fkiraly`
* [ENH] test classifiers on str dtype ``y``, ensure ``predict`` returns same type and labels (:pr:`6428`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Test Parameters for `FinancialHolidaysTransformer` (:pr:`6334`) :user:`sharma-kshitij-ks`
* [ENH] ADI/CV feature extractor (:pr:`6336`) :user:`shlok191`
* [ENH] Bollinger Bands (:pr:`6473`) :user:`ishanpai`

Test framework
^^^^^^^^^^^^^^

* [ENH] enable ``check_estimator`` and ``QuickTester.run_tests`` to work with skip marked ``pytest`` tests (:pr:`6233`) :user:`YelenaYY`
* [ENH] make ``get_packages_with_changed_specs`` safe to mutation of return (:pr:`6451`) :user:`fkiraly`

Visualization
^^^^^^^^^^^^^

* [ENH] ``plot_series`` improved to use ``matplotlib`` conventions;
  ``plot_interval`` can now plot multiple overlaid intervals (:pr:`6416`, :pr:`6501`) :user:`ericjb`


Documentation
~~~~~~~~~~~~~

* [DOC] remove redundant/duplicative classification tutorial notebooks (:pr:`6401`) :user:`fkiraly`
* [DOC] update meetup time to new 1pm slot (:pr:`6402`) :user:`fkiraly`
* [DOC] explanation of ``get_test_params`` in test framework example (:pr:`6434`) :user:`fkiraly`
* [DOC] fix download badges in README (:pr:`6479`) :user:`fkiraly`
* [DOC] improved formatting of transformation docstrings (:pr:`6489`) :user:`fkiraly`
* [DOC] document more tags: transformations (:pr:`6351`) :user:`fkiraly`
* [DOC] Improve docstrings for metrics (:pr:`6419`) :user:`fkiraly`
* [DOC] fixed wrong sentence in the documentation (:pr:`6375`) :user:`helloplayer1`
* [DOC] Correct docstring for conversion functions of ``dask_to_pd`` (:pr:`6439`) :user:`pranavvp16`
* [DOC] Fix hugging face transformers documentation (:pr:`6450``) :user:`benheid`
* [DOC] ``plot_calibration`` docstring - formal explanation of the plot (:pr:`6414`) :user:`fkiraly`
* [DOC] high-level explanation of deprecation policy principles (:pr:`6464`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] [Dependabot](deps): Update holidays requirement from ``<0.49,>=0.29`` to ``>=0.29,<0.50`` (:pr:`6456`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update pycatch22 requirement from ``<0.4.4`` to <0.4.6`` (:pr:`6442`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update sphinx-design requirement from ``<0.6.0`` to ``<0.7.0`` (:pr:`6471`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update dask requirement from ``<2024.5.1`` to ``<2024.5.2`` (:pr:`6444`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update dtw-python requirement from ``<1.5,>=1.3`` to ``>=1.3,<1.6`` (:pr:`6474`) :user:`dependabot[bot]`
* [MNT] include unit tests in ``sktime/tests`` in per module tests (:pr:`6353`) :user:`yarnabrina`
* [MNT] maintenance changes for ``AutoTBATS`` (:pr:`6400`) :user:`yarnabrina`
* [MNT] bound ``transformers<4.41.0`` (:pr:`6447`) :user:`fkiraly`
* [MNT] ``sklearn 1.5.0`` compatibility patch (:pr:`6464`) :user:`fkiraly`
* [MNT] skip doctest for ``all_estimators`` (:pr:`6476`) :user:`fkiraly`
* [MNT] address various deprecation and computation warnings (:pr:`6482`) :user:`fkiraly`
* [MNT] address further deprecation warnings from ``pandas`` (:pr:`6494`) :user:`fkiraly`
* [MNT] fix the docs local build failure due to corrupt notebook (:pr:`6426`) :user:`fnhirwa`

Fixes
~~~~~

Forecasting
^^^^^^^^^^^

* [BUG] fix ``ForecastX`` when ``forecaster_X_exogeneous="complement"`` (:pr:`6433`) :user:`fnhirwa`
* [BUG] Modified VAR code to allow ``predict_quantiles`` of 0.5 (fixes #4742) (:pr:`6441`) :user:`meraldoantonio`

Neural networks
^^^^^^^^^^^^^^^

* [BUG] Remove duplicated ``BaseDeepNetworkPyTorch`` in ``networks.base`` (:pr:`6398`) :user:`luca-miniati`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Resolve ``LSTMFCNClassifier`` changing ``callback`` parameter (:pr:`6239`) :user:`ArthrowAbstract`
* [BUG] fix ``_get_train_probs`` in some classifiers to accept any input data type (:pr:`6377`) :user:`fkiraly`
* [BUG] fix ``BaggingClassifier`` for column subsampling case (:pr:`6429`) :user:`fkiraly`
* [BUG] fix ``ProximityForest``, tree, stump, and ``IndividualBOSS`` returning ``y`` of different type in ``predict`` (:pr:`6432`) :user:`fkiraly`
* [BUG] fix classifier default ``_predict`` returning integer labels always, even if ``fit`` ``y`` was not integer (:pr:`6430`) :user:`fkiraly`
* [BUG] in ``CNNClassifier``, ensure ``filter_sizes`` and ``padding`` is passed on (:pr:`6452`) :user:`fkiraly`
* [BUG] fix ``BaseClassifier.fit_predict`` and ``fit_predict_proba`` for ``pd-multiindex`` mtype (:pr:`6491`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Resolve ``LSTMFCNRegressor`` changing ``callback`` parameter (:pr:`6239`) :user:`ArthrowAbstract`
* [BUG] in ``CNNRegressor``, ensure ``filter_sizes`` and ``padding`` is passed on (:pr:`6452`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix to make ``LabelEncoder`` compatible with ``sktime`` pipelines (:pr:`6458`) :user:`Abhay-Lejith`

Test framework
^^^^^^^^^^^^^^

* [BUG] allow metric classes to be called with ``multilevel`` arg if series is not hierarchical (:pr:`6418`) :user:`fkiraly`
* [BUG] fix ``test_run_test_for_class`` logic check if ``ONLY_CHANGED_MODULES`` flag is ``False`` and all estimator dependencies are present (:pr:`6383`) :user:`fkiraly`
* [BUG] fix ``test_run_test_for_class`` test logic (:pr:`6448`) :user:`fkiraly`

Visualization
^^^^^^^^^^^^^

* [BUG] fix ``xticks`` fore date-like data in ``plot_series`` (:pr:`6416`, :pr:`6501`) :user:`ericjb`

Contributors
~~~~~~~~~~~~

:user:`Abhay-Lejith`,
:user:`ArthrowAbstract`,
:user:`benHeid`,
:user:`cedricdonie`,
:user:`ericjb`,
:user:`fkiraly`,
:user:`fnhirwa`,
:user:`helloplayer1`,
:user:`ishanpai`,
:user:`luca-miniati`,
:user:`meraldoantonio`,
:user:`ninedigits`,
:user:`pranavvp16`,
:user:`sharma-kshitij-ks`,
:user:`shlok191`,
:user:`yarnabrina`,
:user:`YelenaYY`


Version 0.29.0 - 2024-04-28
---------------------------

Kindly note the python 3.8 End-of-life warning below.

Maintenance release:

* scheduled deprecations and change actions
* optimization of test collection speed

For last non-maintenance content updates, see 0.28.1.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``sktime`` now requires ``scikit-base>=0.6.1`` (core dependency), this has changed
  from previously no lower bound.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.8 End-of-life
^^^^^^^^^^^^^^^^^^^^^^

From ``sktime`` 0.30.0, sktime will require Python version >=3.9.
No errors will be raised, but test coverage and support for
Python 3.8 will be dropped from 0.30.0 onwards.

Kindly note for context: python 3.8 will reach end of life
in October 2024, and multiple ``sktime`` core dependencies,
including ``scikit-learn``, have already dropped support for 3.8.

Forecasting
^^^^^^^^^^^

``cINNForecaster`` has been renamed to ``CINNForecaster``.
The estimator is available under its past name at its
current location until 0.30.0, when the old name will be removed.
To prepare for the name change,
replace any imports of ``cINNForecaster`` with imports of ``CINNForecaster``.

Transformations
^^^^^^^^^^^^^^^

* The ``n_jobs`` parameter in the ``Catch22`` transformer has been removed.
  Users should pass parallelization backend parameters via ``set_config`` instead.
  To specify ``n_jobs``, use any of the backends supporting it in the
  ``backend:parallel`` configuration, such as ``"loky"`` or ``"multithreading"``.
  The ``n_jobs`` parameter should be passed via the
  ``backend:parallel:params`` configuration.
  To retain previous behaviour, with a specific setting of ``n_jobs=x``,
  use ``set_config(**{"backend:parallel": "loky", "backend:parallel:params": {"n_jobs": x}})``.

Contents
~~~~~~~~

* [MNT] change cycle (0.29.0) for renaming ``cINNForecaster`` to ``CINNForecaster`` (:pr:`6238`) :user:`geetu040`
* [MNT] ``python 3.8`` End-of-life and ``sktime`` support drop warning (:pr:`6348`) :user:`fkiraly`
* [MNT] speed up test collection - cache differential testing switch utilities (:pr:`6357`) :user:`fkiraly`, :user:`yarnabrina`
* [MNT] temporary skip of estimators involved in timeouts #6344 (:pr:`6361`) :user:`fkiraly`
* [MNT] 0.29.0 deprecations and change actions (:pr:`6350`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`fkiraly`,
:user:`geetu040`,
:user:`yarnabrina`


Version 0.28.1 - 2024-04-25
---------------------------

Highlights
~~~~~~~~~~

* Experimental Hugging Face interface for pre-trained forecasters and foundation models (:pr:`5796`) :user:`benHeid`
* estimator tags are now `systematically documented in the API reference <https://www.sktime.net/en/latest/api_reference/tags.html>`_ (:pr:`6289`) :user:`fkiraly`
* new classifiers, transformers from ``pyts`` interfaced: BOSSVS, learning shapelets, shapelet transform (:pr:`6296`) :user:`johannfaouzi` (author), :user:`fkiraly` (interface)
* new classifiers from ``tslearn`` interfaced: time series SVC, SVR, learning shapelets (:pr:`6273`) :user:`rtavenar` (author), :user:`fkiraly` (interface)
* ``ForecastX`` can now use use future-unknown exogenous variables if passed in ``predict`` (:pr:`6199`) :user:`yarnabrina`
* bagging/bootstrap forecaster can now be applied to multivariate, exogeneous, hierarchical data and produces fully probabilistic forecasts (:pr:`6052`) :user:`fkiraly`
* ``neuralforecast`` models now have settings to auto-detect date-time ``freq``, and pass ``optimizer`` (:pr:`6235`, :pr:`6237`) :user:`pranavvp16`, :user:`geetu040`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``dask`` (data container and parallelization back-end) bounds have been updated to ``<2024.4.2``
* ``arch`` (transformation and parameter estimation soft dependency) bounds have been updated to ``>=5.6,<7.1.0``
* ``holidays`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.48``
* ``mne`` (transformations soft dependency) bounds have been updated to ``>=1.5,<1.8``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

All objects and estimators now can, in addition to the existing PEP 440 package dependency specifier tags,
specify PEP 508 compatible environment markers for their dependencies,
via the ``env_marker`` tag. Values should be PEP 508 compliant strings, e.g., ``platform_system!="Windows"``.

This allows for more fine-grained control over the dependencies of estimators, where needed,
e.g., for estimators that require specific operating systems.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] PEP 508 environment markers for estimators (:pr:`6144`) :user:`fkiraly`
* [ENH] enhancements to tag system, systematic API docs for tags (:pr:`6289`) :user:`fkiraly`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] instance splitter to apply ``sklearn`` splitter to panel data (:pr:`6055`) :user:`fkiraly`
* [ENH] efficient ``_evaluate_by_index`` for MSE and RMSE (``MeanSquaredError``) (:pr:`6248`) :user:`fkiraly`
* [ENH] implement efficient ``_evaluate_by_index`` for ``MedianAbsoluteError`` class (:pr:`6251`) :user:`mobley-trent`

Forecasting
^^^^^^^^^^^

* [ENH] Hugging Face interface for pre-trained forecasters (:pr:`5796`) :user:`benHeid`
* [ENH] bagging/bootstrap forecaster extended to multivariate, exogeneous, hierarchical data (:pr:`6052`) :user:`fkiraly`
* [ENH] Minor ``neuralforecast`` related changes (:pr:`6312`) :user:`yarnabrina`
* [ENH] Option to use future-unknown exogenous variables in ``ForecastX`` if passed in ``predict`` (:pr:`6199`) :user:`yarnabrina`
* [ENH] Add ``optimizer`` param for ``neuralforecast`` models (:pr:`6235`) :user:`pranavvp16`
* [ENH] Update behavior of ``freq="auto"`` in ``neuralforecast`` facing estimators (:pr:`6237`) :user:`geetu040`
* [ENH] ``TBATS`` test parameters to cover doc example (:pr:`6292`) :user:`fkiraly`

Neural networks
^^^^^^^^^^^^^^^

* [ENH] added test parameters to ``CNNNetwork`` and ``ResnetNetwork`` (:pr:`6209`) :user:`julian-fong`
* [ENH] added test parameters for the LSTM FCNN network (:pr:`6281`) :user:`shlok191`

Probability distributions and simulators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] extend ``Empirical`` distribution to hierarchical data (:pr:`6066`) :user:`fkiraly`
* [ENH] mixture distribution, from ``skpro`` (:pr:`6179`) :user:`vandit98`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] added test parameters for ``MatrixProfileClassifier`` (:pr:`6193`) :user:`MMTrooper`
* [ENH] interfaces to further ``tslearn`` estimators (:pr:`6273`) :user:`fkiraly`
* [ENH] interfaces to further ``pyts`` classifiers (:pr:`6296`) :user:`fkiraly`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] clusterer test scenario with unequal length time series; fix clusterer tags (:pr:`6277`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] k-nearest neighbors regressor: support for non-brute algorithms and non-precomputed mode to improve memory efficiency (:pr:`6217`) :user:`Z-Fran`

Transformations
^^^^^^^^^^^^^^^

* [ENH] make ``TabularToSeriesAdaptor`` compatible with ``sklearn`` transformers that accept only ``y``, e.g., ``LabelEncoder`` (:pr:`5982`) :user:`fkiraly`

Test framework
^^^^^^^^^^^^^^

* [ENH] make ``get_examples`` side effect safe via ``deepcopy`` (:pr:`6259`) :user:`fkiraly`
* [ENH] refactor test scenario creation to be lazy rather than on module load (:pr:`6278`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] update installation instructions on ``conda`` soft dependencies (:pr:`6229`) :user:`fkiraly`
* [DOC] add missing import statements to the ``InvertAugmenter`` docstring example (:pr:`6236`) :user:`Anteemony`
* [DOC] Adding Usage Example in docstring (:pr:`6264`) :user:`MihirsinhChauhan`
* [DOC] improve docstring formatting in probabilistic metrics (:pr:`6256`) :user:`fkiraly`
* [DOC] ``authors`` tag - extension template instructions to credit 3rd party interfaced authors (:pr:`5953`) :user:`fkiraly`
* [DOC] Refactor examples directory and link to docs/source/examples (:pr:`6210`) :user:`duydl`
* [DOC] author credits to ``tslearn`` authors (:pr:`6269`) :user:`fkiraly`
* [DOC] author credits to ``pyts`` authors (:pr:`6270`) :user:`fkiraly`
* [DOC] Update README.md - time of Friday meetups (:pr:`6293`) :user:`fkiraly`
* [DOC] systematic API docs for tags (:pr:`6289`) :user:`fkiraly`
* [DOC] in extension templates, clarify handling of soft dependencies (:pr:`6325`) :user:`fkiraly`
* [DOC] author credits to ``pycatch22`` authors, fix missing documentation page (:pr:`6300`) :user:`fkiraly`
* [DOC] added usage examples to multiple estimator docstrings (:pr:`6187`) :user:`MihirsinhChauhan`
* [DOC] Miscellaneous aesthetic improvements to docs UI (:pr:`6211`) :user:`duydl`
* [DOC] Remove redundant code in tutorial section 2.2.4 (:pr:`6267`) :user:`iamSathishR`
* [DOC] Added an example to ``WhiteNoiseAugmenter``  (:pr:`6200`) :user:`SamruddhiNavale`

Maintenance
~~~~~~~~~~~

* [MNT] Basic fix and enhancement of doc local build process (:pr:`6128`) :user:`duydl`
* [MNT] temporary skip for failure #6260 (:pr:`6262`) :user:`fkiraly`
* [MNT] Update dask requirement from ``<2024.2.2`` to ``<2024.4.2``, add new required ``dataframe`` extra to ``pyproject.toml``. (:pr:`6282`) :user:`yarnabrina`
* [MNT] fix isolation of ``mlflow`` soft dependencies (:pr:`6285`) :user:`fkiraly`
* [MNT] add :user:`slavik57` as a maintenance contributor for fixing ``conda-forge`` ``sktime-all-extras 0.28.0`` release (:pr:`6308`) :user:`tm-slavik57`
* [MNT] set GHA macos runner consistently to ``macos-13`` (:pr:`6328`) :user:`fkiraly`
* [MNT] [Dependabot](deps-dev): Update ``holidays`` requirement from ``<0.46,>=0.29`` to ``>=0.29,<0.47`` (:pr:`6250`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``holidays`` requirement from ``<0.47,>=0.29`` to ``>=0.29,<0.48`` (:pr:`6302`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``arch`` requirement from ``<6.4.0,>=5.6`` to ``>=5.6,<7.1.0`` (:pr:`6307`, :pr:`6309`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``pytest-xdist`` requirement from ``<3.6,>=3.3`` to ``>=3.3,<3.7`` (:pr:`6316`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Update ``mne`` requirement from ``<1.7,>=1.5`` to ``>=1.5,<1.8`` (:pr:`6317`) :user:`dependabot[bot]`
* [MNT] Update ``dask`` requirement from ``<2024.2.2`` to ``<2024.4.2``, add new required ``dataframe`` extra to ``pyproject.toml``. (:pr:`6282`) :user:`yarnabrina`

Fixes
~~~~~

Data loaders
^^^^^^^^^^^^

* [BUG] Fix ``tsf`` data error log and make it more precise (:pr:`6258`) :user:`pranavvp16`

Forecasting
^^^^^^^^^^^

* [BUG] Fix ``NaiveForecaster`` with ``sp>1`` (:pr:`5923`) :user:`benHeid`
* [BUG] fix ``FallbackForecaster`` failing with ``ForecastByLevel`` when ``nan_predict_policy='raise'`` (:pr:`6231`) :user:`ninedigits`
* [BUG] Add regression test for bug 3177 (:pr:`6246`) :user:`benHeid`
* [BUG] fix failing test in ``neuralforecast`` auto freq, amid ``pandas`` ``freq`` deprecations (:pr:`6321`) :user:`geetu040`

Probability distributions and simulators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``var`` of ``Laplace`` distribution (:pr:`6324`) :user:`fkiraly`
* [BUG] fix ``Empirical`` index to be ``pd.MultiIndex`` for hierarchical data index (:pr:`6341`) :user:`fkiraly`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix dependent tags of ``TimeSeriesDBSCAN`` (:pr:`6322`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] in ``CNNRegressor``, fix ``self.model not found`` error when ``verbose=True`` (:pr:`6232`) :user:`morestart`

Transformations
^^^^^^^^^^^^^^^

* [BUG] ``Imputer`` bugfix #6224 (:pr:`6253`) :user:`Ram0nB`
* [BUG] Fix backfill of custom function in ``window_feature`` (:pr:`6294`) :user:`toandaominh1997`
* [BUG] fixed indexing of return in ``TSBootstrapAdapter`` (:pr:`6326`) :user:`astrogilda`
* [BUG] Fix ``STLTransformer.inverse_transform`` for univariate case (:pr:`6338`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`Anteemony`,
:user:`astrogilda`,
:user:`benHeid`,
:user:`duydl`,
:user:`fkiraly`,
:user:`geetu040`,
:user:`iamSathishR`,
:user:`julian-fong`,
:user:`MihirsinhChauhan`,
:user:`MMTrooper`,
:user:`mobley-trent`,
:user:`morestart`,
:user:`ninedigits`,
:user:`pranavvp16`,
:user:`Ram0nB`,
:user:`SamruddhiNavale`,
:user:`shlok191`,
:user:`slavik57`,
:user:`tm-slavik57`,
:user:`toandaominh1997`,
:user:`vandit98`,
:user:`yarnabrina`,
:user:`Z-Fran`


Version 0.28.0 - 2024-03-27
---------------------------

Maintenance release:

* scheduled deprecations and change actions
* support for ``pandas 2.2.X``

For last non-maintenance content updates, see 0.27.1.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``sktime`` now supports ``pandas`` ``2.2.X``, bounds have been updated to ``<2.3.0,>=1.1``.
* ``temporian`` (transformations soft dependency) bounds have been updated to ``>=0.7.0,<0.9.0``.
* ``pykalman-bardo`` dependencies have been replaced by the original fork ``pykalman``.
  ``pykalman-bardo`` has been merged back into ``pykalman``,
  which is no longer abandoned.
  This is a soft dependency, and the switch does not affect users installing
  ``sktime`` using one of its dependency sets.


Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* in ``ProphetPiecewiseLinearTrendForecaster``, the seasonality parameters
  ``yearly_seasonality``, ``weekly_seasonality`` and ``daily_seasonality``
  now have default values of ``False``.
  To retain previous behaviour, set these parameters explicitly to ``"auto"``.

Transformations
^^^^^^^^^^^^^^^

* The ``n_jobs`` parameter in the ``Catch22`` transformer is deprecated
  and will be removed in 0.29.0.
  Users should pass parallelization backend parameters via ``set_config`` instead.
  To specify ``n_jobs``, use any of the backends supporting it in the
  ``backend:parallel`` configuration, such as ``"loky"`` or ``"multithreading"``.
  The ``n_jobs`` parameter should be passed via the
  ``backend:parallel:params`` configuration.
  To retain previous behaviour, with a specific setting of ``n_jobs=x``,
  use ``set_config(**{"backend:parallel": "loky", "backend:parallel:params": {"n_jobs": x}})``.
* The ``n_jobs`` parameter in the ``Catch22Wrapper`` transformer has been removed.
  Users should pass parallelization backend parameters via ``set_config`` instead.
  To specify ``n_jobs``, use any of the backends supporting it in the
  ``backend:parallel`` configuration, such as ``"loky"`` or ``"multithreading"``.
  The ``n_jobs`` parameter should be passed via the
  ``backend:parallel:params`` configuration.
  To retain previous behaviour, with a specific setting of ``n_jobs=x``,
  use ``set_config(**{"backend:parallel": "loky", "backend:parallel:params": {"n_jobs": x}})``.
* ``panel.dictionary_based.PAA`` has been renamed to ``PAAlegacy`` in 0.27.0,
  and ``sktime.transformations.series.PAA2`` has been renamed to ``PAA``.
  ``PAA`` is now the primary PAA implementation in ``sktime``.
  After completion of the deprecation cycle, the estimators are no longer available
  under their previous names.
  To migrate dependent code to use the new names, do one of the following:
  1. replace use of ``PAA`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``PAA2`` from ``sktime.transformations.series.paa``, switching
  parameter names appropriately, or
  2. replace use of ``PAA`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``PAAlegacy`` from ``sktime.transformations.panel.dictionary_based``,
  without change of parameter values.
* ``panel.dictionary_based.SAX`` has been renamed to ``SAXlegacy`` in 0.27.0,
  while ``sktime.transformations.series.SAX2`` has been renamed to ``SAX``.
  ``SAX`` is now the primary SAX implementation in ``sktime``,
  while the former ``SAX`` will continue to be available as ``SAXlegacy``.
  After completion of the deprecation cycle, the estimators are no longer available
  under their previous names.
  To migrate dependent code to use the new names, do one of the following:
  1. replace use of ``SAX`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``SAX2`` from ``sktime.transformations.series.paa``, switching
  parameter names appropriately, or
  2. replace use of ``SAX`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``SAXlegacy`` from ``sktime.transformations.panel.dictionary_based``,
  without change of parameter values.

Contents
~~~~~~~~

* [MNT] 0.28.0 deprecations and change actions (:pr:`6198`) :user:`fkiraly`
* [MNT] raise ``pandas`` bound to ``pandas<2.3.0`` (:pr:`5841`) :user:`fkiraly`
* [MNT] update ``temporian`` bound to ``<0.9.0,!=0.8.0`` (:pr:`6222`) :user:`fkiraly`
* [MNT] revert switch from ``pykalman`` to ``pykalman-bardo`` (:pr:`6114`) :user:`fkiraly`
* [MNT] [Dependabot](deps-dev): Update ``pytest-cov`` requirement from ``<4.2,>=4.1`` to ``>=4.1,<5.1`` (:pr:`6215`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump ``tj-actions/changed-files`` from 43 to 44 (:pr:`6226`) :user:`dependabot[bot]`
* [ENH] stricter condition for ``get_test_params`` not failing in repo soft dependency isolation tests (:pr:`6223`) :user:`fkiraly`


Version 0.27.1 - 2024-03-25
---------------------------

Highlights
~~~~~~~~~~

* Phase 1 integration with ``temporian`` - ``TemporianTransformer`` transformer (:pr:`5980`) :user:`ianspektor`, :user:`achoum`, :user:`javiber`
* Phase 1 integration with ``tsbootstrap`` - ``TSBootstrapAdapter`` transformer (:pr:`5887`) :user:`benHeid`, :user:`astrogilda`, :user:`fkiraly`
* Shapelet transform from ``pyts`` available as ``sktime`` transformer (:pr:`6082`) :user:`Abhay-Lejith`
* ``Catch22`` transformer now supports short aliases and parallelization backend selection (:pr:`6002`) :user:`julnow`
* forecasting tuners can now return performances of all parameters, via ``return_n_best_forecasters=-1`` (:pr:`6031`) :user:`HassnHamada`
* ``NeuralForecastRNN`` can now auto-detect ``freq`` (:pr:`6039`) :user:`geetu040`
* time series splitters are now first-class objects, with suite tests and ``check_estimator`` support (:pr:`6051`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``temporian`` is now a soft dependency for ``sktime`` (transformations)
* ``holidays`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.46``
* ``dtw-python`` bounds have been updated to ``>=1.3,<1.5`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

* time series splitters are now full first-class citizens. Interface conformance
  can now be checked with ``check_estimator``.


Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

``cINNForecaster`` will be renamed to CINNForecaster in sktime 0.29.0.
The estimator is available under the future name at its
current location, and will be available under its deprecated name
until 0.30.0. To prepare for the name change,
replace any imports of ``cINNForecaster`` with imports of ``CINNForecaster``.


Enhancements
~~~~~~~~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``check_estimator`` integration for splitters (:pr:`6051`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [ENH] automatic inference of file ending in data loaders for single file types (:pr:`6045`) :user:`SaiRevanth25`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] use ``Index.unique`` instead of ``set`` in conversion from ``pd-multiindex`` to ``df-list`` mtype by :user:`fkiraly` (:pr:`6007`)

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [ENH] Second test parameter set for shapeDTW (:pr:`6093`) :user:`XinyuWuu`
* [ENH] add ``colalign`` functionality to ``ScipyDist`` class as specified in the docstrings (:pr:`6110`) :user:`fnhirwa`

Forecasting
^^^^^^^^^^^

* [ENH] forecasting tuners, ``return_n_best_forecasters=-1`` to return performances of all forecasters (:pr:`6031`) :user:`HassnHamada`
* [ENH] ``NeuralForecastRNN`` ``freq`` auto-detect feature (:pr:`6039`) :user:`geetu040`
* [ENH] ``neuralforecast`` based LSTM model by :user:`pranavvp16` (:pr:`6047`)
* [ENH] fix ``ForecastingHorizon.freq`` handling for ``pandas 2.2.X`` by :user:`fkiraly` (:pr:`6057`)

Neural network templates
^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] added test params to ``RNNNetwork`` (:pr:`6155`) :user:`julian-fong`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] remove private methods from parameters of ``ProximityForest``, ``ProximityTree``, and ``ProximityStump`` by :user:`fnhirwa` (:pr:`6046`)

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] add new test parameter sets for ``TimeSeriesKMeansTslearn`` (:pr:`6195`) :user:`shankariraja`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Migrate DL regressors from ``sktime-dl``: CNTC, InceptionTime, MACNN (:pr:`6038`) :user:`nilesh05apr`
* [ENH] ``MultiplexRegressor`` - autoML multiplexer for time series regressors (:pr:`6075`) :user:`ksharma6`

Transformations
^^^^^^^^^^^^^^^

* [ENH] ``tsbootstrap`` transformer adapter (:pr:`5887`) :user:`benHeid`
* [ENH] ``TemporianTransformer`` - interface to ``temporian`` (:pr:`5980`) :user:`ianspektor`, :user:`achoum`
* [ENH] Refactored and improved ``Catch22`` transformer - support for column names, short aliases, refactor to ``pd.Series``, ``sktime`` native parallelization (:pr:`6002`) :user:`julnow`
* [ENH] Examples for ``YtoX`` transformer (:pr:`6028`, :pr:`6059`) :user:`fkiraly`, :user:`geetu040`
* [ENH] Shapelet transform interfacing ``pyts`` (:pr:`6082`) :user:`Abhay-Lejith`
* [ENH] Add a ``test_mstl`` module checking if ``transform`` returns desired components by :user:`kcentric` (:pr:`6084`)
* [ENH] add test cases for ``HampelFilter`` by :user:`fkiraly` (:pr:`6087`)
* [ENH] Second test parameter set for Kalman Filter (:pr:`6095`) :user:`XinyuWuu`
* [ENH] Add ``MSTL`` import statement in ``detrend`` by :user:`geetu040` (:pr:`6116`)

Test framework
^^^^^^^^^^^^^^

* [ENH] test suite for splitters (:pr:`6051`) :user:`fkiraly`


Documentation
~~~~~~~~~~~~~

* [DOC] Fix invalid use of single-grave in docstrings (:pr:`6023`) :user:`geetu040`
* [DOC] Fix typos in changelog by :user:`yarnabrina` (:pr:`6034`)
* [DOC] corrected Discord channel mention in developer guide (:pr:`6163`) :user:`shankariraja`
* [DOC] add credit to :user:`rikstarmans in ``FallbackForecaster`` (:pr:`6069`) :user:`fkiraly`
* [DOC] Added an example to MLPRegressor #4264  (:pr:`6135`) :user:`vandit98`
* [DOC] in ``BaseSeriesAnnotator``, document the ``int_label`` option (:pr:`6143`) :user:`Alex-JG3`
* [DOC] fix typo in ``_registry.py`` (:pr:`6160`) :user:`pranavvp16`
* [DOC] minor clarifications in mtype descriptions (:pr:`6078`) :user:`fkiraly`
* [DOC] ExponentialSmoothing default method change from L-BFGS-B to SLSQP (:pr:`6186`) :user:`manuel-munoz-aguirre`
* [DOC] fix missing exports in time series regression API ref (:pr:`6191`) :user:`fkiraly`
* [DOC] add more examples to CoC from python software foundation CoC (:pr:`6185`) :user:`fkiraly`
* [DOC] correct deprecation versions in ``BaseDeepClassifier`` docstring (:pr:`6197`) :user:`fkiraly`
* [DOC] update maintainer tag information in docs and PR template (:pr:`6072`) :user:`fkiraly`
* [DOC] Add hall-of-fame widget to README (Added the Hall-of-fame section) #3716  (:pr:`6203`) :user:`KaustubhUp025`
* [DOC] Added docstring example to ``DummyClassifier`` (:pr:`6146`) :user:`YashKhare20`
* [DOC] Added docstring for lstmfcn and MLP classifiers (:pr:`6136`) :user:`vandit98`
* [DOC] Fix syntax error in "getting started" example code block for Time Series Regression (:pr:`6022`) :user:`sahusiddharth`
* [DOC] Added blank lines to properly render ``FourierFeatures`` docstring, ``sp_list`` (:pr:`5984`) :user:`tiloye`
* [DOC] add missing author credits of :user:`ivarzap` (:pr:`6050`) :user:`fkiraly`
* [DOC] fix various typos (:pr:`6043`) :user:`fkiraly`
* [DOC] clarification regarding immutability of ``self``-params in extension templates (:pr:`6053`) :user:`fkiraly`
* [DOC] Fix invalid use of single-grave in docstrings (:pr:`6023`) :user:`geetu040`
* [DOC] Added docstring example to ``CNNRegressor`` (:pr:`6102`) :user:`meraldoantonio`
* [DOC] corrected Discord channel mention in developer guide (:pr:`6163`) :user:`shankariraja`
* [DOC] add credit to :user:`rikstarmans` in ``FallbackForecaster`` (:pr:`6069`) :user:`fkiraly`
* [DOC] Added an example to ``MLPRegressor`` (:pr:`6135`) :user:`vandit98`
* [DOC] fix typo in ``_registry.py`` (:pr:`6160`) :user:`pranavvp16`
* [DOC] minor clarifications in mtype descriptions by :user:`fkiraly` (:pr:`6078`)
* [DOC] ``ExponentialSmoothing`` - fix docstring after default method change from ``L-BFGS-B`` to SLSQP`` (:pr:`6186`) :user:`manuel-munoz-aguirre`
* [DOC] fix missing imports in time series regression API ref (:pr:`6191`) :user:`fkiraly`
* [DOC] add more examples to CoC from python software foundation CoC (:pr:`6185`) :user:`fkiraly`
* [DOC] correct deprecation versions in ``BaseDeepClassifier`` docstring (:pr:`6197`) :user:`fkiraly`
* [DOC] update maintainer tag information in docs and PR template (:pr:`6072`) :user:`fkiraly`
* [DOC] Add hall-of-fame widget and section to README (:pr:`6203`) :user:`KaustubhUp025`


Maintenance
~~~~~~~~~~~

* [MNT] [Dependabot](deps-dev): Update ``holidays`` requirement from ``<0.45,>=0.29`` to ``>=0.29,<0.46`` (:pr:`6164`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``dtw-python`` requirement from ``<1.4,>=1.3`` to ``>=1.3,<1.5`` (:pr:`6165`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump ``tj-actions/changed-files`` from 42 to 43 (:pr:`6125`) :user:`dependabot[bot]`
* [MNT] temporary skip sporadically failing tests for ``ShapeletTransformPyts`` (:pr:`6172`) :user:`fkiraly`
* [MNT] create build tool to check invalid backticks (:pr:`6088`) :user:`geetu040`
* [MNT] decouple ``catch22`` module from ``numba`` utilities (:pr:`6101`) :user:`fkiraly`
* [MNT] bound ``temporian<0.8.0`` (:pr:`6184`) :user:`fkiraly`
* [MNT] Ensure Update Contributors does not run on main (:pr:`6189`) :user:`Greyisheep`, :user:`duydl`
* [MNT] initialize change cycle (0.28.0) for renaming ``cINNForecaster`` to ``CINNForecaster`` (:pr:`6121`) :user:`geetu040`
* [MNT] Fix failing tests due to ``tensorflow`` update (:pr:`6098`) :user:`benHeid`
* [MNT] silence sporadic failure in ``test_evaluate_error_score`` (:pr:`6058`) :user:`fkiraly`
* [MNT] update ``statsforecast`` version in ``forecasting`` extra  (:pr:`6064`) :user:`yarnabrina`
* [MNT] Docker files updated by (:pr:`6076`) :user:`deysanjeeb`
* [MNT] deprecation action timing for ``Catch22`` changes  (:pr:`6123`) :user:`fkiraly`
* [MNT] run ``update-contributors`` workflow only on PR by (:pr:`6133`) :user:`fkiraly`
* [MNT] temporary skip sporadically failing tests for ``ShapeletTransformPyts`` (:pr:`6172`) :user:`fkiraly`
* [MNT] enable concurrency settings in 'Install and Test' GHA workflow (:pr:`6074`) :user:`MEMEO-PRO`
* [MNT] temporary skip for some sporadic failures on ``main`` (:pr:`6208`) :user:`fkiraly`


Fixes
~~~~~

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [BUG] Fix various issues in shapeDTW (:pr:`6093`) :user:`XinyuWuu`
* [BUG] resolve redundant or problematic statements in ``numba`` bounding matrix routines (:pr:`6183`) :user:`albertoazzari`

Estimator registry
^^^^^^^^^^^^^^^^^^

* [BUG] remove unnecessary line in ``all_estimators`` (:pr:`6103`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] Fixed ``SARIMAX`` failure when ``X`` is passed to predict but not ``fit``  (:pr:`6005`) :user:`Abhay-Lejith`
* [BUG] fix ``BaseForecaster.predict_var`` default if ``predict_proba`` is implemented (:pr:`6067`) :user:`fkiraly`
* [BUG] In ``ForecastingHorizon``, ignore ``ValueError`` on ``pd.infer_freq`` when index has fewer than 3 values (:pr:`6097`) :user:`tpvasconcelos`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``super`` calls in deep learning classifiers and regressors (:pr:`6139`) :user:`fkiraly`
* [BUG] Resolved wrong arg name ``lr`` in ``SimpleRNNClassifier`` and regressor, fix minor ``batch_size`` param issue in ``ResNetClassifier`` (:pr:`6154`) :user:`vandit98`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``BaseRegressor.score`` method failing with ``sklearn.metrics r2_score got an unexpected keyword argument 'normalize`` (:pr:`6019`) :user:`Cyril-Meyer`
* [BUG] fix ``super`` calls in deep learning classifiers and regressors (:pr:`6139`) :user:`fkiraly`
* [BUG] fix network construction in ``InceptionTimeRegressor`` (:pr:`6140`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix ``FeatureUnion`` for primitive outputs (:pr:`6079`) :user:`fkiraly`, :user:`fspinna`
* [BUG] Fix unexpected NaN values in ``Summarizer`` (:pr:`6081`) :user:`ShreeshaM07`
* [BUG] Update ``_shapelet_transform_numba.py`` to improve numerical stability (:pr:`6141`) :user:`stevcabello`

Test framework
^^^^^^^^^^^^^^

* [BUG] fix ``deep_equals`` when comparing ``ForecastingHorizon`` of different lengths by :user:`MBristle` (:pr:`5954`)

Webpage
^^^^^^^

* [BUG] fix search function for estimator overview not working (:pr:`6105`) :user:`duydl`

Contributors
~~~~~~~~~~~~

:user:`Abhay-Lejith`,
:user:`achoum`,
:user:`albertoazzari`,
:user:`Alex-JG3`,
:user:`astrogilda`,
:user:`benHeid`,
:user:`Cyril-Meyer`,
:user:`deysanjeeb`,
:user:`duydl`,
:user:`fkiraly`,
:user:`fnhirwa`,
:user:`fspinna`,
:user:`geetu040`,
:user:`Greyisheep`,
:user:`HassnHamada`,
:user:`ianspektor`,
:user:`javiber`,
:user:`julian-fong`,
:user:`julnow`,
:user:`KaustubhUp025`,
:user:`kcentric`,
:user:`ksharma6`,
:user:`manuel-munoz-aguirre`,
:user:`MBristle`,
:user:`MEMEO-PRO`,
:user:`meraldoantonio`,
:user:`nilesh05apr`,
:user:`pranavvp16`,
:user:`SaiRevanth25`,
:user:`sahusiddharth`,
:user:`shankariraja`,
:user:`stevcabello`,
:user:`tiloye`,
:user:`tpvasconcelos`,
:user:`vandit98`,
:user:`XinyuWuu`,
:user:`YashKhare20`


Version 0.27.0 - 2024-02-28
---------------------------

Maintenance release:

* scheduled deprecations and change actions
* support for soft dependency ``numba 0.59`` and ``numba`` under ``python 3.12``
* minor documentation updates, website updates for GSoC 2024

For last non-maintenance content updates, see 0.26.1.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``numba`` bounds have been updated to ``<0.60``.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting tuners
^^^^^^^^^^^^^^^^^^

* in forecasting tuners ``ForecastingGridSearchCV``, ``ForecastingRandomizedSearchCV``,
  ``ForecastingSkoptSearchCV``, the ``joblib`` backend specific parameters ``n_jobs``,
  ``pre_dispatch`` have been removed.
  Users should pass backend parameters via the ``backend_params`` parameter instead.
  Direct replacements are ``backend='joblib'``,
  and ``n_jobs`` and ``pre_dispatch`` passed via ``backend_params``.

Transformations
^^^^^^^^^^^^^^^

* in ``SplitterSummarizer``, the ``remember_data`` argument has been removed.
  Users should use the ``fit_on`` and ``transform_on`` arguments instead.
  Logic identical argument replacements are:
  ``remember_data=True`` with ``fit_on='all_train'`` and
  ``transform_on='all_train'``; and ``remember_data=False`` with
  ``"fit_on='transform_train'`` and ``transform_on='transform_train'``.
* ``panel.dictionary_based.PAA`` has been renamed to ``PAAlegacy`` in 0.27.0,
  and ``sktime.transformations.series.PAA2`` has been renamed to ``PAA``.
  ``PAA`` is now the primary PAA implementation in ``sktime``,
  while the former ``PAA`` will continue to be available as ``PAAlegacy``.
  Both estimators are also available under their former name
  until 0.28.0.
  To prepare for the name change, do one of the following:
  1. replace use of ``PAA`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``PAA2`` from ``sktime.transformations.series.paa``, switching
  parameter names appropriately, or
  2. replace use of ``PAA`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``PAAlegacy`` from ``sktime.transformations.panel.dictionary_based``,
  without change of parameter values.
* ``panel.dictionary_based.SAX`` has been renamed to ``SAXlegacy`` in 0.27.0,
  while ``sktime.transformations.series.SAX2`` has been renamed to ``SAX``.
  ``SAX`` is now the primary SAX implementation in ``sktime``,
  while the former ``SAX`` will continue to be available as ``SAXlegacy``.
  Both estimators are also available under their former name
  until 0.28.0.
  To prepare for the name change, do one of the following:
  1. replace use of ``SAX`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``SAX2`` from ``sktime.transformations.series.paa``, switching
  parameter names appropriately, or
  2. replace use of ``SAX`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``SAXlegacy`` from ``sktime.transformations.panel.dictionary_based``,
  without change of parameter values.


Contents
~~~~~~~~

Documentation
~~~~~~~~~~~~~

* [DOC] improved formatting of ``HierarchyEnsembleForecaster`` docstring (:pr:`6008`) :user:`fkiraly`
* [DOC] add missing ``PluginParamsTransformer`` to API reference (:pr:`6010`) :user:`fkiraly`
* [DOC] update contact links in code of conduct (:pr:`6011`) :user:`fkiraly`
* [DOC] 2024 summer programme links on ``sktime.net`` landing page (:pr:`6013`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] 0.27.0 deprecations and change actions (:pr:`5974`) :user:`fkiraly`
* [MNT] [Dependabot](deps-dev): Update ``numba`` requirement from ``<0.59`` to ``<0.60`` (:pr:`5877`) :user:`dependabot[bot]`


Version 0.26.1 - 2024-02-26
---------------------------

Highlights
~~~~~~~~~~

* Conditional Invertible Neural Network forecaster - from 2022 BigDEAL challenge (:pr:`5339`) :user:`benHeid`
* ``neuralforecast`` adapter and rnn forecaster (:pr:`5962`) :user:`yarnabrina`
* ``FallbackForecaster`` now supports probabilistic forecasters and setting of nan handling policy (:pr:`5847`, :pr:`5924`) :user:`ninedigits`
* ``statsforecast`` ``AutoTBATS`` interface (:pr:`5908`) :user:`yarnabrina`
* k-nearest neighbor classifiers from ``pyts`` and ``tslearn`` (:pr:`5939`, :pr:`5952`) :user:`fkiraly`
* ``pyts`` ``ROCKET`` transformation (:pr:`5851`) :user:`fkiraly`
* deep learning regressors from ``sktime-dl`` migrated: FCN, LSTMFCN, MLP (:pr:`6001`) :user:`nilesh05apr`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``dask`` (data container and parallelization back-end) bounds have been updated to ``<2024.2.2``
* ``holidays`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.44``
* ``pyts`` is now a soft dependency for classification and transformations

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Transformations
^^^^^^^^^^^^^^^

All transformation dunders now automatically coerce ``sklearn`` transformers to
``sktime`` transformers, wrapping in ``TabularToSeriesAdaptor`` is no longer necessary
when using ``sklearn`` transformers in ``sktime`` pipelines specified by dunders.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Transformations
^^^^^^^^^^^^^^^

The ``n_jobs`` parameter of ``Catch22Wrapper`` has been deprecated
and will be removed in ``sktime`` 0.28.0.
Users should pass parallelization backend parameters via ``set_config`` instead.

Enhancements
~~~~~~~~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] efficient ``_evaluate_by_index`` for ``MeanAbsolutePercentageError`` (:pr:`5842`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [ENH] add encoding parameter in data loaders (:pr:`6000`) :user:`Cyril-Meyer`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``polars`` based ``Table`` mtype, feature name metadata (:pr:`5757`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] Conditional Invertible Neural Network forecaster (:pr:`5339`) :user:`benHeid`
* [ENH] Expose seasonality parameters of ``ProphetPiecewiseLinearTrendForecaster`` (:pr:`5834`) :user:`sbuse`
* [ENH] centralize logic for safe clone of delegator tags - forecasting (:pr:`5845`) :user:`fkiraly`
* [ENH] ``FallbackForecaster`` - support for probabilistic forecasters (:pr:`5847`) :user:`ninedigits`
* [ENH] interface to ``ARIMA`` from ``statsmodels`` library (:pr:`5857`) :user:`arnaujc91`
* [ENH] Improved specificity of some error messages in forecasters and transformations (:pr:`5882`) :user:`fkiraly`
* [ENH] ``statsforecast`` ``AutoTBATS`` direct interface estimator (:pr:`5908`) :user:`yarnabrina`
* [ENH] Several updates in direct ``statsforecast`` interface estimators (:pr:`5920`) :user:`yarnabrina`
* [ENH] Add nan policy handler for ``FallbackForecaster`` (:pr:`5924`) :user:`ninedigits`
* [ENH] ``ForecastX`` option to use future-known variables as exogenous variables in forecasting future-unknown exogenous variables (:pr:`5926`) :user:`fkiraly`
* [ENH] ``neuralforecast`` adapter and rnn forecaster (:pr:`5962`) :user:`yarnabrina`
* [ENH] rearchitect ``ForecastingSkoptSearchCV`` on abstract parallelization backend (:pr:`5973`) :user:`fkiraly`
* [ENH] in ``ForecastingPipeline``, allow ``None`` ``X`` to be passed to transformers (:pr:`5977`) :user:`albahhar`, :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] second set of test parameters for ``GMMHMM`` (:pr:`5931`) :user:`sanjayk0508`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] k-nearest neighbors classifier: support for non-brute algorithms and non-precomputed mode to improve memory efficiency (:pr:`5937`) :user:`fkiraly`
* [ENH] adapter to ``pyts`` ``KNeighborsClassifier`` (:pr:`5939`) :user:`fkiraly`
* [ENH] adapter to ``tslearn`` ``KNeighborsTimeSeriesClassifier`` (:pr:`5952`) :user:`fkiraly`
* [ENH] Feature importance capability tag for classifiers (:pr:`5969`) :user:`sanjayk0508`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Migrate DL regressors from ``sktime-dl``: FCN, LSTMFCN, MLP (:pr:`6001`) :user:`nilesh05apr`

Transformations
^^^^^^^^^^^^^^^

* [ENH] ``pyts`` adapter and interface to ``pyts`` ``ROCKET`` (:pr:`5851`) :user:`fkiraly`
* [ENH] in transformer dunders, uniformize coercion of ``sklearn`` transformers (:pr:`5869`) :user:`fkiraly`
* [ENH] Improved specificity of some error messages in forecasters and transformations (:pr:`5882`) :user:`fkiraly`
* [ENH] second set of test parameters for ``TSInterpolator`` (:pr:`5910`) :user:`sanjayk0508`
* [ENH] improved output type checking error messages in ``BaseTransformer.transform`` (:pr:`5921`) :user:`fkiraly`
* [ENH] refactor ``Catch22Wrapper`` transformer to use ``pd.Series`` type internally (:pr:`5983`) :user:`fkiraly`

Test framework
^^^^^^^^^^^^^^

* [ENH] testing estimators whose package dependencies are changed in ``pyproject.toml`` (:pr:`5727`) :user:`fkiraly`


Fixes
~~~~~

Forecasting
^^^^^^^^^^^

* [BUG] Remove duplicative setting of ``_fh`` and ``_y`` in _fit of ``_pytorch.py`` (:pr:`5889`) :user:`benHeid`
* [BUG] ``BaseForecaster`` - move ``check_fh`` to inner loop if vectorized (:pr:`5900`) :user:`ciaran-g`
* [BUG] fix sporadic failure of ``ConformalIntervals`` if ``sample_frac`` is too low (:pr:`59/2`) :user:`fkiraly`

Pipelines
^^^^^^^^^

* [BUG] In ``Pipeline``, empty dummy step's buffer only if new data arrive (:pr:`5837`) :user:`benHeid`

Probability distributions and simulators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix missing loc/scale in ``TDistribution`` methods (:pr:`5942`) :user:`ivarzap`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] corrected default loss function to ``CNNClassifier`` (:pr:`5852`) :user:`Vasudeva-bit`
* [BUG] fix ``BaseClassifier.fit_predict`` for multioutput ``y`` and non-none ``cv`` (:pr:`5928`) :user:`fkiraly`
* [BUG] fix input check error message in ``BaseTransformer`` (:pr:`5947`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix input check error message in `BaseTransformer` (:pr:`5947`) :user:`fkiraly`
* [BUG] Fixed transform method of MSTL transformer (:pr:`5996`) :user:`Abhay-Lejith`


Maintenance
~~~~~~~~~~~

* [MNT] improvements to modular CI framework - part 2, merge frameworks (:pr:`5785`) :user:`fkiraly`
* [MNT] ``pandas 2.2.X`` compatibility fixes (:pr:`5840`) :user:`fkiraly`
* [MNT] fix moto breaking change by using different mocking methods depending on version (:pr:`5858`) :user:`yarnabrina`
* [MNT] address some ``pandas`` deprecations (:pr:`5883`) :user:`fkiraly`
* [MNT] addressed ``FutureWarning`` for RMSE by using newer ``root_mean_absolute_error`` function (:pr:`5884`) :user:`yarnabrina`
* [MNT] Skip ``mlflow`` tests when soft-dependencies are absent (:pr:`5888`) :user:`achieveordie`
* [MNT] fix failing CRON "test all" workflow (:pr:`5925`) :user:`fkiraly`
* [MNT] update versions of several actions (:pr:`5929`) :user:`yarnabrina`
* [MNT] Add codecov token to coverage uploads (:pr:`5930`) :user:`yarnabrina`
* [MNT] CI on main fix: add checkout step to detect steps in CI (:pr:`5945`) :user:`fkiraly`
* [MNT] address some upcoming deprecations (:pr:`5971`) :user:`fkiraly`
* [MNT] avoid running unit tests in CI for documentation/template/etc changes (:pr:`5976`) :user:`yarnabrina`
* [MNT] [Dependabot](deps-dev): Update dask requirement from ``<2024.1.1`` to ``<2024.1.2`` (:pr:`5861`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update dask requirement from ``<2024.1.2`` to ``<2024.2.1`` (:pr:`5958`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update holidays requirement from ``<0.43,>=0.29`` to ``>=0.29,<0.44`` (:pr:`5965`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update dask requirement from ``<2024.2.1`` to ``<2024.2.2`` (:pr:`5991`) :user:`dependabot[bot]`


Documentation
~~~~~~~~~~~~~

* [DOC] recipes for simple parameter change and deprecation management (:pr:`5875`) :user:`fkiraly`
* [DOC] improved deprecation recipes (:pr:`5890`) :user:`fkiraly`
* [DOC] fixing broken link to ``DropNA`` in API reference (:pr:`5899`) :user:`sbuse`
* [DOC] remove obsolete request for contribution (python 3.10 compatibility) in install docs (:pr:`5901`) :user:`fkiraly`
* [DOC] use tags for estimator overview (:pr:`5906`) :user:`fkiraly`
* [DOC] add "maintainers" column to estimator overview (:pr:`5911`) :user:`fkiraly`
* [DOC] fix ``ARDL`` API reference (:pr:`5912`) :user:`fkiraly`
* [DOC] add GitHub ID hyperlinks in estimator table (:pr:`5916`) :user:`fkiraly`
* [DOC] fix odd formatting in deprecation timeline example (:pr:`5957`) :user:`fkiraly`
* [DOC] improved glossary and forecasting docstrings (:pr:`5968`) :user:`fkiraly`
* [DOC] remove type annotations in extension templates (:pr:`5970`) :user:`fkiraly`
* [DOC] Typo in documentation, fixed monotonous to monotonic  (:pr:`5946`) :user:`eduardojp26`
* [DOC] fixed typo in docstring (:pr:`5949`) :user:`yarnabrina`
* [DOC] fixed typo in docstring (:pr:`5950`) :user:`yarnabrina`
* [DOC] format docstrings in ``feature_selection.py`` correctly (:pr:`5994`) :user:`oleeviyababu`


Contributors
~~~~~~~~~~~~

:user:`Abhay-Lejith`,
:user:`achieveordie`,
:user:`albahhar`,
:user:`arnaujc91`,
:user:`benHeid`,
:user:`ciaran-g`,
:user:`Cyril-Meyer`,
:user:`eduardojp26`,
:user:`fkiraly`,
:user:`ivarzap`,
:user:`ninedigits`,
:user:`oleeviyababu`,
:user:`sanjayk0508`,
:user:`sbuse`,
:user:`Vasudeva-bit`,
:user:`yarnabrina`


Version 0.26.0 - 2024-01-27
---------------------------

Maintenance release:

* support for ``scikit-learn 1.4.X``
* scheduled deprecations
* minor bugfix

For last non-maintenance content updates, see 0.25.1.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``scikit-learn`` bounds have been updated to ``>=0.24.0,<1.5.0``.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* in forecasting ``evaluate``, ``kwargs`` have been removed.
  Users should pass backend parameters via the ``backend_params``
  parameter instead.

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* in ``check_is_mtype``, the default of ``msg_return_dict`` has now changed to ``"dict"``

Forecasting tuners
^^^^^^^^^^^^^^^^^^

* in forecasting tuners ``ForecastingGridSearchCV``, ``ForecastingRandomizedSearchCV``,
  ``ForecastingSkoptSearchCV``, use of ``joblib`` backend specific parameters ``n_jobs``,
  ``pre_dispatch`` has been deprecated, and will be removed in ``sktime`` 0.27.0.
  Users should pass backend parameters via the ``backend_params`` parameter instead.

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* In ``SimpleRNNClassifier``, the ``num_epochs`` parameter has been
  renamed to ``n_epochs``. The original parameter of name ``num_epochs`` has now
  been removed.

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* In ``SimpleRNNRegressor``, the ``num_epochs`` parameter has been
  renamed to ``n_epochs``. The original parameter of name ``num_epochs`` has now
  been removed.

Contents
~~~~~~~~

* [MNT] 0.26.0 deprecations and change actions (:pr:`5817`) :user:`fkiraly`
* [MNT] [Dependabot](deps-dev): Update ``scikit-learn`` requirement from
  ``<1.4.0,>=0.24`` to ``>=0.24,<1.5.0`` (:pr:`5776`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump styfle/cancel-workflow-action from ``0.12.0``
  to ``0.12.1`` (:pr:`5839`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump dorny/paths-filter
  from ``2`` to ``3`` (:pr:`5838`) :user:`dependabot[bot]`
* [BUG] fix tag handling in ``IgnoreX`` (:pr:`5843`) :user:`tpvasconcelos`, :user:`fkiraly`


Version 0.25.1 - 2024-01-24
---------------------------

Highlights
~~~~~~~~~~

* in ``make_reduction``, direct reduction forecaster now supports probabilistic tabular regressors from ``skpro`` (:pr:`5536`) :user:`fkiraly`
* new, efficient, parallelizable PAA and SAX transformer implementations, available as ``PAA2``, ``SAX2`` (:pr:`5742`) :user:`steenrotsman`
* ``FallbackForecaster``, fallback chain of multiple forecaster for exception handling (:pr:`5779`) :user:`ninedigits`
* time series classification: ``sktime`` native grid search, multiplexer for autoML (:pr:`4596`, :pr:`5678`) :user:`achieveordie`, :user:`fkiraly`
* ``IgnoreX`` - forecasting compositor to ignore exogenous data, for use in tuning (:pr:`5769`) :user:`hliebert`, :user:`fkiraly`
* classifier migrated from ``sktime-dl``: CNTC classifier (:pr:`3978`) :user:`aurumnpegasus`
* authors and maintainers of algorithms are now tracked via tags ``"authors"`` and ``"maintainers"``, see below

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``arch`` (forecasting and parameter estimation soft dependency) bounds have been updated to ``>=5.6,<6.4.0`` (:pr:`5771`) :user:`dependabot[bot]`
* ``mne`` (transformations soft dependency) bounds have been updated to  ``>=1.5,<1.7`` (:pr:`5585`) :user:`dependabot[bot]`
* ``dask`` (data container and parallelization back-end) bounds have been updated to ``<2024.1.1`` (:pr:`5748`) :user:`dependabot[bot]`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* estimators and objects now record author and maintainer information in the new
  tags ``"authors"`` and ``"maintainers"``. This is required only for estimators
  in ``sktime`` proper and compatible third party packages. It is also used to generate
  mini-package headers used in lookup functionality of the ``sktime`` webpage.
* author and maintainer information in the ``sktime`` package is no longer recorded in
  ``CODEOWNERS``, but in the new tags ``"authors"`` and ``"maintainers"``.
  Authors and maintainer do not need to action this change, as it has been carried out
  by the ``sktime`` maintainers. However, authors and maintainers are encouraged to
  check the information in the tags, and to flag any accidental omissions or errors.

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* forecasting point prediction metrics now also support parallelization via
  ``set_config``, for broadcasting on hierarchical or multivariate data

Forecasting
^^^^^^^^^^^

* forecasters can now be prevented from storing a reference to all seen data
  as ``self._y`` and ``self._X`` by setting the config ``"remember_data"`` to
  ``False`` via ``set_config``. This is useful for serialization of forecasters.
  Currently, the setting is only supported for a combination of data and forecasters
  where instance or variable broadcasting is not triggered,
  but the feature will be extended to all situations in the future.

Parameter estimation and hypothesis testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Parameter plugin or estimation based parameter tuning estimators can now be quickly constructed
  with the ``*`` dunder, which will construct a ``PluginParamsForecaster`` or ``PluginParamsTransformer``
  with all fitted parameters (``get_fitted_params``) of the left element plugged in into the right element
  (``set_params``), where parameter names match.
  For instance, ``SeasonalityACF() * Deseasonalizer()`` will construct
  a ``Deseasonalizer`` whose ``sp`` (seasonality period) parameter is tuned
  by ``SeasonalityACF``,  estimating ``sp`` via the ACF significance criterion on the series.
* The ``*`` dunder binds to the left, for instance
  ``Differencer() * SeasonalityACF() * Deseasonalizer()`` will construct
  a ``Deseasonalizer`` whose ``sp`` (seasonality period) parameter is tuned
  by ``SeasonalityACF``, estimating ``sp`` via the ACF significance criterion
  on first differenced data (for stationarity).
  Here first differencing is not applied to the ``Deseasonalizer``,
  but only to the input of ``SeasonalityACF``, as the first ``*`` constructs
  a parameter estimator, and the second ``*`` plugs in the parameter estimator into
  the ``Deseasonalizer``.

Transformations
^^^^^^^^^^^^^^^

* transformations, i.e., ``BaseTransformer`` descendant instances,
  can now also return ``None`` in ``_transform``, this is interpreted as empty data.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Transformations
^^^^^^^^^^^^^^^

* ``panel.dictionary_based.PAA`` will be renamed to ``PAAlegacy`` in ``sktime`` 0.27.0,
  while ``sktime.transformations.series.PAA2`` will be renamed to ``PAA``.
  ``PAA2`` will become the primary PAA implementation in ``sktime``,
  while the current ``PAA`` will continue to be available as ``PAAlegacy``.
  Both estimators are also available under their future name at their
  current location, and will be available under their deprecated name
  until 0.28.0.
  To prepare for the name change, do one of the following:
  1. replace use of ``PAA`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``PAA2`` from ``sktime.transformations.series.paa``, switching
  parameter names appropriately, or
  2. replace use of ``PAA`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``PAAlegacy`` from ``sktime.transformations.panel.dictionary_based``,
  without change of parameter values.
* ``panel.dictionary_based.SAX`` will be renamed to ``SAXlegacy`` in ``sktime`` 0.27.0,
  while ``sktime.transformations.series.SAX2`` will be renamed to ``SAX``.
  ``SAX2`` will become the primary SAX implementation in ``sktime``,
  while the current ``SAX`` will continue to be available as ``SAXlegacy``.
  Both estimators are also available under their future name at their
  current location, and will be available under their deprecated name
  until 0.28.0.
  To prepare for the name change, do one of the following:
  1. replace use of ``SAX`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``SAX2`` from ``sktime.transformations.series.paa``, switching
  parameter names appropriately, or
  2. replace use of ``SAX`` from ``sktime.transformations.panel.dictionary_based``
  by use of ``SAXlegacy`` from ``sktime.transformations.panel.dictionary_based``,
  without change of parameter values.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] update ``deep_equals`` to accommodate plugins, e.g., for ``polars`` (:pr:`5504`) :user:`fkiraly`
* [ENH] Replace ``isinstance`` by ``object_type`` tag based checks (:pr:`5657`) :user:`benheid`
* [ENH] author and maintainer tags (:pr:`5754`) :user:`fkiraly`
* [ENH] enable ``all_tags`` to retrieve estimator and object tags (:pr:`5798`) :user:`fkiraly`
* [ENH] remove maintainer information from ``CODEOWNERS`` in favour of estimator tags (:pr:`5808`) :user:`fkiraly`
* [ENH] author and maintainer tags for alignment and distances modules (:pr:`5801`) :user:`fkiraly`
* [ENH] author and maintainer tags for forecasting module (:pr:`5802`) :user:`fkiraly`
* [ENH] author and maintainer tags for distributions and parameter fitting module (:pr:`5803`) :user:`fkiraly`
* [ENH] author and maintainer tags for classification, clustering and regression modules (:pr:`5807`) :user:`fkiraly`
* [ENH] author and maintainer tags for transformer module (:pr:`5800`) :user:`fkiraly`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Repeat splitter composition (:pr:`5737`) :user:`fkiraly`
* [ENH] parallelization support and config for forecasting performance metrics (:pr:`5813`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] in ``VectorizedDF``, partially decouple internal data store from methods (:pr:`5681`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] ``Imputer``: conditional parameter handling logic (:pr:`3916`) :user:`aiwalter`, :user:`fkiraly``
* [ENH] support for probabilistic regressors (``skpro``) in ``make_reduction``, direct reduction (:pr:`5536`) :user:`fkiraly`
* [ENH] private utility for ``BaseForecaster`` get columns, for all ``predict``-like functions (:pr:`5590`) :user:`fkiraly`
* [ENH] adding second test parameters for ``TBATS`` (:pr:`5689`) :user:`NguyenChienFelix33`
* [ENH] config to turn off data memory in forecasters (:pr:`5676`) :user:`fkiraly`, :user:`corradomio`
* [ENH] Simplify conditional statements in direct reducer (:pr:`5725`) :user:`fkiraly`
* [ENH] forecasting compositor to ignore exogenous data (:pr:`5769`) :user:`hliebert`, :user:`fkiraly`
* [ENH] add ``disp`` parameter to ``SARIMAX`` to control output verbosity (:pr:`5770`) :user:`tvdboom`
* [ENH] expose parameters supported by ``fit`` method of ``SARIMAX`` in ``statsmodels`` (:pr:`5787`) :user:`yarnabrina`
* [ENH] ``FallbackForecaster``, fallback upon fail with multiple forecaster chain (:pr:`5779`) :user:`ninedigits`

Parameter estimation and hypothesis testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Simplify ``BaseEstimator._get_fitted_params()`` and ``BaseParamFitter`` inheritance of that method (:pr:`5633`) :user:`tpvasconcelos`
* [ENH] parameter plugin for estimator into transformers, right concat dunder (:pr:`5764`) :user:`fkiraly`

Probability distributions and simulators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] bring distributions module on par with ``skpro`` distributions (:pr:`5708`) :user:`fkiraly`, :user:`alex-jg3`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] migrating CNTC network and classifier for classification from ``sktime-dl`` (:pr:`3978`) :user:`aurumnpegasus`, :user:`fkiraly`
* [ENH] grid search for time series classification (:pr:`4596`) :user:`achieveordie`, :user:`fkiraly`
* [ENH] reduce private coupling of ``IndividualBOSS`` classifier and ``BaseClassifier`` (:pr:`5654`) :user:`fkiraly`
* [ENH] multiplexer classifier (:pr:`5678`) :user:`fkiraly`
* [ENH] refactor structure of time series forest classifier related files (:pr:`5751`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] better explanation about fit/transform instance linking in instance-wise transformers in error messages, and pointer to common solution (:pr:`5652`) :user:`fkiraly`
* [ENH] New ``PAA`` and ``SAX`` transformer implementations (:pr:`5742`) :user:`steenrotsman`
* [ENH] feature upgrade for ``SplitterSummarizer`` - granular control of inner ``fit``/``transform`` input (:pr:`5750`) :user:`fkiraly`
* [ENH] allow ``BaseTransformer._transform`` to return ``None`` (:pr:`5772`) :user:`fkiraly`, :user:`hliebert`

Test framework
^^^^^^^^^^^^^^

* [ENH] refactor tests with parallelization backend fixtures to programmatic backend fixture lookup (:pr:`5714`) :user:`fkiraly`
* [ENH] further refactor parallelization backend test fixtures to use central location (:pr:`5734`) :user:`fkiraly`


Fixes
~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix scitype inference utility for all cases (:pr:`5672`) :user:`fkiraly`
* [BUG] fixes for minor typos in error message related to custom ``joblib`` backend selection (:pr:`5724`) :user:`fkiraly`
* [BUG] handles ``AttributeError`` in ``show_versions`` when dependency lacks ``__version__`` (:pr:`5793`) :user:`yarnabrina`
* [BUG] fix type error in parallelization backend test fixture refactor (:pr:`5760`) :user:`fkiraly`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix dynamic ``make_forecasting_scorer`` for newer ``sklearn`` metrics (:pr:`5717`) :user:`fkiraly`
* [BUG] fix ``test_evaluate_error_score`` to skip test of expected warning raised if the ``joblib`` backend is ``"loky"`` or ``"multiprocessing"`` (:pr:`5780`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [BUG] fix ``extract_path`` arg in ``sktime.datasets.load_UCR_UEA_dataset`` (:pr:`5744`) :user:`steenrotsman`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``deep_equals`` for ``np.array`` with ``dtype="object"`` (:pr:`5697`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] fix ``ForecastingHorizon.get_expected_pred_idx`` ``sort_time`` (:pr:`5726`) :user:`fkiraly`
* [BUG] in ``BaggingForecaster``, fix ``random_state`` handling (:pr:`5730`) :user:`fkiraly`

Pipelines
^^^^^^^^^

* [BUG] Enable ``pipeline.fit`` without X (:pr:`5656`) :user:`benheid`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``predict`` output conversion failure in ``BaseClassifier``, ``BaseRegressor``, if ``y_inner_mtype`` tag is a list (:pr:`5680`) :user:`fkiraly`
* [BUG] fix ``test_multioutput`` for genuinely multioutput classifiers (:pr:`5700`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``predict`` output conversion failure in ``BaseClassifier``, ``BaseRegressor``, if ``y_inner_mtype`` tag is a list (:pr:`5680`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] skip sporadic test errors in ``ExponentialSmoothing`` (:pr:`5516`) :user:`achieveordie`
* [BUG] fix sporadic permutation of internal feature columns in ``TSFreshClassifier.predict`` (:pr:`5673`) :user:`fkiraly`
* [BUG] fix backend strings in transformer ``test_base`` (:pr:`5695`) :user:`fkiraly`
* [BUG] Ensure ``MultiRocketMultivariate`` uses ``random_state`` (:pr:`5710`) :user:`chrico-bu-uab`

Test framework
^^^^^^^^^^^^^^

* [BUG] Fixing dockerized tests (:pr:`5426`) :user:`kurayami07734`


Maintenance
~~~~~~~~~~~

* [MNT] [Dependabot](deps-dev): Update sphinx-issues requirement from ``<4.0.0`` to ``<5.0.0`` (:pr:`5792`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump tj-actions/changed-files from 41 to 42 (:pr:`5777`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update arch requirement from ``<6.3.0,>=5.6`` to ``>=5.6,<6.4.0`` (:pr:`5771`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update mne requirement from ``<1.6,>=1.5`` to ``>=1.5,<1.7`` (:pr:`5585`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update dask requirement from ``<2023.12.2`` to ``<2024.1.1`` (:pr:`5748`) :user:`dependabot[bot]`
* [MNT] improvements to modular CI framework - clearer naming, ``pyproject`` handling (:pr:`5713`) :user:`fkiraly`
* [MNT] temporary deactivation of new CI (:pr:`5795`) :user:`fkiraly`
* [MNT] fix faulty deprecation logic for ``n_jobs``, ``pre_dispatch`` in forecasting tuners, bump deprecation to 0.27.0 (:pr:`5784`) :user:`fkiraly`
* [MNT] update python version in binder dockerfile to 3.11 (:pr:`5762`) :user:`fkiraly`
* [MNT] address various deprecations from ``pandas`` (:pr:`5733`) :user:`fkiraly`, :user:`yarnabrina`
* [MNT] ``scikit-learn 1.4.0`` compatibility patches (:pr:`5782`, :pr:`5811`) :user:`fkiraly`
* [MNT] Code quality updates (:pr:`5786`) :user:`yarnabrina`
* [MNT] change cycle for making ``SAX2`` and ``PAA2`` primary implementation renamed to ``SAX``, ``PAA`` (:pr:`5799`) :user:`fkiraly`
* [MNT] remove maintainer information from ``CODEOWNERS`` in favour of estimator tags (:pr:`5808`) :user:`fkiraly`
* [MNT] addressing more ``pandas`` deprecations (:pr:`5816`) :user:`fkiraly`
* [MNT] address ``pd.DataFrame.groupby(axis=1)`` deprecation in ``EnsembleForecaster`` (:pr:`5707`) :user:`ninedigits`
* [MNT] add missing ``__author__`` field for ``MultiRocket`` and ``MultiRocketMultivariate`` (:pr:`5698`) :user:`fkiraly`
* [MNT] addressing ``DataFrame.groupby(axis=1)`` deprecation in metric classes (:pr:`5709`) :user:`fkiraly`
* [MNT] added upper bound ``pycatch22<0.4.5`` in ``transformations`` dependency set to avoid installation error on windows (:pr:`5670`) :user:`yarnabrina`
* [MNT] refactoring new CI to fix some bugs and other minor enhancements (:pr:`5638`) :user:`yarnabrina`
* [MNT] Update ``tslearn`` dependency version in pyproject.toml (:pr:`5686`) :user:`DManowitz`
* [MNT] fix several spelling mistakes (:pr:`5639`) :user:`yarnabrina`

Documentation
~~~~~~~~~~~~~

* [DOC] comment in ``CONTRIBUTORS.md`` that source file is ``all-contributorsrc`` (:pr:`5687`) :user:`fkiraly`
* [DOC] improved docstring for ``TrendForecaster`` and ``PolynomialTrendForecaster`` (:pr:`5747`) :user:`fkiraly`
* [DOC] updated algorithm inclusion guide (:pr:`5753`) :user:`fkiraly`
* [DOC] improved docstring for ``TimeSeriesForestClassifier`` (:pr:`5741`) :user:`fkiraly`
* [DOC] fix ``scitype`` string of transformers in API ref (:pr:`5759`) :user:`fkiraly`
* [DOC] improved formatting of tag section in extension templates (:pr:`5812`) :user:`fkiraly`
* [DOC] ``Imputer``: docstring clarity improvement, conditional parameter handling logic (:pr:`3916`) :user:`aiwalter`, :user:`fkiraly``
* [DOC] extension template for time series splitters (:pr:`5769`) :user:`fkiraly`
* [DOC] update soft dependency handling guide for tests with tag based dependency checking (:pr:`5756`) :user:`fkiraly`
* [DOC] fix all import failures in API docs and related missing exports (:pr:`5752`) :user:`fkiraly`
* [DOC] improve clarity in describing ``strategy="refit"`` in forecasting tuners' docstrings (:pr:`5711`) :user:`fkiraly`
* [DOC] correct type statement in forecasting tuner regarding ``forecaster`` (:pr:`5699`) :user:`fkiraly`
* [DOC] various minor API reference improvements (:pr:`5721`) :user:`fkiraly`
* [DOC] add ``ReducerTransform`` and ``DirectReductionForecaster`` to API reference (:pr:`5690`) :user:`fkiraly`
* [DOC] remove outdated ``sktime-dl`` reference in ``README.md`` (:pr:`5685`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`aiwalter`,
:user:`alex-jg3`,
:user:`aurumnpegasus`,
:user:`benheid`,
:user:`chrico-bu-uab`,
:user:`corradomio`,
:user:`DManowitz`,
:user:`fkiraly`,
:user:`hliebert`,
:user:`NguyenChienFelix33`,
:user:`ninedigits`,
:user:`kurayami07734`,
:user:`steenrotsman`,
:user:`tpvasconcelos`,
:user:`tvdboom`,
:user:`yarnabrina`


Version 0.25.0 - 2023-12-26
---------------------------

Release with base class updates and scheduled deprecations:

* framework support for multioutput classifiers, regressors
  (:pr:`5408`, :pr:`5651`, :pr:`5662`) :user:`Vasudeva-bit`, :user:`fkiraly`
* framework support for panel-to-series transformers (:pr:`5351`) :user:`benHeid`
* scheduled deprecations

For last larger feature update, see 0.24.2.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Time series classification and regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* the base class framework now supports multioutput classifiers or regressors.
  All classifiers and regressors are now able to make multioutput predictions,
  including all third party classifiers and regressors.
  A multioutput ``y`` can now be passed, in the form of a 2D ``np.ndarray`` or
  ``pd.DataFrame``, with one column per output.
  The ``predict`` method will then return a predicted output of the same type.
  To retain downwards compatibility, ``predict`` will always return a 1D ``np.ndarray``
  for univariate outputs, this is currently not subject to deprecation.

* Genuinely multioutput classifiers and regressors are labelled with the new
  tag ``capability:multioutput`` being ``True``.
  All other classifiers and regressors broadcast by column of ``y``,
  and a parallelization backend can be selected via ``set_config``,
  by setting the ``backend:parallel`` and ``backend:parallel:params`` configuration
  flags, see the ``set_config`` docstring for details.
  Broadcasting extends automatically to all existing third party classifiers
  and regressors via base class inheritance once ``sktime`` is updated,
  the estimator classes themselves do not need to be updated.

* classifiers and regressors now have a tag ``y_inner_mtype``, this allows extenders
  to specify an internal ``mtype``, of ``Table`` scitype.
  The mtype specified i the tag is the guaranteed
  mtype of ``y`` seen in the private ``_fit`` method.
  The default is the same as previously
  implicit, the ``numpy1D`` mtype.
  Therefore, third party classifiers and regressors do not need to be updated,
  and should be fully upwards compatible.

Transformations
^^^^^^^^^^^^^^^

* the base class framework now supports transformations that aggregate ``Panel`` data
  to ``Series`` data, i.e., panel-to-series transformers, e.g., averaging.
  Such transformers are identified by the tags
  ``scitype:transform-input`` being ``"Panel"``,
  and ``scitype:transform-output`` being ``"Series"``.
  An example is ``Merger``.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* time series splitters, i.e., descendants of ``BaseSplitter``, have moved from
  ``sktime.forecasting.model_selection`` to ``sktime.split``.
  They are no longer available in the old location
  ``sktime.forecasting.model_selection``, since 0.25.0.
  Forecasting tuners are still present in ``sktime.forecasting.model_selection``,
  and their locationn is not subject to deprecation.

* in forecasting ``evaluate``, the order of columns in the return data frame
  has changed. Users should consult the docstring of ``evaluate`` for details.

* in forecasting ``evaluate``, the ``compute`` argument was removed,
  after deprecation in 0.24.0.
  Its purpose was to distinguish lazy or eager evaluation in
  the ``dask`` parallelization backend.
  To switch between lazy and eager evaluation, users should instead
  select ``dask`` or ``dask_lazy`` via the ``backend`` parameter.

* in forecasting ``evaluate``, ``kwargs`` are deprecated, removal has been
  moved to 0.26.0. Users should pass backend parameters via the ``backend_params``
  parameter instead.


Contents
~~~~~~~~

* [ENH] Multioutput capability for all time series classifiers and regressors, broadcasting and tag (:pr:`5408`) :user:`Vasudeva-bit`
* [ENH] Support for panel-to-series transformers, merger transformation (:pr:`5351`) :user:`benHeid`
* [ENH] allow object ``dtype``-s in ``pandas`` based ``Table`` mtype-s (:pr:`5651`) :user:`fkiraly`
* [ENH] intermediate base class for panel tasks - classification, regression (:pr:`5662`) :user:`fkiraly`
* [MNT] CI element to test blogpost notebooks (:pr:`5663`) :user:`fkiraly`, :user:`yarnabrina`
* [MNT] 0.25.0 deprecations and change actions (:pr:`5613`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`benHeid`,
:user:`fkiraly`,
:user:`Vasudeva-bit`,
:user:`yarnabrina`

Version 0.24.2 - 2023-12-24
---------------------------

Highlights
~~~~~~~~~~

* ``FunctionParamFitter`` for custom parameter switching, e.g., applying forecaster or transformer
  conditional on instance properties (:pr:`5630`) :user:`tpvasconcelos`
* ``calibration_plot`` for probabilistic forecasts (:pr:`5632`) :user:`benHeid`
* ``prophet`` based piecewise linear trend forecaster (:pr:`5592`) :user:`sbuse`
* new transformer: dilation mapping (:pr:`5557`) :user:`fspinna`
* custom ``joblib`` backends are now supported in parallelization via ``set_config`` (:pr:`5537`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``dask`` (data container and parallelization back-end) bounds have been updated to ``<2023.12.2``.
* ``holidays`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.40``.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* ``fit_predict`` now allows specification of ``X_pred`` argument for ``predict``.
  If passed, ``X_pred`` is used as ``X`` in ``predict``, instead of ``X``.
  This is useful for forecasters that expect ``X`` to be subset to the
  forecasting horizon.
* custom ``joblib`` backends for hierarchical and multivariate forecast broadcasting
  are now supported. To use a custom ``joblib`` backend, use ``set_config`` to
  set the ``backend:parallel`` configuration flag to ``"joblib"``,
  and set the ``backend`` parameter in the ``dict`` set via ``backend:parallel:params``
  to the name of the custom ``joblib`` backend. Further backend parameters
  can be passed in the same ``dict``. See docstring of ``set_config`` for details.

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* In ``SimpleRNNClassifier``, the ``num_epochs`` parameter is deprecated and has been
  renamed to ``n_epochs``. ``num_epochs`` can be used until ``sktime`` 0.25.last,
  but will be removed in ``sktime`` 0.26.0. A deprecation warning is raised if
  ``num_epochs`` is used.

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* In ``SimpleRNNRegressor``, the ``num_epochs`` parameter is deprecated and has been
  renamed to ``n_epochs``. ``num_epochs`` can be used until ``sktime`` 0.25.last,
  but will be removed in ``sktime`` 0.26.0. A deprecation warning is raised if
  ``num_epochs`` is used.

Transformations
^^^^^^^^^^^^^^^

* custom ``joblib`` backends for hierarchical and multivariate transformer broadcasting
  are now supported. To use a custom ``joblib`` backend, use ``set_config`` to
  set the ``backend:parallel`` configuration flag to ``"joblib"``,
  and set the ``backend`` parameter in the ``dict`` set via ``backend:parallel:params``
  to the name of the custom ``joblib`` backend. Further backend parameters
  can be passed in the same ``dict``. See docstring of ``set_config`` for details.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] improved error messages for input checks in base classes (:pr:`5510`) :user:`fkiraly`
* [ENH] support for custom ``joblib`` backends in parallelization (:pr:`5537`) :user:`fkiraly`
* [ENH] consistent use of ``np.ndarray`` for mtype tags (:pr:`5648`) :user:`fkiraly`
* [ENH] set output format parameter in ``sktime`` internal ``check_is_mtype`` calls to silence deprecation warnings (:pr:`5563`) :user:`benHeid`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] cutoff and forecasting horizon ``loc`` based splitter (:pr:`5575`) :user:`fkiraly`
* [ENH] enable tag related registry tests for ``splitter`` estimator type (:pr:`5576`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``sklearn`` facing coercion utility for ``pd.DataFrame``, to ``str`` columns (:pr:`5550`) :user:`fkiraly`
* [ENH] ``deep_equals`` - clearer return on diffs from ``dtypes`` and ``index``, relaxation of ``MultiIndex`` equality check (:pr:`5560`) :user:`fkiraly`
* [ENH] Uniformization of ``pandas`` index types in mtypes (:pr:`5561`) :user:`fkiraly`
* [ENH] ``n_features`` and ``feature_names`` metadata field for time series mtypes (:pr:`5596`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] expected forecast prediction index utility in ``ForecastingHorizon`` (:pr:`5501`) :user:`fkiraly`
* [ENH] refactor index generation in reducers to use ``ForecastingHorizon`` method (:pr:`5539`) :user:`fkiraly`
* [ENH] fix index name check for reduction forecasters (:pr:`5543`) :user:`fkiraly`
* [ENH] forecaster ``fit_predict`` with ``X_pred`` argument for ``predict`` (:pr:`5562`) :user:`fkiraly`
* [ENH] refactor ``DirectReductionForecaster``to use ``sklearn`` input coercion utility (:pr:`5581`) :user:`fkiraly`
* [ENH] export and test ``DirectReductionForecaster`` (:pr:`5582`) :user:`fkiraly`
* [ENH] ``prophet`` based piecewise linear trend forecaster (:pr:`5592`) :user:`sbuse`
* [ENH] Add ``fit_kwargs`` to ``Prophet`` (:pr:`5597`) :user:`tpvasconcelos`
* [ENH] ``Croston`` test parameters - integer smoothing parameter (:pr:`5608`) :user:`NguyenChienFelix33`
* [ENH] ``prophet`` adapter - safer handling of ``fit_kwargs`` (:pr:`5622`) :user:`fkiraly`

Parameter estimation and hypothesis testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Add new ``FunctionParamFitter`` parameter estimator (:pr:`5630`) :user:`tpvasconcelos`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^
* [ENH] Change ``GGS`` to inherit from ``BaseSeriesAnnotator`` (:pr:`5315`) :user:`Alex-JG3`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] enable testing ``MrSQM`` for persistence in ``nsfa>0`` case after upstream bugfix (:pr:`5171`) :user:`fkiraly`
* [ENH] ``num_epochs`` renamed to ``n_epochs`` in ``SimpleRNNClassifier`` and ``SimpleRNNRegressor`` (:pr:`5607`) :user:`aeyazadil`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] enable tag related registry tests for ``clusterer`` estimator type (:pr:`5576`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] dilation mapping transformer (:pr:`5557`) :user:`fspinna`
* [ENH] second test parameter set for ``TSFreshRelevantFeatureExtractor`` (:pr:`5623`) :user:`fkiraly`

Visualization
^^^^^^^^^^^^^

* [ENH] Add ``calibration_plot`` for probabilistic forecasts (:pr:`5632`) :user:`benHeid`

Test framework
^^^^^^^^^^^^^^

* [ENH] reactivate and fix ``test_multiprocessing_idempotent`` (:pr:`5573`) :user:`fkiraly`
* [ENH] test class register, refactor ``check_estimator`` test gathering to central location (:pr:`5574`) :user:`fkiraly`
* [ENH] conditional testing of objects - test if covering test class has changed (:pr:`5579`) :user:`fkiraly`


Fixes
~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``scitype`` ``coerce_to_list`` parameter, add test coverage (:pr:`5578`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix typos in mtype tags ``np.ndarray``, from erroneous ``nd.array`` (:pr:`5645`) :user:`yarnabrina`

Forecasting
^^^^^^^^^^^

* [BUG] in ``ARCH``, fix ``str`` coercion of ``pd.Series`` name (:pr:`5407`) :user:`Vasudeva-bit`
* [BUG] in reduced regressor, copy or truncate ``X`` if it does not fit the forecasting horizon (:pr:`5542`) :user:`benHeid`
* [BUG] pass correct level argument from ``StatsForecastBackAdapter`` to ``statsforecast`` (:pr:`5587`) :user:`sd2k`
* [BUG] fix ``HierarchyEnsembleForecaster`` returned unexpected predictions if data had only one hierarchy level and forecasters specified by node (:pr:`5615`) :user:`VyomkeshVyas`
* [BUG] fix loss of time zone attribute in ``ForecastingHorizon.to_absolute`` (:pr:`5628`) :user:`fkiraly`
* [BUG] change index match to integer in ``_StatsModelsAdapter`` predict (:pr:`5642`) :user:`ciaran-g`

Transformations
^^^^^^^^^^^^^^^

* [BUG] ``TsFreshFeatureExtractor`` - correct wrong forwarded parameter name ``profiling`` (:pr:`5600`) :user:`sssilvar`
* [BUG] Correct inference of ``TransformerPipeline`` output type tag (:pr:`5625`) :user:`fkiraly`

Visualization
^^^^^^^^^^^^^

* [BUG] Fix multiple figures created by ``plot_windows`` (:pr:`5636`) :user:`benHeid`


Maintenance
~~~~~~~~~~~

* [MNT] CI Modifications (:pr:`5498`) :user:`yarnabrina`
* [MNT] rename variables in base (:pr:`5502`) :user:`yarnabrina`
* [MNT] addressing various ``pandas`` related deprecations (:pr:`5583`) :user:`fkiraly`
* [MNT] Update pre commit hooks (:pr:`5646`) :user:`yarnabrina`
* [MNT] [Dependabot](deps-dev): Update ``pytest-xdist`` requirement from ``<3.4,>=3.3`` to ``>=3.3,<3.5`` (:pr:`5551`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``dask`` requirement from ``<2023.7.1`` to ``<2023.11.1`` (:pr:`5552`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``dask`` requirement from ``<2023.11.1`` to ``<2023.12.2`` (:pr:`5629`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``holidays`` requirement from ``<0.36,>=0.29`` to ``>=0.29,<0.37`` (:pr:`5538`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``holidays`` requirement from ``<0.37,>=0.29`` to ``>=0.29,<0.38`` (:pr:`5565`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``holidays`` requirement from ``<0.38,>=0.29`` to ``>=0.29,<0.40`` (:pr:`5637`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``sphinx-gallery`` requirement from ``<0.15.0`` to ``<0.16.0`` (:pr:`5566`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``pytest-xdist`` requirement from ``<3.5,>=3.3`` to ``>=3.3,<3.6`` (:pr:`5567`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``pycatch22`` requirement from ``<0.4.4`` to ``<0.4.5`` (:pr:`5542`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump actions/download-artifact from 3 to 4 (:pr:`5627`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump actions/setup-python from 4 to 5 (:pr:`5605`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump actions/upload-artifact from 3 to 4 (:pr:`5626`) :user:`dependabot[bot]`

Documentation
~~~~~~~~~~~~~

* [DOC] splitter full API reference page (:pr:`5577`) :user:`fkiraly`
* [DOC] Correct ReST syntax in "RocketClassifier" (:pr:`5564`) :user:`rahulporuri`
* [DOC] Added notebook accompanying Joanna Lenczuk's blog post for testing (:pr:`5604`) :user:`onyekaugochukwu`, :user:`joanlenczuk`
* [DOC] Remove extra parameter in docstring with incorrect definition (:pr:`5617`) :user:`wayneadams`
* [DOC] fix and complete ``YfromX`` docstring (:pr:`5593`) :user:`fkiraly`
* [DOC] fix typo in ``AA_datatypes_and_datasets.ipynb`` panel data loading example (:pr:`5594`) :user:`fkiraly`
* [DOC] forecasting ``evaluate`` utility - improved algorithm description in docstring #5603  (:pr:`5603`) :user:`adamkells`
* [DOC] add explanation about fit/transform instance linking behaviour of rocket transformers (:pr:`5621`) :user:`fkiraly`
* [DOC] Adjust ``FunctionTransformer``'s docstring (:pr:`5634`) :user:`tpvasconcelos`
* [DOC] fixed typo in ``pytest.mark.skipif`` (:pr:`5640`) :user:`yarnabrina`

Contributors
~~~~~~~~~~~~

:user:`adamkells`,
:user:`aeyazadil`,
:user:`Alex-JG3`,
:user:`benHeid`,
:user:`ciaran-g`,
:user:`fkiraly`,
:user:`fspinna`,
:user:`joanlenczuk`,
:user:`NguyenChienFelix33`,
:user:`onyekaugochukwu`,
:user:`rahulporuri`,
:user:`sbuse`,
:user:`sd2k`,
:user:`sssilvar`,
:user:`tpvasconcelos`,
:user:`Vasudeva-bit`,
:user:`VyomkeshVyas`,
:user:`wayneadams`,
:user:`yarnabrina`

Version 0.24.1 - 2023-11-05
---------------------------

Highlights
~~~~~~~~~~

* ``torch`` adapter, LTSF forecasters - linear, D-linear, N-linear (:pr:`4891`, :pr:`5514`) :user:`luca-miniati`
* more period options in ``FourierFeatures``: ``pandas`` period alias and from offset column (:pr:`5513`) :user:`Ram0nB`
* ``iisignature`` backend option for ``SignatureTransformer`` (:pr:`5398`) :user:`sz85512678`
* ``TimeSeriesForestClassifier`` feature importance and optimized interval generation (:pr:`5338`) :user:`YHallouard`
* all stationarity tests from ``arch`` package available as estimators (:pr:`5439`) :user:`Vasudeva-bit`
* Hyperbolic sine transformation and its inverse, ``ScaledAsinhTransformer``, for soft input or output clipping (:pr:`5389`) :user:`ali-parizad`
* estimator serialization: user choice of ``serialization_format`` in ``save`` method and ``mlfow`` plugin,
  support for ``cloudpickle`` (:pr:`5486`, :pr:`5526`) :user:`achieveordie`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``holidays`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.36``.
* ``torch`` is now a managed soft dependency for neural networks (``dl`` test set)

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

* if using ``scikit-base>=0.6.1``: ``set_params`` now recognizes unique ``__``-separated
  suffixes as aliases for full parameter string, e.g., ``set_params(foo="bar")``
  instead of ``set_params(estimator__detrender__forecaster__supercalifragilistic__foo="bar")``.
  This extends to use of parameter names in tuners, e.g., ``ForecastingGridSearchCV`` grids,
  and estimators internally using ``set_params``. The behaviour of ``get_params`` is unchanged.
* ``sktime`` now supports ``cloudpickle`` for estimator serialization, with ``pickle``
  being the standard serialization backend.
  To select the serialization backend, use the ``serialization_format`` parameter
  of estimators' ``save`` method.
  ``cloudpickle`` is already a soft dependency, therefore no dependency change is required.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] test that ``set_params`` recognizes unique suffixes as aliases for full parameter string (:pr:`2931`) :user:`fkiraly`
* [ENH] estimator serialization: user choice of ``serialization_format``, support for ``cloudpickle`` (:pr:`5486`) :user:`achieveordie`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] in ``ExpandingGreedySplitter``, allow ``float`` ``step_size`` (:pr:`5329`) :user:`fkiraly`
* [ENH] Sensible default for ``BaseSplitter.get_n_splits`` (:pr:`5412`) :user:`fkiraly`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Add tecator dataset for time series regression as ``sktime`` onboard dataset (:pr:`5428`) :user:`JonathanBechtel`

Forecasting
^^^^^^^^^^^

* [ENH] ``LTSFLinearForecaster``, ``LTSFLinearNetwork``, ``BaseDeepNetworkPyTorch`` (:pr:`4891`) :user:`luca-miniati`
* [ENH] ``LTSFDLinearForecaster``, ``LTSFNLinearForecaster`` (:pr:`5514`) :user:`luca-miniati`
* [ENH] parallel backend selection for forecasting tuners (:pr:`5430`) :user:`fkiraly`
* [ENH] in ``NaiveForecaster``, add valid variance prediction for in-sample forecasts (:pr:`5499`) :user:`fkiraly`

MLOps & Deployment
~~~~~~~~~~~~~~~~~~

* [ENH] in ``mlflow`` plugin, improve informativity of ``ModuleNotFoundError`` messages (:pr:`5487`) :user:`achieveordie`
* [ENH] Add support for DL estimator persistence in ``mlflow`` plugin (:pr:`5526`) :user:`achieveordie`

Neural networks
^^^^^^^^^^^^^^^

* [ENH] ``pytorch`` adapter for neural networks (:pr:`4891`) :user:`luca-miniati`
* [ENH] add placeholder test suite for neural networks (:pr:`5511`) :user:`fkiraly`

Parameter estimation and hypothesis testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Interface to stationarity tests from ``arch`` package (:pr:`5439`) :user:`Vasudeva-bit`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Add unit tests for change point and segmentation plotting functions (:pr:`5509`) :user:`adamkells`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``TimeSeriesForestClassifier`` feature importance and optimized interval generation (:pr:`5338`) :user:`YHallouard`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Add Hyperbolic Sine transformation and its inverse (ScaledAsinhTransformer) (:pr:`5389`) :user:`ali-parizad`
* [ENH] ``iisignature`` backend option for ``SignatureTransformer`` (:pr:`5398`) :user:`sz85512678`
* [ENH] general inverse transform for ``MSTL`` transformer (:pr:`5457`) :user:`fkiraly`
* [ENH] more period options in ``FourierFeatures``: ``pandas`` period alias and from offset column (:pr:`5513`) :user:`Ram0nB`

Maintenance
~~~~~~~~~~~

* [MNT] Auto format pyproject (:pr:`5425`) :user:`yarnabrina`
* [MNT] bound ``pycatch22<0.4.4`` due to breaking change in patch version (:pr:`5434`) :user:`fkiraly`
* [MNT] removed two recently added hooks (:pr:`5453`) :user:`yarnabrina`
* [MNT] xfail remote data loaders to silence sporadic failures (:pr:`5461`) :user:`fkiraly`
* [MNT] new CI workflow to test extras (:pr:`5375`) :user:`yarnabrina`
* [MNT] Split CI jobs per components with specific soft-dependencies (:pr:`5304`) :user:`yarnabrina`
* [MNT] Programmatically fix (all) typos (:pr:`5424`) :user:`kianmeng`
* [MNT] fix typos in ``base`` module (:pr:`5313`) :user:`yarnabrina`
* [MNT] fix typos in ``forecasting`` module (:pr:`5314`) :user:`yarnabrina`
* [MNT] added missing checkout steps (:pr:`5471`) :user:`yarnabrina`
* [MNT] adds code quality checks without outdated/deprecated Github actions (:pr:`5427`) :user:`yarnabrina`
* [MNT] revert PR #4681 (:pr:`5508`) :user:`yarnabrina`
* [MNT] address ``pandas`` constructor deprecation message from ``ExpandingGreedySplitter`` (:pr:`5500`) :user:`fkiraly`
* [MNT] address deprecation of ``pd.DataFrame.fillna`` with ``method`` arg (:pr:`5497`) :user:`fkiraly`
* [MNT] Dataset downloader testing workflow (:pr:`5437`) :user:`yarnabrina`
* [MNT] shorter names for CI workflow elements (:pr:`5470`) :user:`fkiraly`
* [MNT] skip ``load_solar`` in doctests (:pr:`5528`) :user:`fkiraly`
* [MNT] revert PR #4681 (:pr:`5508`) :user:`yarnabrina`
* [MNT] exclude downloads in "no soft dependencies" CI element (:pr:`5529`) :user:`fkiraly`

* [MNT] [Dependabot](deps): Bump actions/setup-node from 3 to 4 (:pr:`5483`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update pytest-timeout requirement from <2.2,>=2.1 to >=2.1,<2.3 (:pr:`5482`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump tj-actions/changed-files from 39 to 40 (:pr:`5492`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update holidays requirement from <0.35,>=0.29 to >=0.29,<0.36 (:pr:`5443`) :user:`dependabot[bot]`

Documentation
~~~~~~~~~~~~~

* [DOC] fixing docstring example for ``FhPlexForecaster`` (:pr:`4931`) :user:`fkiraly`
* [DOC] Programmatically fix (all) typos (:pr:`5424`) :user:`kianmeng`
* [DOC] comments for readability of ``pyproject.toml`` (:pr:`5472`) :user:`fkiraly`
* [DOC] streamlining API reference, fixing minor issues (:pr:`5466`) :user:`fkiraly`
* [DOC] Fix more typos (:pr:`5478`) :user:`szepeviktor`
* [DOC] update docstring of ``STLTransformer`` to correct statements on inverse and pipelines (:pr:`5455`) :user:`fkiraly`
* [DOC] improved docstrings for ``statsforecast`` estimators (:pr:`5409`) :user:`fkiraly`
* [DOC] add missing API reference entries for five deep learning classifiers (:pr:`5522`) :user:`fkiraly`
* [DOC] fixed docstrings for stationarity tests (:pr:`5531`) :user:`fkiraly`

Fixes
~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix error message in ``_check_python_version`` (:pr:`5473`) :user:`fkiraly`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix bug in deprecation logic of ``kwargs`` in ``evaluate`` that always set
  backend to ``dask_lazy`` if deprecated ``kwargs`` are passed (:pr:`5469`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] Fix ``pandas`` ``FutureWarning`` for silent upcasting (:pr:`5395`) :user:`tpvasconcelos`
* [BUG] fix predict function of ``make_reduction`` (recursive, global) to work with tz aware data (:pr:`5464`) :user:`ciaran-g`
* [BUG] in ``TransformedTargetForecaster``, ensure correct setting of ``ignores-exogenous-X`` tag if forecaster ignores ``X``, but at least one transformer uses ``y=X``, e.g., feature selector (:pr:`5521`) :user:`fkiraly`

Parameter estimation and hypothesis testing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fixed incorrect signs for some stationarity tests (:pr:`5531`) :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] CLASP logic: remove indexes from exclusion zone that are out of range (:pr:`5459`) :user:`Alex-JG3`
* [BUG] in ``ClaSPSegmentation``, deal with ``k`` when it is too large for ``np.argpartition`` (:pr:`5490`) :user:`Alex-JG3`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix missing epochs parameter in ``MCDCNNClassifier._fit`` (#4996) (:pr:`5422`) :user:`pseudomo`
* [BUG] add missing exports five deep learning classifiers (:pr:`5522`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix test excepts for ``SignatureTransformer`` (:pr:`5474`) :user:`fkiraly`

Visualization
^^^^^^^^^^^^^

* [BUG] fix ``plot_series`` prediction interval plotting for 3 or less points in forecasting horizon (:pr:`5494`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`adamkells`,
:user:`Alex-JG3`,
:user:`ali-parizad`,
:user:`ciaran-g`,
:user:`fkiraly`,
:user:`JonathanBechtel`,
:user:`kianmeng`,
:user:`luca-miniati`,
:user:`pseudomo`,
:user:`Ram0nB`,
:user:`sz85512678`,
:user:`szepeviktor`,
:user:`tpvasconcelos`,
:user:`Vasudeva-bit`,
:user:`yarnabrina`,
:user:`YHallouard`


Version 0.24.0 - 2023-10-13
---------------------------

Maintenance release:

* support for python 3.12
* scheduled deprecations
* soft dependency updates

For last non-maintenance content updates, see 0.23.1.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``pykalman`` dependencies have been replaced by the fork ``pykalman-bardo``.
  ``pykalman`` is abandoned, and ``pykalman-bardo`` is a maintained fork.
  This is a soft dependency, and the switch does not affect users installing
  ``sktime`` using one of its dependency sets.
  Mid-term, we expect ``pykalman-bardo`` to be merged back into ``pykalman``,
  after which the dependency will be switched back to ``pykalman``.
* ``holidays`` (transformations soft dependency) bounds have been updated to ``>=0.29,<0.35``.
* ``numba`` (classification, regression, and transformations soft dependency) bounds have been updated to ``>=0.53,<0.59``.
* ``skpro`` (forecasting soft dependency) bounds have been updated to ``>=2.0.0,<2.2.0``.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

* in forecasting tuners ``ForecastingGridSearchCV``, ``ForecastingRandomizedSearchCV``,
  ``ForecastingSkoptSearchCV``, the default of parameter ``tune_by_variable``
  has been switched from ``True`` to ``False``.

Contents
~~~~~~~~

* [MNT] Update ``numba`` requirement from ``<0.58,>=0.53`` to ``>=0.53,<0.59`` (:pr:`5299`, :pr:`5319`) :user:`dependabot[bot]`, :user:`fkiraly`
* [MNT] [Dependabot](deps-dev): Update ``skpro`` requirement from ``<2.1.0,>=2.0.0`` to ``>=2.0.0,<2.2.0`` (:pr:`5396`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update ``holidays`` requirement from ``<0.34,>=0.29`` to ``>=0.29,<0.35`` (:pr:`5342`) :user:`dependabot[bot]`
* [MNT] Migrate from ``pykalman`` to ``pykalman-bardo`` (:pr:`5277`) :user:`mbalatsko`
* [MNT] 0.24.0 deprecations and change actions (:pr:`5404`) :user:`fkiraly`
* 🚀 python 3.12 🚀  (:pr:`5345`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`fkiraly`,
:user:`mbalatsko`


Version 0.23.1 - 2023-10-12
---------------------------

Highlights
~~~~~~~~~~

* all hierarchical/multivariate forecaster and transformer broadcasting can now use parallelization backends ``joblib``, ``dask`` via ``set_config`` (:pr:`5267`, :pr:`5268`, :pr:`5301`, :pr:`5311`, :pr:`5405`) :user:`fkiraly`
* ``PeakTimeFeatures`` transformer to generate indicator features for one or multiple peak hours, days, etc (:pr:`5191`) :user:`ali-parizad`
* ARCH forecaster interfacing ``arch`` package (:pr:`5326`) :user:`Vasudeva-bit`
* forecasting reducer ``YfromX`` now makes probabilistic forecasts when using ``skpro`` probabilistic tabular regressors (:pr:`5271`) :user:`fkiraly`
* forecasting compositors ``ForecastX`` now allows fitting ``forecaster_y`` on forecasted ``X`` (:pr:`5334`) :user:`benHeid`
* lucky dynamic time warping distance and aligner, for use in time series classifiers, regressors, clusterers (:pr:`5341`) :user:`fkiraly`
* splitters have now moved to their own module, ``sktime.split`` (:pr:`5017`) :user:`BensHamza`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``attrs`` is no longer a soft dependency (time series annotation) of ``sktime``
* ``arch`` is now a soft dependency (forecasting) of ``sktime``
* ``skpro`` is now a soft dependency (forecasting) of ``sktime``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* the ``sktime`` framework now inspects estimator type primarily via the tag ``object_type``.
  This is not a breaking change as inheriting from respective base classes automatically sets the tag as well,
  via the tag inheritance system. The type inspection utility ``scitype`` is also unaffected.
  For extenders, the change enables polymorphic and dynamically typed estimators.
* warnings from ``sktime`` can now be silenced on a per-estimator basis via
  the ``warnings`` config that can be set via ``set_config`` (see docstring).

Forecasting
^^^^^^^^^^^

* hierarchical and multivariate forecasts can now use parallelization and distributed backends,
  including ``joblib`` and ``dask``, if the forecast is obtained via broadcasting.
  To enable parallelization, set the ``backend:parallel`` and/or the ``backend:parallel:params``
  configuration flags via ``set_config`` (see docstring) before fitting the forecaster.
  This change instantaneously extends to all existing third party forecasters
  that are interface conformant, via inheritance from the updated base framework.

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* time series regressors now allow single-column ``pd.DataFrame`` as ``y``.
  Current behaviour is unaffected, this is not a breaking change for existing code.

Transformations
^^^^^^^^^^^^^^^

* hierarchical and multivariate transformers can now use parallelization and distributed backends,
  including ``joblib`` and ``dask``, if the transformation is obtained via broadcasting.
  To enable parallelization, set the ``backend:parallel`` and/or the ``backend:parallel:params``
  configuration flags via ``set_config`` (see docstring) before fitting the transformer.
  This change instantaneously extends to all existing third party transformers
  that are interface conformant, via inheritance from the updated base framework.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* time series splitters, i.e., descendants of ``BaseSplitter``, have moved from
  ``sktime.forecasting.model_selection`` to ``sktime.split``.
  The old location ``model_selection`` is deprecated and will be removed in 0.25.0.
  Until 0.25.0, it is still available but will raise an informative warning message.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] warnings config (:pr:`4536`) :user:`fkiraly`
* [ENH] add exports of common utilities in ``utils`` module (:pr:`5266`) :user:`fkiraly`
* [ENH] in scitype check, replace base class register logic with type tag inspection (:pr:`5288`) :user:`fkiraly`
* [ENH] parallelization backend calls in utility module - part 1, refactor to utility module (:pr:`5268`) :user:`fkiraly`
* [ENH] parallelization backend calls in utility module - part 2, backend parameter passing (:pr:`5311`) :user:`fkiraly`
* [ENH] parallelization backend calls in utility module - part 3, backend parameter passing in base class broadcasting (:pr:`5405`) :user:`fkiraly`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] consolidating splitters as their own module with systematic tests and extension (:pr:`5017`, :pr:`5331`) :user:`BensHamza`,  :user:`fkiraly`
* [ENH] allow ``evaluate`` to accept any combination of multiple metrics with correct predict method (:pr:`5192`) :user:`hazrulakmal`
* [ENH] add tests for ``temporal_train_test_split`` (:pr:`5332`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [ENH] dataset loaders module restructure (:pr:`5239`) :user:`hazrulakmal`

Forecasting
^^^^^^^^^^^

* [ENH] Add a ``CurveFitForecaster`` based on ``scipy`` ``optimize_curve`` (:pr:`5240`) :user:`benHeid`
* [ENH] Restructure the ``trend`` forecasters module (:pr:`5242`) :user:`benHeid`
* [ENH] ``YfromX`` - probabilistic forecasts (:pr:`5271`) :user:`fkiraly`
* [ENH] Link ``test_interval_wrappers.py`` to changes in ``evaluate`` for conditional testing (:pr:`5337`) :user:`fkiraly`
* [ENH] ``joblib`` and ``dask`` backends in broadcasting of estimators in multivariate or hierarchical case - part 1, ``VectorizedDF.vectorize_est`` (:pr:`5267`) :user:`fkiraly`
* [ENH] ``joblib`` and ``dask`` backends in broadcasting of estimators in multivariate or hierarchical case - part 2, base class config (:pr:`5301`) :user:`fkiraly`
* [ENH] ARCH model interfacing ``arch`` package (:pr:`5326`) :user:`Vasudeva-bit`
* [ENH] in ``ForecastX``, enable fitting ``forecaster_y`` on forecasted ``X`` (:pr:`5334`) :user:`benHeid`
* [ENH] Skip unnecessary fit in ``ForecastX`` if inner ``forecaster_y`` ignores ``X`` (:pr:`5353`) :user:`yarnabrina`
* [ENH] remove legacy except in ``TestAllEstimators`` for ``predict_proba`` (:pr:`5386`) :user:`fkiraly`

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [ENH] lucky dynamic time warping aligner (:pr:`5341`) :user:`fkiraly`
* [ENH] sensible default ``_get_distance_matrix`` for time series aligners (:pr:`5347`) :user:`fkiraly`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] delegator for pairwise time series distances and kernels (:pr:`5340`) :user:`fkiraly`
* [ENH] lucky dynamic time warping distance (:pr:`5341`) :user:`fkiraly`
* [ENH] simplified delegator interface to ``dtw-python`` based dynamic time warping distances (:pr:`5348`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] in ``BaseRegressor``, allow ``y`` to be 1D ``pd.DataFrame`` (:pr:`5282`) :user:`mdsaad2305`

Transformations
^^^^^^^^^^^^^^^

* [ENH] ``PeakTimeFeatures`` transformer to generate indicator features for one/multiple peak/hours-day-week-, working hours, etc (:pr:`5191`) :user:`ali-parizad`
* [ENH] ``VmdTransformer``, add decompose-forecast-recompose as a docstring example and test (:pr:`5250`) :user:`fkiraly`* [ENH] improve ``evaluate`` failure error message (:pr:`5269`) :user:`fkiraly`
* [ENH] add proper ``inverse_transform`` to ``STLTransformer`` (:pr:`5300`) :user:`fkiraly`
* [ENH] ``joblib`` and ``dask`` backends in broadcasting of estimators in multivariate or hierarchical case - part 1, ``VectorizedDF.vectorize_est`` (:pr:`5267`) :user:`fkiraly`
* [ENH] ``joblib`` and ``dask`` backends in broadcasting of estimators in multivariate or hierarchical case - part 2, base class config (:pr:`5301`) :user:`fkiraly`
* [ENH] Refactor of ``DateTimeFeatures`` tests to ``pytest`` fixtures (:pr:`5397`) :user:`adamkells`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] add error message return to ``deep_equals`` assert in ``test_reconstruct_identical``  (:pr:`4927`) :user:`fkiraly`
* [ENH] incremental testing to also test if any parent class in sktime has changed (:pr:`5379`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] revert update numba requirement from <0.58,>=0.53 to >=0.53,<0.59" (:pr:`5297`) :user:`fkiraly`
* [MNT] bound ``numba<0.58`` (:pr:`5303`) :user:`fkiraly`
* [MNT] Remove ``attrs`` dependency (:pr:`5296`) :user:`Alex-JG3`
* [MNT] simplified CI - merge windows CI step with test matrix (:pr:`5362`) :user:`fkiraly`
* [MNT] towards 3.12 compatibility - replace ``distutils`` calls with equivalent functionality (:pr:`5376`) :user:`fkiraly`
* [MNT] ``skpro`` as a soft dependency (:pr:`5273`) :user:`fkiraly`
* [MNT] removed ``py37.dockerfile`` and update doc entry for CI (:pr:`5356`) :user:`kurayami07734`
* [MNT] [Dependabot](deps): Bump styfle/cancel-workflow-action from 0.11.0 to 0.12.0 (:pr:`5355`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump stefanzweifel/git-auto-commit-action from 4 to 5 (:pr:`5373`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update holidays requirement from <0.33,>=0.29 to >=0.29,<0.34 (:pr:`5276`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update numpy requirement from <1.26,>=1.21.0 to >=1.21.0,<1.27 (:pr:`5275`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update arch requirement from <6.2.0,>=5.6.0 to >=5.6.0,<6.3.0 (:pr:`5392`) :user:`dependabot[bot]`

Documentation
~~~~~~~~~~~~~

* [DOC] prevent line break in ``README.md`` badges table (:pr:`5263`) :user:`fkiraly`
* [DOC] forecasting extension template - add insample capability tags (:pr:`5272`) :user:`fkiraly`
* [DOC] add ``blog`` badge for ``fkiraly``, for ODSC blog post (:pr:`5291`) :user:`fkiraly`
* [DOC] speed improvement of ``partition_based_clustering`` notebook (:pr:`5278`) :user:`alexfilothodoros`
* [DOC] Documented ax argument and the figure in plot_series (:pr:`5325`) :user:`ShreeshaM07`
* [DOC] Improve Readability of Notebook 2 - Classification, Regression & Clustering (:pr:`5312`) :user:`achieveordie`
* [DOC] Added all feature names to docstring for DateTimeFeatures class (:pr:`5283`) :user:`Abhay-Lejith`
* [DOC] ``sktime`` intro notebook (:pr:`3793`) :user:`fkiraly`
* [DOC] Correct code block formatting for pre-commit install command (:pr:`5377`) :user:`alhridoy`
* [DOC] fix broken docstring example of ``AlignerDtwNumba`` (:pr:`5374`) :user:`fkiraly`
* [DOC] fix typo in classification notebook (:pr:`5390`) :user:`pirnerjonas`
* [DOC] Improved PR template for new contributors (:pr:`5381`) :user:`fkiraly`
* [DOC] dynamic docstring for ``set_config`` (:pr:`5306`) :user:`fkiraly`
* [DOC] update docstring of ``temporal_train_test_split`` (:pr:`4170`) :user:`xansh`
* [DOC] Document ``ax`` argument and the figure in ``plot_series`` (:pr:`5325`) :user:`ShreeshaM07`

Fixes
~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``temporal_train_test_split`` for hierarchical and panel data in case where ``fh`` is not passed (:pr:`5330`) :user:`fkiraly`
* [BUG] allow ``alpha`` and ``coverage`` to be passed again via metrics to ``evaluate`` (:pr:`5354`) :user:`fkiraly`, :user:`benheid`

Forecasting
^^^^^^^^^^^

* [BUG] fix ``STLForecaster`` tag ``ignores-exogeneous-X`` to be correctly set for composites (:pr:`5365`) :user:`yarnabrina`
* [BUG] ``statsforecast 1.6.0`` compatibility - in ``statsforecast`` adapter, fixing ``RuntimeError: dictionary changed size during iteration`` (:pr:`5317`) :user:`arnaujc91`
* [BUG] ``statsforecast 1.6.0`` compatibility - fix argument differences between ``sktime`` and ``statsforecast`` (:pr:`5393`) :user:`luca-miniati`
* [BUG] Fix ``ARCH._check_predict_proba`` (:pr:`5384`) :user:`Vasudeva-bit`

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [BUG] minor fixes to ``NaiveAligner`` (:pr:`5344`) :user:`fkiraly`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix ``numba`` errors when calling ``tslearn`` ``lcss`` (:pr:`5368`) :user:`benHeid`, :user:`BensHamza`, :user:`fkiraly`


Transformations
^^^^^^^^^^^^^^^

* [BUG] in ``Imputer``, fix ``y`` not being passed in ``method="forecaster"`` (:pr:`5287`) :user:`fkiraly`
* [BUG] ensure ``Catch22`` parameter setting ``n_jobs = -1`` uses all cores (:pr:`5361`) :user:`julnow`

Visualization
^^^^^^^^^^^^^

* [BUG] Fix inconsistent date/time index in ``plot_windows`` #4919 (:pr:`5321`) :user:`geronimos`

Contributors
~~~~~~~~~~~~

:user:`Abhay-Lejith`,
:user:`achieveordie`,
:user:`adamkells`,
:user:`Alex-JG3`,
:user:`alexfilothodoros`,
:user:`alhridoy`,
:user:`ali-parizad`,
:user:`arnaujc91`,
:user:`benHeid`,
:user:`BensHamza`,
:user:`fkiraly`,
:user:`geronimos`,
:user:`hazrulakmal`,
:user:`julnow`,
:user:`kurayami07734`,
:user:`luca-miniati`,
:user:`mdsaad2305`,
:user:`pirnerjonas`,
:user:`ShreeshaM07`,
:user:`Vasudeva-bit`,
:user:`xansh`,
:user:`yarnabrina`

Version 0.23.0 - 2023-09-17
---------------------------

Maintenance release - scheduled deprecations.

For last non-maintenance content updates, see 0.22.1.

Contents
~~~~~~~~

* end of change period in column naming convention for univariate probabilistic forecasts,
  see below for details for users and developers
* scheduled 0.23.0 deprecation actions

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting - change of column naming for univariate probabilistic forecasts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Returns of forecasters' ``predict_quantiles`` and ``predict_intervals``
are now consistent between the univariate case and multivariate cases:
the name of the uppermost (0-indexed) column level is always the variable name.

Previously, in the univariate case, it was always ``Coverage`` or ``Quantiles``.

This has been preceded by a change transition period since 0.21.0.
See the 0.21.0 and 0.22.0 changelogs for further details.

Users and extenders who have not yet completed their downstream actions
should remain on 0.22.X until they have completed their actions, and then upgrade
to 0.23.0 or later.


Version 0.22.1 - 2023-09-17
---------------------------

Highlights
~~~~~~~~~~

* Graphical Pipelines for any learning task (polymorphic) - ``Pipeline`` (:pr:`4652`) :user:`benHeid`
* all ``tslearn`` distances and kernels are now available in ``sktime`` (:pr:`5039`) :user:`fkiraly`
* new transformer: ``VmdTransformer`` (variational mode decomposition) - ``vmdpy`` is now maintained in ``sktime`` (:pr:`5129`) :user:`DaneLyttinen`, :user:`vrcarva`
* new transformer: interface to ``statsmodels`` MSTL (:pr:`5125`) :user:`luca-miniati`
* new classifier: ``MrSEQL`` time series classifier (:pr:`5178`) :user:`lnthach`, :user:`heerme`, :user:`fkiraly`
* new ``sktime`` native probability distributions: Cauchy, empirical, Laplace, Student t (:pr:`5050`, :pr:`5094`, :pr:`5161`) :user:`Alex-JG3`, :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``sktime`` now supports ``pandas`` 2.1.X
* ``sktime`` now supports ``holidays`` 0.32 (soft dependency)
* ``sktime`` now supports ``statsforecast`` 1.6.X (soft dependency)

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Transformations
^^^^^^^^^^^^^^^

* Transformations (``BaseTransformer`` descendants) now have two new optional tags:
  ``"capability:inverse_transform:range"`` and ``"capability:inverse_transform:exact"``.
  The tags should be specified in the ``_tags`` class attribute of the transformer,
  in case the transformer implements ``inverse_transform`` and has
  the restrictions described below.

  * ``"capability:inverse_transform:range"`` specifies the domain of invertibility of
    the transform, must be list [lower, upper] of float".
    This is used for documentation and testing purposes.
  * ``"capability:inverse_transform:exact"`` specifies whether ``inverse_transform``
    is expected to be an exact inverse to ``transform``.
    This is used for documentation and testing purposes.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] test for specification conformance of tag register (:pr:`5170`) :user:`fkiraly`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] speed up ``BaseSplitter`` boilerplate (:pr:`5063`) :user:`fkiraly`
* [ENH] Allow unrestricted ID string for ``BaseBenchmarking`` (:pr:`5130`) :user:`hazrulakmal`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] set mirrors for time series classification data loaders (:pr:`5260`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] speed up tests in ``test_fh`` (:pr:`5098`) :user:`fkiraly`
* [ENH] Robustifying ``ForecastingGridSearchCV`` towards free kwarg methods in estimators, e.g., graphical pipeline (:pr:`5210`) :user:`benHeid`
* [ENH] make ``statsforecast`` adapter compatible with optional ``predict`` ``level`` arguments, and different init param sets (:pr:`5112`) :user:`arnaujc91`
* [ENH] fix ``test_set_freq_hier`` for ``pandas 2.1.0`` (:pr:`5185`) :user:`fkiraly`

Pipelines
^^^^^^^^^

* [ENH] Graphical Pipelines for any learning task (polymorphic) (:pr:`4652`) :user:`benHeid`
* [ENH] add warning that graphical pipeline is experimental (:pr:`5235`) :user:`benHeid`
* [ENH] ensure ``ForecastingPipeline`` is compatible with "featurizers" (:pr:`5252`) :user:`fkiraly`

Probability distributions and simulators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Student's t-distribution (:pr:`5050`) :user:`Alex-JG3`
* [ENH] empirical distribution (:pr:`5094`) :user:`fkiraly`
* [ENH] Laplace distribution (:pr:`5161`) :user:`fkiraly`
* [ENH] Refactor of ``BaseDistribution`` and descendants - generalised distribution param broadcasting in base class (:pr:`5176`) :user:`Alex-JG3`
* [ENH] fixture names in probability distribution tests (:pr:`5159`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``MrSEQL`` time series classifier (:pr:`5178`) :user:`fkiraly`, :user:`lnthach`, :user:`heerme`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``tslearn`` distances and kernels including adapter (:pr:`5039`) :user:`fkiraly`
* [ENH] conditional execution of ``test_distance`` and ``test_distance_params`` (:pr:`5099`) :user:`fkiraly`
* [ENH] refactor and add conditional execution to ``numba`` based distance tests (:pr:`5141`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Interface statsmodels MSTL - transformer (:pr:`5125`) :user:`luca-miniati`
* [ENH] VMD (variational mode decomposition) transformer based on ``vmdpy`` (:pr:`5129`) :user:`DaneLyttinen`
* [ENH] add tag for inexact ``inverse_transform``-s (:pr:`5166`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] speed up ``test_probabilistic_metrics`` by explicit fixture generation instead of using forecaster fit/predict (:pr:`5115`) :user:`Ram0nB`
* [ENH] test forecastingdata downloads only on a small random subset (:pr:`5146`) :user:`fkiraly`
* [ENH] widen scope of change-conditional test execution (:pr:`5100`, :pr:`5135`, :pr:`5147`) :user:`fkiraly`
* [ENH] differential testing of ``cython`` based estimators (:pr:`5206`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] upgrade CI runners to latest stable images (:pr:`5031`) :user:`yarnabrina`
* [MNT] bound ``statsforecast<1.6.0`` due to recent failures (:pr:`5149`) :user:`fkiraly`
* [MNT] test forecastingdata downloads only on a small random subset (:pr:`5146`) :user:`fkiraly`
* [MNT] lower dep bound compatibility patch - ``binom_test`` (:pr:`5152`) :user:`fkiraly`
* [MNT] fix dependency isolation of ``DateTimeFeatures`` tests (:pr:`5154`) :user:`fkiraly`
* [MNT] move fixtures in ``test_reduce_global`` to ``pytest`` fixtures (:pr:`5157`) :user:`fkiraly`
* [MNT] move fixtures in ``test_dropna`` to ``pytest`` fixtures (:pr:`5153`) :user:`fkiraly`
* [MNT] Extra dependency specifications per component (:pr:`5136`) :user:`yarnabrina`
* [MNT] add ``numba`` to ``python`` 3.11 tests (:pr:`5179`) :user:`fkiraly`
* [MNT] autoupdate for copyright range in ``sphinx`` docs (:pr:`5212`) :user:`fkiraly`
* [MNT] move ``Pipeline`` exception from ``test_all_estimators`` to test ``_config`` (:pr:`5251`) :user:`fkiraly`
* [MNT] Update versions of pre commit hooks and fix ``E721`` issues pointed out by ``flake8`` (:pr:`5163`) :user:`yarnabrina`
* [MNT] [Dependabot](deps-dev): Update sphinx-gallery requirement from <0.14.0 to <0.15.0 (:pr:`5124`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update pandas requirement from <2.1.0,>=1.1.0 to >=1.1.0,<2.2.0 (:pr:`5183`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump actions/checkout from 3 to 4 (:pr:`5189`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update holidays requirement from <0.32,>=0.29 to >=0.29,<0.33 (:pr:`5214`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps-dev): Update statsforecast requirement from <1.6,>=0.5.2 to >=0.5.2,<1.7 (:pr:`5215`) :user:`dependabot[bot]`

Documentation
~~~~~~~~~~~~~

* [DOC] provisions for treasurer role (:pr:`4798`) :user:`marrov`, :user:`kiraly`
* [DOC] Fix ``make_pipeline``, ``make_reduction``, ``window_summarizer`` & ``load_forecasting`` data docstrings  (:pr:`5065`) :user:`hazrulakmal`
* [DOC] minor docstring typo fixes in ``_DelegatedForecaster`` module (:pr:`5168`) :user:`fkiraly`
* [DOC] update forecasting extension template on ``predict_proba`` (:pr:`5138`) :user:`fkiraly`
* [DOC] speed-up tutorial notebooks - deep learning classifiers (:pr:`5169`) :user:`alexfilothodoros`
* [DOC] Fix rendering issues in ``ColumnEnsembleForecaster`` docstring, add ``ColumnEnsembleTransformer`` example (:pr:`5201`) :user:`benHeid`
* [DOC] installation instruction docs for learning task specific dependency sets (:pr:`5204`) :user:`fkiraly`
* [DOC] add allcontributors badges of benHeid (:pr:`5209`) :user:`benHeid`
* [DOC] fix typo in forecaster API reference (:pr:`5211`) :user:`fkiraly`
* [DOC] Fixing typos in ``installation.rst`` (:pr:`5213`) :user:`Akash190104`
* [DOC] Added examples for ``temporal_train_test_split`` docstring (:pr:`5216`) :user:`JonathanBechtel`
* [DOC] update to README badges: license, tutorials, and community further up (:pr:`5227`) :user:`fkiraly`
* [DOC] Simple edits to make ``STLForecaster`` docstring render properly (:pr:`5220`) :user:`hazrulakmal`
* [DOC] fixing ``conftest.py`` docstrings (:pr:`5228`) :user:`fkiraly`
* [DOC] clarify docstrings in ``trend.py`` (:pr:`5231`) :user:`sniafas`

Fixes
~~~~~

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] in splitters, correctly infer series frequency for datetime datatype if not given (:pr:`5009`) :user:`hazrulakmal`
* [BUG] fix ``BaseWindowSplitter`` ``get_n_split`` method for hierarchical data (:pr:`5012`) :user:`hazrulakmal`

Forecasting
^^^^^^^^^^^

* [BUG] fix check causing exception in ``ConformalIntervals`` in ``_predict`` (:pr:`5134`) :user:`fkiraly`
* [BUG] ensure forecasting tuners do not vectorize over columns (variables) (:pr:`5145`) :user:`fkiraly`, :user:`SmirnGregHM`
* [BUG] Fix tag to indicate support of exogenous features by ``NaiveForecaster`` (:pr:`5162`) :user:`yarnabrina`
* [BUG] Add missing ``return`` statement for ``y_dict`` in tests for composite forecasters (:pr:`5253`) :user:`BensHamza`
* [BUG] Fix missing ``y_train`` key in ``y_dict`` in tests for composite forecasters (:pr:`5255`) :user:`fkiraly`
* [BUG] Fix ``ForecastKnownValues`` failure on ``pd-multiindex`` (:pr:`5256`) :user:`mattiasatqubes`

Pipelines
^^^^^^^^^

* [BUG] fix missing ``Pipeline`` export in ``sktime.pipeline`` (:pr:`5232`) :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] prevent exception in ``PyODAnnotator.get_test_params`` (:pr:`5151`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] adds missing tag ``skip-inverse-transform`` to ``ColumnSelect`` (:pr:`5208`) :user:`benHeid`

Visualisations
^^^^^^^^^^^^^^

* [BUG] address ``matplotlib`` deprecation of ``label`` attribute (:pr:`5246`) :user:`benHeid`


Contributors
~~~~~~~~~~~~

:user:`Akash190104`,
:user:`Alex-JG3`,
:user:`alexfilothodoros`,
:user:`arnaujc91`,
:user:`benHeid`,
:user:`BensHamza`,
:user:`DaneLyttinen`,
:user:`fkiraly`,
:user:`hazrulakmal`,
:user:`heerme`,
:user:`lnthach`,
:user:`JonathanBechtel`,
:user:`luca-miniati`,
:user:`mattiasatqubes`,
:user:`Ram0nB`,
:user:`SmirnGregHM`,
:user:`sniafas`,
:user:`vrcarva`,
:user:`yarnabrina`

Version 0.22.0 - 2023-08-18
---------------------------

Maintenance release - dependency updates, scheduled deprecations.

For last non-maintenance content updates, see 0.21.1.

Contents
~~~~~~~~

* midpoint of change period in column naming convention for univariate probabilistic forecasts,
  in preparation for 0.23.0 - see below for details for users and developers
* scheduled 0.22.0 deprecation actions

Dependency changes
~~~~~~~~~~~~~~~~~~

* the ``deprecated`` has been removed as a core dependency of ``sktime``.
  No action is required of users
  or developers, as the package was used only for internal deprecation actions.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting - change of column naming for univariate probabilistic forecasts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From 0.23.0, returns of forecasters' ``predict_quantiles`` and ``predict_intervals``
in the univariate case will be made consistent with the multivariate case:
the name of the uppermost (0-indexed) column level will always be the variable name.
Previously, in the univariate case, it was always ``Coverage`` or ``Quantiles``.

The transition period is managed by the ``legacy_interface`` argument of the two methods.
See the 0.21.0 changelog for further details.

In 0.22.0, the ``legacy_interface`` argument defaults have been changed to ``False``,
which ensures outputs are of the future, post-change naming convention.

Reminder of recommended action for users:

* Users should aim to upgrade dependent code to ``legacy_interface=False`` behaviour by 0.21.last,
  and to remove ``legacy_interface`` arguments after 0.22.0 and before 0.23.0.
  Users who need more time to upgrade dependent code can set ``legacy_interface=True`` until 0.22.last.

Extenders should use the new ``"pred_int:legacy_interface:testcfg"`` config field to upgrade their third party extensions,
this is as described in the 0.21.0 changelog.

Transformations
^^^^^^^^^^^^^^^

* in ``DateTimeFeatures``, the feature ``hour_of_week`` feature
  has been added to the ``"comprehensive"`` feature set.
  Users who would like to continue using the previous feature set
  should use the argument ``manual_selection`` instead.

List of PR
~~~~~~~~~~

* [MNT] ``failfast=False`` in the release workflow (:pr:`5120`) :user:`fkiraly`
* [MNT] 0.22.0 release action - deprecate ``deprecated`` in 0.21.0, remove in 0.22.0 (:pr:`4822`) :user:`fkiraly`
* [MNT] 0.22.0 deprecations and change actions (:pr:`5106`) :user:`fkiraly`


Version 0.21.1 - 2023-08-16
---------------------------

Highlights
~~~~~~~~~~

* holiday feature transformers (country, financial holidays; 1:1 interface) based on ``holidays`` (:pr:`4893`, :pr:`4909`) :user:`VyomkeshVyas`, :user:`yarnabrina`
* ``DropNA`` transformer to drop rows or columns with nan (:pr:`5049`) :user:`hliebert`
* ``ExpandingGreedySplitter`` for slicing test sets from end (:pr:`4917`) :user:`davidgilbertson`
* ``statsforecast`` interfaces: MSTL forecaster, ARCH family forecasters (:pr:`4865`, :pr:`4938`) :user:`luca-miniati`, :user:`eyjo`
* full rework of time series classification notebook (:pr:`5045`) :user:`fkiraly`
* improved developer experience - speedups for testing :user:`julia-kraus`, :user:`tarpas`, :user:`benheid`, :user:`fkiraly`, :user:`yarnabrina`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* Time series aligners now accept all ``Panel`` mtypes as input,
  from only ``df-list`` previously. This is not a breaking change.
* Time series aligners now have a tag ``"alignment_type"``, which can have values
  ``"full"`` and ``"partial"``, to distinguish between a full and partial alignment
  produced by ``get_alignment``. The tag can depend on parameters of the aligner.

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Pairwise transformers now have a tag ``"pwtrafo_type"``, which can have values
  ``"kernel"``, ``"distance"``, or ``"other"``, to allow the user to inspect
  whether the transformer is a kernel or distance transformer.
  This does not impact the interface. The tag is mainly for search and retrieval
  by the user. This also allows to check against methodological requirements
  of estimators, e.g., support vector machines requiring a kernel.
  However, as stated, this is not enforced by the base interface.

Enhancements
~~~~~~~~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Speed-up ``deep_equals`` - lazy evaluation of costly error message string coercions (:pr:`5044`) :user:`benHeid`
* [ENH] sktime str/object aliasing registry mechanism (:pr:`5058`) :user:`fkiraly`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] private ``split_loc`` and tag to control dispatch of ``split_series`` to ``split`` vs ``split_loc`` (:pr:`4903`) :user:`fkiraly`
* [ENH] applying forecasting metrics disregarding index - docstrings and tests (:pr:`4960`) :user:`fkiraly`
* [ENH] metrics classes - add testing parameters (:pr:`5097`) :user:`fkiraly`
* [ENH] tests and fixes for ``numpy`` weights in performance metrics (:pr:`5086`) :user:`fkiraly`
* [ENH] input checks for ``BaseBenchmark``, allow ``add_estimator`` to accept multiple estimators (:pr:`4877`) :user:`hazrulakmal`
* [ENH] tests and fixes for ``numpy`` weights in performance metrics - probabilistic metrics (:pr:`5104`) :user:`fkiraly`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] rework data loader module, ability to specify download mirrors (:pr:`4985`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] improvements to ``_ColumnEstimator`` - refactor to reduce coupling with ``BaseForecaster`` (:pr:`4791`) :user:`fkiraly`
* [ENH] rewrite ``test_probabilistic_metrics`` using proper ``pytest`` fixtures (:pr:`4946`) :user:`julia-kraus`
* [ENH] add expanding greedy splitter (:pr:`4917`) :user:`davidgilbertson`
* [ENH] Interface ``statsforecast`` MSTL, ``statsforecast`` back-adapter (:pr:`4865`) :user:`luca-miniati`
* [ENH] contiguous ``fh`` option for ``FhPlexForecaster`` (:pr:`4926`) :user:`fkiraly`
* [ENH] ensure robustness of ``StatsForecastBackAdapter`` w.r.t. change of ``predict_interval`` return format (:pr:`4991`) :user:`fkiraly`
* [ENH] improve ``SARIMAX`` test parameter coverage (:pr:`4932`) :user:`janpipek`
* [ENH] interface to ``statsforecast`` ARCH family estimators (:pr:`4938`) :user:`eyjo`
* [ENH] add test cases for ``Croston`` and ``ExponentialSmoothing`` (:pr:`4935`) :user:`Gigi1111`
* [ENH] applying forecasting metrics disregarding index - docstrings and tests (:pr:`4960`) :user:`fkiraly`
* [ENH] alias strings for ``scoring`` argument in forecasting tuners (:pr:`5058`) :user:`fkiraly`
* [ENH] allow ``YfromX`` to take missing data (:pr:`5062`) :user:`eenticott-shell`

Parameter estimators
^^^^^^^^^^^^^^^^^^^^

* [ENH] speed up parameter fitter base class boilerplate (:pr:`5057`) :user:`fkiraly`

Probability distributions and simulators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] add length option to ``_bottom_hier_datagen`` hierarchical data generator, speed up ``ReconcilerForecaster`` doctest (:pr:`4979`) :user:`fkiraly`

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [ENH] edit distance alignment algorithms from ``sktime`` native ``numba`` based aligners (:pr:`5075`) :user:`fkiraly`
* [ENH] extend ``BaseAligner.fit`` with input conversion (:pr:`5077`) :user:`fkiraly`
* [ENH] naive multiple aligners for baseline comparisons (:pr:`5076`) :user:`fkiraly`
* [ENH] tag for full/partial alignment, exact tests for full alignment output (:pr:`5080`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] full rework of time series classification notebook (:pr:`5045`) :user:`fkiraly`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Explicit centroid init for ``TimeSeriesLloyds``, ``TimeSeriesKMeans`` and ``TimeSeriesKMedoids`` (:pr:`5001`) :user:`Alex-JG3`
* [ENH] generalized ``tslearn`` adapter and clusterer refactor (:pr:`4992`) :user:`fkiraly`
* [ENH] interface to all ``tslearn`` clusterers (:pr:`5037`) :user:`fkiraly`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] distance/kernel tag, uniformize base module (:pr:`5038`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] inverse transform for ``CosineTransformer``, tag handling for limited range of invertibility (:pr:`3671`) :user:`fkiraly`
* [ENH] Holiday indicator transformers by country or market based on ``holidays`` package (:pr:`4893`) :user:`yarnabrina`
* [ENH] ``HolidayFeatures`` transformer (:pr:`4909`) :user:`VyomkeshVyas`
* [ENH] enable use of ``TabularToSeriesAdaptor`` with feature selectors, and passing of ``y`` (:pr:`4978`) :user:`fkiraly`
* [ENH] speed up ``BaseTransformer`` checks and conversion boilerplate (:pr:`5036`) :user:`fkiraly`
* [ENH] ``DropNA`` transformer to drop rows or columns with nan (:pr:`5049`) :user:`hliebert`
* [ENH] speed up ``Lag`` transformer (:pr:`5035`) :user:`fkiraly`
* [ENH] option to remember data in ``SplitterSummarizer`` (:pr:`5070`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] speed-up test collection by improvements to ``_testing.scenarios`` (:pr:`4901`) :user:`tarpas`
* [ENH] test for more than one parameter sets per estimator (:pr:`2862`) :user:`fkiraly`
* [ENH] remove ``sklearn`` dependency in ``test_get_params`` (:pr:`5011`) :user:`fkiraly`
* [ENH] testing only estimators from modules that have changed compared to ``main`` (:pr:`5019`) :user:`fkiraly`, :user:`yarnabrina`
* [ENH] dependency and diff test switch for individual estimators to decorate non-suite tests (:pr:`5084`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] add ``statsforecast`` to the ``pandas2`` compatible dependency set (:pr:`4878`) :user:`fkiraly`
* [MNT] bound ``dask<2023.7.1`` to diagnose and remove bug #4925 from ``main`` (:pr:`4928`) :user:`fkiraly`
* [MNT] [Dependabot](deps-dev): Update sphinx-design requirement from <0.5.0 to <0.6.0 (:pr:`4969`) :user:`dependabot[bot]`
* [MNT] speed up ``test_gscv_proba`` test (:pr:`4962`) :user:`fkiraly`
* [MNT] speed up ``test_stat`` benchmarking test (:pr:`4990`) :user:`fkiraly`
* [MNT] speed up clustering dunder test (:pr:`4982`) :user:`fkiraly`
* [MNT] speed up various tests in the forecasting module (:pr:`4963`) :user:`fkiraly`
* [MNT] speed up basic ``check_estimator`` tests (:pr:`4980`) :user:`fkiraly`
* [MNT] speed up costly redundant ``ElasticEnsemble`` classifier doctest (:pr:`4981`) :user:`fkiraly`
* [MNT] address various deprecation warnings (:pr:`5018`) :user:`fkiraly`
* [MNT] rename ``TestedMockClass`` to ``MockTestedClass`` (:pr:`5005`) :user:`fkiraly`
* [MNT] updated sphinx intersphinx links for other libraries (:pr:`5016`) :user:`yarnabrina`
* [MNT] fix duplication of ``pytest`` ``durations`` parameter in CI (:pr:`5034`) :user:`fkiraly`
* [MNT] speed up various non-suite tests (:pr:`5027`) :user:`fkiraly`
* [MNT] speed up various non-suite tests, part 2 (:pr:`5071`) :user:`fkiraly`
* [MNT] add more soft dependencies to ``show_versions`` (:pr:`5059`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] minor improvements to the dependencies guide (:pr:`4896`) :user:`fkiraly`
* [DOC] remove outdated references from transformers API (:pr:`4895`) :user:`fkiraly`
* [DOC] Installation documentation: Pip install without soft dependencies for conda environments (:pr:`4936`) :user:`Verogli`
* [DOC] clarifications to different installations in install documentation (:pr:`4937`) :user:`julia-kraus`
* [DOC] Contributors update (:pr:`4892`) :user:`fkiraly`
* [DOC] correct docstring of ``BaseForecaster.score``, reference to use of non-symmetric MAPE (:pr:`4948`) :user:`MBristle`
* [DOC] Contributors update (:pr:`4944`) :user:`fkiraly`
* [DOC] remove duplication of troubleshooting 'matches_not_found' in install instructions (:pr:`4956`) :user:`julia-kraus`
* [DOC] Contributors update (:pr:`4961`) :user:`fkiraly`
* [DOC] Resolve broken link (governance) in README.md (:pr:`4942`) :user:`eyjo`
* [ENH] in doc build, add copy clipboard button for Example sections (#5015) (:pr:`5015`) :user:`yarnabrina`
* [DOC] improve description of ``scoring`` in docstrings of tuning forecasters such as ``ForecastingGridSearchCV`` (:pr:`5022`) :user:`fkiraly`
* [DOC] API reference for time series aligners (:pr:`5074`) :user:`fkiraly`
* [DOC] Contributors update (:pr:`5010`) :user:`fkiraly`
* [DOC] improve formatting of docstring examples (:pr:`5078`) :user:`yarnabrina`
* [DOC] Contributors update (:pr:`5085`) :user:`fkiraly`
* [DOC] docstring example for ``PinballLoss`` (#5068) (:pr:`5068`) :user:`Ram0nB`
* [DOC] Contributors update (:pr:`5088`) :user:`fkiraly`

Fixes
~~~~~

BaseObject and base framework
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] in ``craft``, fix false positive detection of ``True``, ``False`` as class names (:pr:`5066`) :user:`fkiraly`

Benchmarking, Metrics, Splitters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] use correct arguments in ``geometric_mean_absolute_error`` (:pr:`4987`) :user:`yarnabrina`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix ``vectorize_est`` returning jumbled rows for row vectorization, pd.DataFrame return, if row names were not lexicographically ordered (:pr:`5110`) :user:`fkiraly`, :user:`hoesler`

Forecasting
^^^^^^^^^^^

* [BUG] clarify forecasting tuning estimators' docstrings and error messages in case of ``refit=False`` (:pr:`4945`) :user:`fkiraly`
* [BUG] fix ``ConformalIntervals`` failure if wrapped estimator supports hierarchical mtypes (:pr:`5091`, :pr:`5093`) :user:`fkiraly`

Parameter estimators
^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``PluginParamsForecaster`` in ``params: dict`` case (:pr:`4922`) :user:`fkiraly`

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix missing transpose in ``AlignerDtwNumba`` (:pr:`5080`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``sklearn`` interface non-conformance for estimators in ``_proximity_forest.py``, add further test parameter sets (:pr:`3520`) :user:`Abelarm`, :user:`fkiraly`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] add informative error messages for incompatible ``scitype`` in ``BaseClusterer`` (:pr:`4958`) :user:`achieveordie`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix ``DataConversionWarning`` in ``FeatureSelection`` (:pr:`4883`) :user:`fkiraly`
* [BUG] Fix forecaster based imputation strategy in ``Imputer`` if forecaster requires ``fh`` in ``fit`` (:pr:`4999`) :user:`MCRE-BE`
* [BUG] fix ``Differencer`` for integer index (:pr:`4984`) :user:`fkiraly`
* [BUG] Fix ``Differencer.inverse_transform`` on train data if ``na_handling=fill_zero`` (:pr:`4998`) :user:`benHeid`, :user:`MCRE-BE`
* [BUG] fix wrong logic for ``index_out="shift"`` in ``Lag`` transformer (:pr:`5069`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`Abelarm`,
:user:`achieveordie`,
:user:`Alex-JG3`,
:user:`benHeid`,
:user:`davidgilbertson`,
:user:`eenticott-shell`,
:user:`eyjo`,
:user:`fkiraly`,
:user:`Gigi1111`,
:user:`hazrulakmal`,
:user:`hliebert`,
:user:`janpipek`,
:user:`julia-kraus`,
:user:`luca-miniati`,
:user:`MBristle`,
:user:`MCRE-BE`,
:user:`Ram0nB`,
:user:`tarpas`,
:user:`Verogli`,
:user:`VyomkeshVyas`,
:user:`yarnabrina`

Version 0.21.0 - 2023-07-19
---------------------------

Maintenance release - dependency updates, scheduled deprecations.

For last non-maintenance content updates, see 0.20.1.

Contents
~~~~~~~~

* ``sktime`` is now compatible with ``sklearn 1.3.X``
* start of change in column naming convention for univariate probabilistic forecasts,
  in preparation for 0.23.0 - see below for details for users and developers
* scheduled 0.21.0 deprecation actions

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``scikit-learn`` version bounds now allow versions ``1.3.X``
* the ``deprecated`` package is deprecated as a core dependency of ``sktime``, and
  will cease to be a dependency from 0.22.0 onwards. No action is required of users
  or developers, as the package was used only for internal deprecation actions.
* ``pycatch22`` has been added back as a soft dependency, after python 3.7 EOL

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting - change of column naming for univariate probabilistic forecasts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From 0.23.0, returns of forecasters' ``predict_quantiles`` and ``predict_intervals``
in the univariate case will be made consistent with the multivariate case:
the name of the uppermost (0-indexed) column level will always be the variable name.
Previously, in the univariate case, it was always ``Coverage`` or ``Quantiles``,
irrespective of the variable name present in ``y``, whereas in the multivariate case,
it was always the variable names present in ``y``.

The change will take place over two MINOR cycles, 0.21.X (early phase) and 0.22.X (late phase),
the union of which makes up the change period.
We explain the schedule below, for users, and then for maintainers of third party forecasters ("extenders").

Users should use a new, temporary ``legacy_interface`` argument to handle the change:

* Users - change period. The two forecaster methods ``predict_quantiles`` and ``predict_intervals``
  will have a new boolean argument, ``legacy_interface``. If ``True``, the methods
  produce returns with the current naming convention. If ``False``, the methods produce
  returns with the future, post-change naming convention.
* Users - early and late phase. In the early phase (0.21.X), the default value of ``legacy_interface``
  will be ``True``. In the late phase (0.22.X), the default value of ``legacy_interface`` will be ``False``.
  This change of default will occur in 0.22.0, and may be breaking for users who do not specify the argument.
* Users - post-deprecation. In 0.23.0, the ``legacy_interface`` argument will be removed.
  The methods will always produce returns with the future, post-change naming convention.
  This change may be breaking for users who do not remove the argument by 0.23.0.
* Appropriate deprecation warnings will be raised from 0.21.0 onwards, until 0.22.last.
* Users - recommended change actions. Users should aim to upgrade dependent code to ``legacy_interface=False`` behaviour by 0.21.last,
  and to remove ``legacy_interface`` arguments after 0.22.0 and before 0.23.0.
  Users who need more time to upgrade dependent code can set ``legacy_interface=True`` until 0.22.last.

Extenders should use the new ``"pred_int:legacy_interface:testcfg"`` config field to upgrade their third party extensions:

* Extenders - change period. The config field ``"pred_int:legacy_interface:testcfg"`` has been added
  to all descendants of the ``BaseForecaster`` class. This config controls the contract
  that the ``check_estimator`` and ``pytest`` tests check against, and can be set by ``set_config``.
* The default value of the tag is ``"auto"`` - this means that the tests will check against the current
  naming convention in the early phase (0.21.X), and against the future naming convention in the late phase (0.22.X),
  for ``_predict_quantiles`` or ``_predict_intervals`` having the standard signature, without ``legacy_interface``.
  From 0.23.0 on, the tag will have no effect.
* In the change period: if the tag is set to ``"new"``, the tests will always check against the new interface;
  if the tag is set to ``"old"``, the tests will check against the old interface, irrespective of the phase.
  From 0.23.0, the setting will have no effect and the tests will always check against the new interface.
* Extenders - recommended change actions: Extenders should aim to upgrade their third party extensions
  to ``"pred_int:legacy_interface:testcfg=new"`` behaviour by 0.21.last. Tests against late stage
  and post-deprecation behaviour can be enforced by setting ``forecaster.set_config({"pred_int:legacy_interface:testcfg": "new"})``,
  before passing it to ``check_estimator``.
  The ``set_config`` call can be removed after 0.22.0, and should be removed before 0.23.0, but will not be breaking if not removed.
* Extenders with a substantial user base of their own can, alternatively, implement and release ``_predict_quantiles`` and ``_predict_intervals``
  with a ``legacy_interface`` argument before 0.22.0, the default of which should be ``False`` from the beginning on (even in early phase).
  In this case, the ``"pred_int:legacy_interface:testcfg"`` tag should be set to ``"auto"``,
  and the tests will check both new and old behaviour. The ``legacy_interface`` argument can be removed after 0.23.0.
  This will result in the same transition experience for users of the extenders' forecasters as for users of ``sktime`` proper.


List of PR
~~~~~~~~~~

* [ENH] replace ``"Coverage"`` and ``"Quantiles"`` default variable name in univariate case with variable name (:pr:`4880`) :user:`fkiraly`, :user:`benheid`
* [BUG] 0.21.0 release bugfix - fix interaction of ``sklearn 1.3.0`` with dynamic error metic based on ``partial`` in ``test_make_scorer`` (:pr:`4915`) :user:`fkiraly`
* [MNT] xfail ``mlflow`` failure #4904 until debugged, gitignore for ``py-spy`` (:pr:`4913`) :user:`fkiraly`
* [DOC] 0.21.0 release action - update deprecation guide to reflect deprecation of use of `deprecated` (:pr:`4914`) :user:`fkiraly`
* [MNT] 0.21.0 release action - update ``sklearn`` bound to ``<1.4.0`` (:pr:`4778`) :user:`fkiraly`
* [MNT] 0.21.0 release action - add back ``pycatch22`` as a soft dependency post python 3.7 EOL (:pr:`4790`) :user:`fkiraly`

Version 0.20.1 - 2023-07-14
---------------------------

Highlights
~~~~~~~~~~

* data loader for Monash Forecasting Repository (:pr:`4826`) :user:`hazrulakmal`
* estimator crafter = string serialization/deserialization for all object/estimator blueprint specifications (:pr:`4738`) :user:`fkiraly`
* ``SkoptForecastingCV`` - hyperparameter tuning for forecasters using ``scikit-optimize`` (:pr:`4580`) :user:`hazrulakmal`
* new forecaster - ``statsmodels`` ``AutoReg`` interface (:pr:`4774`) :user:`CTFallon`, :user:`mgazian000`, :user:`JonathanBechtel`
* new forecaster - by-horizon ``FhPlexForecaster``, for different estimator/parameter per horizon (:pr:`4811`) :user:`fkiraly`
* new transformer - ``SplitterSummarizer`` to apply transformer by fold (:pr:`4759`) :user:`BensHamza`
* ``ColumnEnsembleTransformer`` - ``remainder`` argument (:pr:`4789`) :user:`fkiraly`
* new classifier and regressor - MCDCNN estimators migrated from ``sktime-dl`` (:pr:`4637`) :user:`achieveordie`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* object blueprint (specification) serialization/deserialization to string has been added.
  "blueprints" in this sense are object composites at init state, e.g., a pristine forecasting pipeline.
  All objects serialize by ``str`` coercion, e.g., ``str(my_pipeline)``, and deserialize
  via ``sktime.registry.craft : str -> object``. The deserializer ``craft`` is a pseudo-inverse
  of the serializer ``str`` for a fixed python environment, so can be used for fully reproducible
  specification storage and sharing, e.g., in reproducible science or performance benchmarking.
* further utilities ``registry.deps`` and ``registry.imports`` complement the serialization
  toolbox. In an environment with only core dependencies of ``sktime``, the utility
  ``deps : str -> list[str]`` produces a list of PEP 440 soft dependency specifiers
  required to craft the serialized object (e.g., a forecasting pipeline) which can be used
  to set up a python environment install before crafting. The utility ``imports : str -> str``
  produces a code block of all python compilable imports required to craft the serialized object.
* the tag ``python_dependencies_alias`` was added to manage estimator specific
  dependencies where the package name differs from the import name.
  See the estimator developer guide for further details.

Transformations
^^^^^^^^^^^^^^^

* the transformations base interface, i.e., estimators inheriting From
  ``BaseTransformer``, now allow ``X=None`` in ``transform`` without raising an
  exception.
  Individual transformers may now implement their own logic to deal with ``X=None``.

Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] estimator crafter aka deserialization of estimator spec from string (:pr:`4738`) :user:`fkiraly`
* [ENH] ``_HeterogenousMetaEstimator`` to accept list of tuples of any length (:pr:`4793`) :user:`fkiraly`
* [ENH] Improve handling of dependencies with alias (:pr:`4832`) :user:`hazrulakmal`
* [ENH] Add an explicit context manager during estimator dump (:pr:`4859`) :user:`achieveordie`, :user:`yarnabrina`

Benchmarking and Metrics
^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] refactored ``evaluate`` routine, use splitters internally and allow for separate ``X``-split (:pr:`4861`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [ENH] data loader for Monash Forecasting Repository (:pr:`4826`) :user:`hazrulakmal`

Forecasting
^^^^^^^^^^^

* [ENH] refactoring of ``ForecastingHorizon`` to use ``Index`` based ``cutoff`` in private methods (:pr:`4463`) :user:`fkiraly`
* [ENH] ``SkoptForecastingCV`` - hyperparameter tuning using ``scikit-optimize`` (:pr:`4580`) :user:`hazrulakmal`
* [ENH] add more contract tests for ``predict_interval``, ``predict_quantiles`` (:pr:`4763`) :user:`yarnabrina`
* [ENH] ``statsmodels`` ``AutoReg`` interface (:pr:`4774`) :user:`CTFallon`, :user:`mgazian000`, :user:`JonathanBechtel`
* [ENH] remove private defaults in forecasting module (:pr:`4810`) :user:`fkiraly`
* [ENH] by-horizon forecaster, for different estimator/parameter per horizon (:pr:`4811`) :user:`fkiraly`
* [ENH] splitter that replicates ``loc`` of another splitter (:pr:`4851`) :user:`fkiraly`
* [ENH] test-plus-train splitter compositor (:pr:`4862`) :user:`fkiraly`
* [ENH] set ``ForecastX`` missing data handling tag to ``True`` to properly cope with future unknown variables (:pr:`4876`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ensure ``BaggingClassifier`` can be used as univariate-to-multivariate compositor (:pr:`4788`) :user:`fkiraly`
* [ENH] migrate MCDCNN classifier, regressor, network from ``sktime-dl`` (:pr:`4637`) :user:`achieveordie`
* [ENH] in ``CNNNetwork``, add options to control ``padding`` and ``filter_size`` logic (:pr:`4784`) :user:`alan191006`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] migrate MCDCNN classifier, regressor, network from ``sktime-dl`` (:pr:`4637`) :user:`achieveordie`

Transformations
^^^^^^^^^^^^^^^

* [ENH] allow ``X=None`` in ``BaseTransformer.transform`` (:pr:`4112`) :user:`fkiraly`
* [ENH] Add ``hour_of_week`` option to ``DateTimeFeatures`` transformer (:pr:`4724`) :user:`VyomkeshVyas`
* [ENH] ``ColumnEnsembleTransformer`` - ``remainder`` argument (:pr:`4789`) :user:`fkiraly`
* [ENH] ``SplitterSummarizer`` transformer to apply transformer by fold (:pr:`4759`) :user:`BensHamza`

Visualisations
^^^^^^^^^^^^^^

* [ENH] remove assumption about column names from ``plot_series`` / ``plot_interval`` (:pr:`4779`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] Temporarily skip all DL Estimators (:pr:`4760`) :user:`achieveordie`
* [MNT] remove verbose flag on windows CI (:pr:`4761`) :user:`fkiraly`
* [MNT] address deprecation of ``sklearn`` ``if_delegate_has_method`` in 1.3 (:pr:`4764`) :user:`fkiraly`
* [MNT] bound ``tslearn<0.6.0`` due to bad dependency handling and massive imports (:pr:`4819`) :user:`fkiraly`
* [MNT] ensure CI for python 3.8-3.10 runs on ``pandas 2`` (:pr:`4795`) :user:`fkiraly`
* [MNT] also restrict ``tslearn`` on the ``pandas 2`` testing dependency set (:pr:`4825`) :user:`fkiraly`
* [MNT] clean-up of ``CODEOWNERS`` (:pr:`4782`) :user:`fkiraly`
* [MNT] skip failing ``test_transform_and_smooth_fp`` on ``main`` (:pr:`4836`) :user:`fkiraly`
* [MNT] unpin ``sphinx`` and plugins, with defensive upper bounds (:pr:`4823`) :user:`fkiraly`
* [MNT] Dependabot Setup (:pr:`4852`) :user:`yarnabrina`
* [MNT] update readthedocs env to python 3.11 and ubuntu 22.04 (:pr:`4821`) :user:`fkiraly`
* [MNT] [Dependabot](deps): Bump actions/download-artifact from 2 to 3 (:pr:`4854`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump styfle/cancel-workflow-action from 0.9.1 to 0.11.0 (:pr:`4855`) :user:`dependabot[bot]`
* [MNT] [Dependabot](deps): Bump actions/upload-artifact from 2 to 3 (:pr:`4856`) :user:`dependabot[bot]`
* [MNT] fix remaining ``sklearn 1.3.0`` compatibility issues (:pr:`4860`) :user:`fkiraly`, :user:`hazrulakmal`
* [MNT] remove forgotten ``deprecated`` import from 0.13.0 (:pr:`4824`) :user:`fkiraly`
* [MNT] Extend softdep error message tests support for packages with version specifier and alias (:pr:`4867`) :user:`hazrulakmal`, :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] Update Get Started docs, add regression vignette (:pr:`4216`) :user:`GargiChakraverty-yb`
* [DOC] adds a banner for non-latest branches in read-the-docs (:pr:`4681`) :user:`yarnabrina`
* [DOC] greatly simplified forecaster and transformer extension templates (:pr:`4729`) :user:`fkiraly`
* [DOC] Added examples to docstrings for K-Means and K-Medoids (:pr:`4736`) :user:`CTFallon`
* [DOC] Improvements to formulations in the README  (:pr:`4757`) :user:`mswat5`
* [DOC] testing guide: add ellipsis flag to doctest command (:pr:`4768`) :user:`mdsaad2305`
* [DOC] Examples added to docstrings for Time Series Forest Regressor and Dummy Regressor (:pr:`4775`) :user:`mgazian000`
* [DOC] add missing metrics to API reference (:pr:`4813`) :user:`fkiraly`
* [DOC] update date/year in LICENSE and readthedocs license constant (:pr:`4816`) :user:`fkiraly`, :user:`yarnabrina`
* [DOC] improved guide for soft dependencies (:pr:`4831`) :user:`fkiraly`
* [DOC] sort slightly disordered forecasting API reference (:pr:`4815`) :user:`fkiraly`
* [DOC] fix ``ColumnSelect`` typos in documentation (:pr:`4800`) :user:`fkiraly`
* [DOC] minor improvements to forecasting and transformer extension templates (:pr:`4828`) :user:`fkiraly`

Fixes
~~~~~

Benchmarking and Metrics
^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] allow unused parameters in metric when using ``make_forecasting_scorer`` (:pr:`4833`) :user:`fkiraly`
* [BUG] fix ``evaluate`` utility for case where ``y`` and ``X`` are not equal length (:pr:`4861`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] Add temporary fix to ``_BaseWindowForecaster`` to handle simultaneous in and out-of-sample forecasts (:pr:`4812`) :user:`felipeangelimvieira`
* [BUG] fix for ``make_reduction`` with unequal panels time index for ``global`` pooling (:pr:`4644`) :user:`kbpk`
* [BUG] allows probabilistic predictions in ``DynamicFactor`` in presence of exogenous variables by (:pr:`4758`) :user:`yarnabrina`
* [BUG] Fix ``predict_residuals`` internal data type conversion (:pr:`4772`) :user:`fkiraly`, :user:`benHeid`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix ``BoxCoxTransformer`` failure after ``scipy`` 1.11.0 (:pr:`4770`) :user:`fkiraly`
* [BUG] ``ColumnEnsembleTransformer`` - bugfixing broken logic (:pr:`4789`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [BUG] fix sporadic failures in ``utils.plotting`` tests - set the ``matplotlib``
  backend to ``agg`` to avoid that a GUI is triggered (:pr:`4781`) :user:`benHeid`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`alan191006`,
:user:`benHeid`,
:user:`BensHamza`,
:user:`CTFallon`,
:user:`felipeangelimvieira`,
:user:`fkiraly`,
:user:`GargiChakraverty-yb`,
:user:`hazrulakmal`,
:user:`JonathanBechtel`,
:user:`kbpk`,
:user:`mdsaad2305`,
:user:`mgazian000`,
:user:`mswat5`,
:user:`VyomkeshVyas`,
:user:`yarnabrina`


Version 0.20.0 - 2023-06-21
---------------------------

Maintenance release - python 3.7 end-of-life maintenance update,
scheduled deprecations.

For last non-maintenance content updates, see 0.19.2 and 0.19.1.

Contents
~~~~~~~~

* python 3.7 is no longer supported by ``sktime``, as python 3.7 end-of-life is
  imminent (June 27), with ``sktime`` dependencies already having dropped support.
* pre-commit and coding style upgrades (3.8 plus)
* scheduled 0.20.0 deprecation actions

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``numpy`` version bounds now allow versions ``1.25.X``

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.7 end-of-life
^^^^^^^^^^^^^^^^^^^^^^

``sktime`` no longer supports python 3.7 with ``sktime`` 0.20.0 and later.

python reaches end-of-life on Jun 27, 2023, and core dependencies of ``sktime``
have already dropped support for python 3.7 with their most recent versions
(e.g., ``scikit-learn``).

Time Series Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

``ComposableTimeSeriesClassifier`` and ``WeightedEnsembleClassifier``
have finished their move to ``classification.ensemble``, they are no longer
importable in their original locations.

List of PR
~~~~~~~~~~

* [MNT] 0.20.0 deprecation actions (:pr:`4733`) :user:`fkiraly`
* [MNT] 0.20.0 release action - remove python 3.7 support (:pr:`4717`) :user:`fkiraly`
* [MNT] 0.20.0 release action - increase ``scikit-base`` bound to ``<0.6.0`` (:pr:`4735`) :user:`fkiraly`
* [MNT] 0.20.0 release action - support for ``numpy 1.25`` (:pr:`4720`) :user:`jorenham`
* [MNT] 0.20.0 release action - remove initial utf comments in all python modules which are unnecessary in python 3 (:pr:`4725`) :user:`yarnabrina`
* [MNT] 0.20.0 release action - upgrade to coding style of python 3.8 and above using ``pyupgrade`` (:pr:`4726`) :user:`yarnabrina`

Contributors
~~~~~~~~~~~~

:user:`fkiraly`,
:user:`jorenham`,
:user:`yarnabrina`


Version 0.19.2 - 2023-06-19
---------------------------

Highlights
~~~~~~~~~~

* ``statsforecast`` ``AutoETS`` and ``AutoCES`` interfaces (:pr:`4648`, :pr:`4649`) :user:`yarnabrina`
* developer guide on remote setup of test suite (:pr:`4689`) :user:`fkiraly`
* update to all pre-commit hook versions, corresponding changes throughout the code base (:pr:`4680`) :user:`yarnabrina`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

* ``ForecastingHorizon`` and forecasters' ``fit``, ``predict`` now support ``range``
  as input. Caveat: ``range(n)`` starts at ``0`` and ends at ``n-1``.
  For an ``n``-step-ahead forecast, including all ``n`` integer steps in the horizon, pass ``range(1, n+1)``.

Enhancements
~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* [ENH] ``statsforecast`` ``AutoETS`` direct interface estimator (:pr:`4648`) :user:`yarnabrina`
* [ENH] ``statsforecast`` ``AutoCES`` direct interface estimator (:pr:`4649`) :user:`yarnabrina`
* [ENH] improved ``BaseForecaster`` exception messages, with reference to ``self`` class name (:pr:`4699`) :user:`fkiraly`
* [ENH] support passing horizons as ``range`` object in ``ForecastingHorizon`` and in ``fit`` and ``predict`` methods (:pr:`4716`) :user:`yarnabrina`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] migrate ``ResNetRegressor`` from ``sktime-dl`` (:pr:`4638`) :user:`achieveordie`

Documentation
~~~~~~~~~~~~~

* [DOC] correct accidental duplication of 0.19.0 changelog (:pr:`4662`) :user:`fkiraly`
* [DOC] developer guide on remote setup of test suite (:pr:`4689`) :user:`fkiraly`
* [DOC] User registration link on documentation landing page (:pr:`4675`) :user:`fkiraly`
* [DOC] correct some failing doctests (:pr:`4679`) :user:`mdsaad2305`

Maintenance
~~~~~~~~~~~

* [MNT] resolve pre-commit issues on ``main`` (:pr:`4673`) :user:`yarnabrina`
* [MNT] except some DL and ``numba`` based estimators from tests to prevent memory overload (:pr:`4682`) :user:`fkiraly`
* [MNT] remove private imports from ``sklearn`` - ``set_random_state`` (:pr:`4672`) :user:`fkiraly`
* [MNT] update pre-commit hook versions and corresponding changes (:pr:`4680`) :user:`yarnabrina`
* [MNT] add ``skbase`` to default package version display of ``show_versions`` (:pr:`4694`) :user:`fkiraly`
* [MNT] reduce CI test log verbosity (:pr:`4715`) :user:`fkiraly`
* [MNT] remove python 3.7 tests from CI (:pr:`4722`) :user:`fkiraly`

Fixes
~~~~~

BaseObject
^^^^^^^^^^

* [BUG] fix ``clone`` / ``set_params`` with nested ``sklearn`` objects (:pr:`4707`) :user:`fkiraly`, :user:`hazrulakmal`

Benchmarking
^^^^^^^^^^^^

* [BUG] bugfix for ``no-update_params`` strategy in ``evaluate`` (:pr:`4686`) :user:`hazrulakmal`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix dead source link for UEA datasets (:pr:`4705`) :user:`fkiraly`
* [BUG] remove ``IOError`` dataset from TSC data registry (:pr:`4711`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix conversion from ``pd-multiindex`` to ``df-list`` if not all index levels are present (:pr:`4693`) :user:`fkiraly`
* [BUG] Fix ``vectorize_est`` returning jumbled columns for column vectorization, ``pd.DataFrame`` return, if column names were not lexicographically ordered (:pr:`4684`) :user:`fkiraly`, :user:`hoesler`

Forecasting
^^^^^^^^^^^

* [BUG] correct ``ForecastX`` behaviour in case of multivariate ``y`` (:pr:`4719`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`fkiraly`,
:user:`hazrulakmal`,
:user:`hoesler`,
:user:`mdsaad2305`,
:user:`yarnabrina`


Version 0.19.1 - 2023-05-30
---------------------------

Maintenance release - scheduled ``pandas`` dependency updates, scheduled deprecations.

For last non-maintenance content update, see 0.18.1.

Contents
~~~~~~~~

* ``pandas 2`` is now fully supported.
  All ``sktime`` native functionality remains compatible with ``pandas 1``, ``>=1.1.0``.
* scheduled deprecation of ``tensorflow`` based probability interface.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``pandas`` version bounds now allow versions ``2.0.X`` in addition to
  currently supported ``pandas 1`` versions.
  This concludes the interim period for experimental support and
  begins full support for ``pandas 2``, with aim to support any ``pandas 2`` version.
* ``tensorflow-probability`` is no longer a dependency or soft dependency,
  it has also been removed from all dependency sets (including ``dl``)

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.7 end-of-life
^^^^^^^^^^^^^^^^^^^^^^

Python 3.7 reaches end-of-life on Jun 27, 2023, and core dependencies of ``sktime``
have already dropped support for python 3.7 with their most recent versions
(e.g., ``scikit-learn``).

``sktime`` will drop support for python 3.7 with 0.20.0, or the first minor release
after Jun 27, 2023, whichever is later.

Dependencies
^^^^^^^^^^^^

* ``tensorflow-probability`` is no longer a dependency or soft dependency,
  it has also been removed from all dependency sets (including ``dl``)

Forecasting
^^^^^^^^^^^

* The ``legacy_interface`` argument has been removed from
  forecasters' ``predict_proba``. The method now always returns a ``BaseDistribution``
  object, in line with prior default behaviour, i.e., ``legacy_interface=False``.

List of PR
~~~~~~~~~~

* [MNT] 0.19.0 change action - relax ``pandas`` bound to ``<2.1.0`` (:pr:`4429`) :user:`fkiraly`
* [MNT] 0.19.0 release action - tests for both ``pandas 1`` and ``pandas 2`` (:pr:`4622`) :user:`fkiraly`
* [MNT] 0.19.0 deprecations and changes (:pr:`4646`) :user:`fkiraly`


Version 0.19.0
--------------

Skipped for maintenance purposes, should not be used.
(yanked from pypi)


Version 0.18.1 - 2023-05-22
---------------------------

Highlights
~~~~~~~~~~

* ``sktime`` now has a generic adapter class to ``statsforecast`` (:pr:`4539`, :pr:`4629`) :user:`yarnabrina`
* ``statsforecast`` ``AutoTheta`` was added with direct interface using this, more to follow (:pr:`4539`) :user:`yarnabrina`
* the time series alignment module has been updated: extension template for aligners (:pr:`4613`),
  ``numba`` based alignment paths are availableas ``sktime`` aligners (:pr:`4620`) :user:`fkiraly`
* the forecasting benchmarking framework now allows to pass multiple metrics (:pr:`4586`) :user:`hazrulakmal`
* new time series classifiers: bagging, MACNN, RNN (:pr:`4185`, :pr:`4533`, :pr:`4636`) :user:`ArushikaBansal`, :user:`fkiraly`, :user:`achieveordie`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Probability distributions
^^^^^^^^^^^^^^^^^^^^^^^^^

* specification of default sample sizes for Monte Carlo approximations now
  use the ``scikit-base`` config system
* a ``quantile`` method was added, which returns a table of quantiles in
  the same format as ``BaseForecaster.predict_quantiles`` return quantile forecasts
* a ``ppf`` method was added for returning quantiles

Enhancements
~~~~~~~~~~~~

Benchmarking
^^^^^^^^^^^^

* [ENH] Clearer error message on fitting fail of ``evaluate`` (:pr:`4545`) :user:`fkiraly`
* [ENH] Extend forecasting benchmarking framework to multiple metrics, add test coverage (:pr:`4586`) :user:`hazrulakmal`

Forecasting
^^^^^^^^^^^

* [ENH] ``statsforecast`` ``AutoTheta`` direct interface estimator (:pr:`4539`) :user:`yarnabrina`
* [ENH] remove warning for length 1 forecasting pipelines (:pr:`4546`) :user:`fkiraly`
* [ENH] simple tabular prediction reduction for forecasting (:pr:`4564`) :user:`fkiraly`
* [ENH] rewrite of ``_StatsForecastAdapter`` in a generic way to support other models than ``AutoARIMA`` (:pr:`4629`) :user:`yarnabrina`

Probability distributions
^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] move probability distribution statistic approximation sample sizes to config interface (:pr:`4561`) :user:`fkiraly`
* [ENH] add more test parameter sets to ``AutoETS`` (:pr:`4588`) :user:`fkiraly`
* [ENH] improving ``BaseDistribution`` defaulting, and add test coverage (:pr:`4583`) :user:`fkiraly`

Time series alignment
^^^^^^^^^^^^^^^^^^^^^

* [ENH] full test suite for time series aligners (:pr:`4614`) :user:`fkiraly`
* [ENH] ``numba`` alignment paths as ``sktime`` aligners (:pr:`4620`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] `SimpleRNN` DL time series regressor, migrated from ``sktime-dl`` (:pr:`4185`) :user:`ArushikaBansal`
* [ENH] move classification ensembles to ``classification.ensembles`` (:pr:`4532`) :user:`fkiraly`
* [ENH] better documentation and test coverage for custom estimators and parameters in ``DrCIF`` (:pr:`4621`) :user:`Taise228`
* [ENH] Add MACNN classifier and network (:pr:`4636`) :user:`achieveordie`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] independent distance and multivariate aggregated kernel wrapper (:pr:`4598`) :user:`fkiraly`
* [ENH] variable subsetting dunder for distances and kernels (:pr:`4596`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] remove unnecessary conversion in ``TSFreshFeatureExtractor`` (:pr:`4571`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] replace ``joblib.hash`` with ``deep_equals`` in ``test_fit_does_not_overwrite_hyper_params`` for ``pandas`` based parameters (:pr:`4538`) :user:`fkiraly`
* [ENH] added ``msg`` argument in ``check_estimator`` (:pr:`4552`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] add silent dependencies to core dependency set (:pr:`4551`) :user:`fkiraly`
* [MNT] bound ``tensorflow-probability`` to ``<0.20.0`` (:pr:`4567`) :user:`fkiraly`
* [MNT] changing ``Union[int | float]`` to float as per issue #4379 (:pr:`4575`) :user:`mdsaad2305`
* [MNT] remove remaining soft dependency related module import warnings (:pr:`4554`) :user:`fkiraly`
* [MNT] ``pytest`` isolation in ``check_estimator`` (:pr:`4552`) :user:`fkiraly`
* [MNT] remove remaining soft dependency related module import warnings (:pr:`4554`) :user:`fkiraly`
* [MNT] temporary bound ``holidays`` to avoid error in ``Prophet``, later reverted (:pr:`4594`, :pr:`4600`) :user:`fkiraly`, :user:`yarnabrina`
* [MNT] remove ``tsfresh`` python version bounds from estimators (:pr:`4573`) :user:`fkiraly`
* [MNT] excepting ``FCNClassifier`` from CI to prevent memouts until bugfix (:pr:`4616`) :user:`fkiraly`
* [MNT] address ``kulsinski`` deprecation in ``scipy`` (:pr:`4618`) :user:`fkiraly`
* [MNT] remove forgotten ``legacy_interface`` reference from ``check_is_scitype`` docstring (:pr:`4630`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] fix typos in ``DynamicFactor`` docstrings (:pr:`4523`) :user:`kbpk`
* [DOC] improved docstrings in distances/kernels module (:pr:`4526`) :user:`fkiraly`
* [DOC] adds sktime internship link on the docs page (:pr:`4559`) :user:`fkiraly`
* [DOC] Improve Docstring for MAPE Metrics (:pr:`4563`) :user:`hazrulakmal`
* [DOC] Update link in minirocket.ipynb (:pr:`4577`) :user:`panozzaj`
* [DOC] additional glossary terms (:pr:`4556`) :user:`sanjayk0508`
* [DOC] fix warnings in make sphinx - language (``conf.py``) and ``dists_kernels.rst`` wrong imports (:pr:`4593`) :user:`mdsaad2305`
* [DOC] Add SVG version of the sktime logo (:pr:`4604`) :user:`marrov`
* [DOC] change logos to vector graphic ``png`` (:pr:`4605`) :user:`fkiraly`
* [DOC] change ``sktime`` logos to vector graphic ``svg`` (:pr:`4606`) :user:`fkiraly`
* [DOC] Remove white fill from ``svg`` and ``png`` ``sktime`` logos (:pr:`4607`) :user:`fkiraly`
* [DOC] ``AutoETS`` docstring - clarify conditional ignore of parameters dependent on ``auto`` (:pr:`4597`) :user:`luca-miniati`
* [DOC] correcting module path in ``dists_kernels.rst`` (:pr:`4625`) :user:`mdsaad2305`
* [DOC] Contributors update (:pr:`4609`) :user:`fkiraly`
* [DOC] updated ``PULL_REQUEST_TEMPLATE.md`` (:pr:`4599`) :user:`fkiraly`
* [DOC] docstring for ``SimpleRNNClassifier`` (:pr:`4572`) :user:`wasup-yash`
* [DOC] Contributors update (:pr:`4640`) :user:`fkiraly`
* [DOC] update to team/roles page (:pr:`4641`) :user:`fkiraly`
* [DOC] add examples for loading data from common tabular csv formats (:pr:`4612`) :user:`TonyZhangkz`
* [DOC] extension template for sequence aligners (:pr:`4613`) :user:`fkiraly`
* [DOC] fix minor issues in coding standards guide (:pr:`4619`) :user:`fkiraly`
* [DOC] remove forgotten ``legacy_interface`` reference from ``check_is_scitype`` docstring (:pr:`4630`) :user:`fkiraly`
* [DOC] adding doctest guide to the testing documentation (:pr:`4634`) :user:`mdsaad2305`

Fixes
~~~~~

BaseObject, BaseEstimator
^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix for ``get_fitted_params`` in ``_HeterogenousMetaEstimator`` (:pr:`4633`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] corrected default logic for ``_predict_interval`` in case ``_predict_quantiles`` is not implemented but ``_predict_proba`` is (:pr:`4529`) :user:`fkiraly`
* [BUG] ``RecursiveReductionForecaster`` pandas 2 fix (:pr:`4568`) :user:`fkiraly`
* [BUG] in ``_StatsModelsAdapter``, avoid passing ``exog`` to ``get_prediction`` of ``statsmodels`` in ``_predict_interval`` if parameter is not supported (:pr:`4589`) :user:`yarnabrina`
* [BUG] fix incorrect sorting of ``n_best_forecasters_`` in ``BaseGridCV`` if metric's ``lower_is_better`` is ``False`` (:pr:`4590`) :user:`hazrulakmal`

Probability distributions
^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix error messages in ``BaseDistribution`` if default methods are not implemented (:pr:`4628`) :user:`fkiraly`
* [BUG] fix wrong ``alpha`` sorting in ``BaseDistribution`` ``quantile`` return (:pr:`4631`) :user:`fkiraly`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix input check error message in ``BasePairwiseTransformerPanel``  (:pr:`4499`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^
* [BUG] fix broken RNN classifier (:pr:`4531`) :user:`achieveordie`
* [BUG] fix bug from clash between ``ABC`` inheritance and RNN ``fit`` override (:pr:`4527`) :user:`achieveordie` :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix broken RNN regressor (:pr:`4531`) :user:`achieveordie`
* [BUG] fix bug from clash between ``ABC`` inheritance and RNN ``fit`` override (:pr:`4527`) :user:`achieveordie` :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix informative error message on ``y`` input type check in ``BaseTransformer`` (:pr:`4525`) :user:`fkiraly`
* [BUG] fix incorrect values returned by ``DateTimeFeatures`` ``'month_of_quarter'`` feature (:pr:`4542`) :user:`fkiraly`
* [BUG] Fixes incorrect window indexing in ``HampelFilter`` (:pr:`4560`) :user:`antonioramos1`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`antonioramos1`,
:user:`ArushikaBansal`,
:user:`fkiraly`,
:user:`hazrulakmal`,
:user:`kbpk`,
:user:`luca-miniati`,
:user:`marrov`,
:user:`mdsaad2305`,
:user:`panozzaj`,
:user:`sanjayk0508`,
:user:`Taise228`,
:user:`TonyZhangkz`,
:user:`wasup-yash`,
:user:`yarnabrina`


Version 0.18.0 - 2023-04-28
---------------------------

Maintenance release - scheduled ``numba``, ``scikit-base``, ``pandas`` dependency updates,
scheduled deprecations.

For last non-maintenance content update, see 0.17.2.

Contents
~~~~~~~~

* ``numba`` has been changed to be a soft dependency. All ``numba`` based estimators
  continue working unchanged, but require explicit ``numba`` installation.
* the base module of ``sktime`` has been factored out to ``scikit-base``,
  the abstract base layer for ``scikit-learn`` like packages maintained by ``sktime``
* ``pandas 2`` support continues in testing/experimental period until 0.18.last.
  All ``sktime`` native functionality is ``pandas 2`` compatible, the transition period
  allows testing of deployments and custom extensions.
  See instructions below for upgrading dependent code to ``pandas 2``, or remaining on ``pandas 1``.
* scheduled deprecation of ``tensorflow`` based probability interface and ``VectorizedDF`` methods.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``numba`` is no longer a core dependency, it has changed to soft dependency
* ``scikit-base`` is a new core dependency

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies
^^^^^^^^^^^^

* ``numba`` has changed from core dependency to soft dependency in ``sktime 0.18.0``.
  To ensure functioning of setups of ``sktime`` code dependent on ``numba`` based estimators
  going forward, ensure to install ``numba`` in the environment explicitly,
  or install the ``all_extras`` soft dependency set which will continue to contain ``numba``.
  Besides this, ``numba`` dependent estimators will function identically as before.
* ``sktime``'s base module has moved to a new core dependency, ``scikit-base``, from ``sktime 0.18.0``.
  This will not impact functionality or imports directly from ``sktime``, or any usage.
* ``tensorflow-probability`` will cease to be a soft dependency from 0.19.0,
  as the only dependency locus (forecasters' old ``predict_proba`` return type)
  is being deprecated.

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``VectorizedDF.get_iloc_indexer`` was removed.
  Developers and users should use ``iter``, ``__iter__``, or ``get_iter_indices`` instead.

Forecasting
^^^^^^^^^^^

* forecasters' ``predict_proba`` now by default returns a ``BaseDistribution``.
  The old ``tensorflow-probability`` based return from pre-0.17.0 can still be obtained
  by setting the argument ``legacy_interface=False`` in ``predict_proba``.
  This is useful for handling deprecation.
* from 0.19.0, the ``legacy_interface`` argument will be removed from ``predict_proba``,
  together with the option to return ``tensorflow-probability`` based returns.

``pandas 2`` upgrade and testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* support for ``pandas 2`` is being introduced gradually:

  * experimental support period until 0.19.0 (all 0.17.X and 0.18.X versions)
  * full support from 0.19.0 (0.19.0, 0.19.X and onwards)

* in the experimental period (0.17.1-0.18.last):

  * ``sktime`` will have a dependency bound of ``pandas<2.0.0``
  * ``sktime`` will aim to be compatible with ``pandas 2.0.X`` as well as ``pandas 1, >=1.1.0``,
  * ``sktime`` can be run and tested with ``pandas 2.0.X`` by force-installing ``pandas 2.0.X``
  * estimators can be tested for ``pandas 2`` compatibility via ``check_estimator`` under force-installed ``pandas 2.0.X``
  * reports on compatibility issues are appreciated in :issue:`4426` (direct input or link from)

* in the full support period (0.19.0-onwards):

  * ``sktime`` requirements will allow ``pandas 2.0.X`` and extend support with ``pandas`` releases
  * ``sktime`` will aim to be compatible with ``pandas 2`` (any version), as well as ``pandas 1, >=1.1.0``
  * users choose their preferred ``pandas`` version by requirements on their downstream environment
  * the bug and issue trackers should be used as normal

List of PR
~~~~~~~~~~

* [MNT] 0.18.0 change action - ``numba`` as soft dependency (:pr:`3843`) :user:`fkiraly`
* [MNT] 0.18.0 deprecation actions (:pr:`4510`) :user:`fkiraly`
* [MNT] ensure ``predict_proba`` calls in ``mlflow`` forecasting interface explicitly call ``legacy_interface`` (:pr:`4514`) :user:`fkiraly`
* [MNT] ``skbase`` refactor - part 1: ``BaseObject`` and package dependencies (:pr:`3151`) :user:`fkiraly`
* [MNT] ``skbase`` refactor - part 2: ``all_estimators`` lookup (:pr:`3777`) :user:`fkiraly`
* [ENH] ``quantile`` method for distributions, default implementation of forecaster ``predict_quantiles`` if ``predict_proba`` is present (:pr:`4513`) :user:`fkiraly`
* [ENH] add test for ``all_estimators`` tag filter (:pr:`4512`) :user:`fkiraly`


Version 0.17.2 - 2023-04-24
---------------------------

Highlights
~~~~~~~~~~

* the transformers and pipelines tutorial from pydata global 2022 is now available in ``sktime``, see `examples <https://mybinder.org/v2/gh/sktime/sktime/main?filepath=examples>`_ (:pr:`4381`) :user:`dashapetr`
* probabilistic prediction functionality for ``SARIMAX`` (:pr:`4439`) :user:`yarnabrina`
* ``InceptionTime`` classifier from ``sktime-dl`` migrated (:pr:`3003`) :user:`tobiasweede`
* ``SplitterBootstrapTransformer`` for booststrapping based on any splitter (:pr:`4455`) :user:`fkiraly`
* ``IxToX`` transformer that creates features from time index or hierarchy label (:pr:`4416`) :user:`fkiraly`
* many bugfixes to probabilistic forecasting interfaces - ``BaggingForecaster``, ``BATS``, ``TBATS``, ``DynamicFactor``, ``VECM``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* all forecasters (``Baseforecaster`` descendants) now have the following new tags:

  * ``capability:insample``, boolean, indicating whether the classifier can make
    in-sample forecasts.
  * ``capability:pred_int:insample``, boolean, indicating whether the classifier can make
    probabilistic in-sample forecasts, e.g., prediction intervals in-sample.

* all forecasters are now tested for interface conformance for interval forecasts,
  in-sample (based on the above tags) and out-of-sample, via ``check_estimator``

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* all time series classifiers (``BaseClassifier`` descendants) now have a tag
  ``capability:predict_proba``. This indicates whether the classifier implements a
  non-default (non-delta-mass) probabilistic classification functionality.

Enhancements
~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] allow inclusive/exclusive bounds in ``get_slice`` (:pr:`4483`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] Adds ``_predict_interval`` to ``SARIMAX`` to support ``predict_interval`` and ``predict_quantiles`` (:pr:`4439`) :user:`yarnabrina`
* [ENH] shift ``ForecastingHorizon``-``BaseForecaster`` ``cutoff`` interface to rely on public point (:pr:`4456`) :user:`fkiraly`
* [ENH] testing in-sample forecasting - replace try/except in ``test_predict_time_index`` by tag and tag dependent contract (:pr:`4476`) :user:`fkiraly`
* [ENH] remove monotonicity requirement from quantile prediction contract (:pr:`4480`) :user:`fkiraly`
* [ENH] remove superfluous implementation checks in ``_predict_interval`` and ``_predict_quantiles`` of ``BaseForecaster`` (:pr:`4481`) :user:`yarnabrina`
* [ENH] seasonal tabulation utility (:pr:`4490`) :user:`fkiraly`, :user:`marrov`
* [ENH] testing all forecasters ``predict_quantiles``, ``predict_interval`` in-sample (:pr:`4470`) :user:`fkiraly`
* [ENH] performant re-implementation of ``NaiveForecaster`` - ``"last"`` strategy (:pr:`4461`) :user:`fkiraly`
* [ENH] adds ``_predict_interval`` in ``_StatsModelsAdapter`` and inherits in other estimator to reduce code duplication (:pr:`4465`) :user:`yarnabrina`
* [ENH] in ``ForecastingHorizon``, refactor ``to_absolute().to_pandas()`` calls to a method (:pr:`4464`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``predict_proba`` capability tag for classifiers (:pr:`4012`) :user:`fkiraly`
* [ENH] migrate ``InceptionTime`` classifier and example (from ``sktime-dl``) (:pr:`3003`) :user:`tobiasweede`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

Transformations
^^^^^^^^^^^^^^^

* [ENH] ``IxToX`` transformer that creates features from time index or hierarchy label (:pr:`4416`) :user:`fkiraly`
* [ENH] ``SplitterBootstrapTransformer`` for booststrapping based on any splitter (:pr:`4455`) :user:`fkiraly`
* [ENH] transformer compositor to apply by panel or instance (:pr:`4477`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] improved ``_make_series`` utility and docstring (:pr:`4487`) :user:`fkiraly`
* [ENH] remove calls to ``return_numpy`` arg in ``_make_series`` (:pr:`4488`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] Changed line endings of ``ElectricDevices.csv`` and ``GunPoint.csv`` from ``CRLF`` to ``LF`` (:pr:`4452`) :user:`yarnabrina`
* [MNT] ensure all elements in test matrix complete runs (:pr:`4472`) :user:`fkiraly`
* [MNT] add ``InceptionTimeClassifier`` and ``LSTMFCNClassifier`` as direct module export (:pr:`4484`) :user:`fkiraly`
* [MNT] address some warnings and deprecation messages from dependencies (:pr:`4486`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] Fix error in ``MiniRocket`` example code - wrong transformer (:pr:`4497`) :user:`doncarlos999`
* [DOC] add ``InceptionTimeClassifier`` and ``LSTMFCNClassifier`` to API docs (:pr:`4484`) :user:`fkiraly`
* [DOC] fix typo in cython interface reference, ``MySQM`` -> ``MrSQM`` (:pr:`4493`) :user:`fkiraly`
* [DOC] move content from pydata global 2022 (transformers, pipelines tutorial) to sktime main repo (:pr:`4381`) :user:`dashapetr`
* [DOC] improvements to description of ``sktime`` on the readthedocs landing page (:pr:`4444`) :user:`howdy07`

Fixes
~~~~~

Forecasting
^^^^^^^^^^^

* [BUG] fix ``pandas`` write error in probabilistic forecasts of ``BaggingForecaster`` (:pr:`4478`) :user:`fkiraly`
* [BUG] fix ``predict_quantiles`` in ``_PmdArimaAdapter`` and ``_StatsForecastAdapter`` post 0.17.1 (:pr:`4469`) :user:`fkiraly`
* [BUG] ``ForecastingHorizon`` constructor - override erroneously inferred ``freq`` attribute from regular ``DatetimeIndex`` based horizon (:pr:`4466`) :user:`fkiraly`, :user:`yarnabrina`
* [BUG] fix broken ``DynamicFactor._predict_interval`` (:pr:`4479`) :user:`fkiraly`
* [BUG] fix ``pmdarima`` interfaces breaking for ``X`` containing more indices than forecasting horizon (:pr:`3667`) :user:`fkiraly`, :user:`SzymonStolarski`
* [BUG] fix ``BATS`` and ``TBATS`` ``_predict_interval`` interface (:pr:`4492`, :pr:`4505`) :user:`fkiraly``
* [BUG] fix ``VECM._predict_interval`` interface for date-like indices (:pr:`4506`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [BUG] fix index error in nullable input test (:pr:`4474`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`dashapetr`,
:user:`doncarlos999`,
:user:`fkiraly`,
:user:`howdy07`,
:user:`marrov`,
:user:`SzymonStolarski`,
:user:`tobiasweede`,
:user:`yarnabrina`


Version 0.17.1 - 2023-04-10
---------------------------

Maintenance patch (``pandas 2``, ``attrs``). For last content update, see 0.17.0.

* ``pandas 2`` compatibility patch
* experimental support for ``pandas 2`` with testing and upgrade instructions for users
* ``sktime`` will continue to support ``pandas 1`` versions

User feedback and ``pandas 2`` compatibility issues are appreciated in :issue:`4426`.

Dependency changes
~~~~~~~~~~~~~~~~~~

* the version bound ``pandas<2.0.0`` will be relaxed to ``pandas<2.1.0`` in ``sktime 0.19.0``

  * option 1: to keep using ``pandas 1.X`` from ``0.19.0`` onwards, simply introduce the ``pandas<2.0.0`` bound in downstream requirements
  * option 2: to upgrade safely to ``pandas 2.X``, follow the upgrade and testing instructions below
  * neither option impacts public interfaces of ``sktime``, i.e., there are no removals, deprecations,
    or changes of contract besides the change of ``pandas`` bound in ``sktime`` requirements

* ``attrs`` changes from an implied (non-explicit) soft dependency to an explicit soft dependency (in ``all_extras``)

``pandas 2`` upgrade and testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* support for ``pandas 2`` will be introduced gradually:

  * experimental support period until 0.19.0 (all 0.17.X and 0.18.X versions)
  * full support from 0.19.0 (0.19.0, 0.19.X and onwards)

* in the experimental period (0.17.1-0.18.last):

  * ``sktime`` will have a dependency bound of ``pandas<2.0.0``
  * ``sktime`` will aim to be compatible with ``pandas 2.0.X`` as well as ``pandas 1, >=1.1.0``,
  * ``sktime`` can be run and tested with ``pandas 2.0.X`` by force-installing ``pandas 2.0.X``
  * estimators can be tested for ``pandas 2`` compatibility via ``check_estimator`` under force-installed ``pandas 2.0.X``
  * reports on compatibility issues are appreciated in :issue:`4426` (direct input or link from)

* in the full support period (0.19.0-onwards):

  * ``sktime`` requirements will allow ``pandas 2.0.X`` and extend support with ``pandas`` releases
  * ``sktime`` will aim to be compatible with ``pandas 2`` (any version), as well as ``pandas 1, >=1.1.0``
  * users choose their preferred ``pandas`` version by requirements on their downstream environment
  * the bug and issue trackers should be used as normal

List of PR
~~~~~~~~~~

* [MNT] address deprecation of ``"mad"`` option on ``DataFrame.agg`` and ``Series.agg`` (:pr:`4435`) :user:`fkiraly`
* [MNT] address deprecation of automatic drop on ``DataFrame.agg`` on non-numeric columns (:pr:`4436`) :user:`fkiraly`
* [MNT] resolve ``freq`` related deprecations and ``pandas 2`` failures in reducers (:pr:`4438`) :user:`fkiraly`
* [MNT] except ``Prophet`` from ``test_predict_quantiles`` due to sporadic failures (:pr:`4432`) :user:`fkiraly`
* [MNT] except ``VECM`` from ``test_predict_quantiles`` due to sporadic failures (:pr:`4442`) :user:`fkiraly`
* [MNT] fix and sharpen soft dependency isolation logic for ``statsmodels`` and ``pmdarima`` (:pr:`4443`) :user:`fkiraly`
* [MNT] isolating ``attrs`` imports (:pr:`4450`) :user:`fkiraly`

Version 0.17.0 - 2023-04-03
---------------------------

Highlights
~~~~~~~~~~

* Full support for python 3.11
* reworked probabilistic forecasting & new metrics (``LogLoss``, ``CRPS``), integration with tuning (:pr:`4190`, :pr:`4276`, :pr:`4290`, :pr:`4367`) :user:`fkiraly`
* conditional transformer ``TransformIf``, e.g., deseasonalize after seasonality test (:pr:`4248`) :user:`fkiraly`
* new transformer interfaces: Christiano-Fitzgerald and Hodrick-Prescott filter (``statsmodels``), Fourier transform (:pr:`4342`, :pr:`4402`) :user:`ken-maeda`, :user:`blazingbhavneek`
* new forecaster: ``ForecastKnownValues`` forn known or expert forecasts (:pr:`4243`) :user:`fkiraly`
* new classifier: MrSQM (:pr:`4337`) :user:`fkiraly`, :user:`lnthach`, :user:`heerme`

Dependency changes
~~~~~~~~~~~~~~~~~~

* a new soft dependency was added, the ``seasonal`` package,
  required (only) for the ``SeasonalityPeriodogram`` estimator.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject, BaseEstimator
^^^^^^^^^^^^^^^^^^^^^^^^^

* all ``sktime`` objects and estimators now possess a config interface, via new
  ``get_config`` and ``set_config`` methods. This is currently experimental,
  and there are no externally facing config fields at the moment.

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``sktime`` now recognizes nullable ``pandas`` ``dtypes`` and coerces them to
  non-nullable if necessary. Previously, nullable ``dtype`` would cause exceptions.

Forecasting
^^^^^^^^^^^

* the ``BaseDistribution`` object has been introduced as a potential return
  of full distribution forecasts and simulation queries.
  This is currently experimental, feedback and contributions are appreciated.
* Forecasters' ``predict_proba`` now returns an ``sktime`` ``BaseDistribution`` object,
  if ``tensorflow-probability`` is not present (e.g., on python 3.11), or if the
  temporary deprecation argument ``legacy_interface=False`` is set.
  The old ``tensorflow`` based interfaced will be deprecated over two cycles, see below.
* ``sktime`` now contains metrics and losses for probability distribution forecasts.
  These metrics assume ``BaseDistribution`` objects as forecasts.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies
^^^^^^^^^^^^

* ``numba`` will change from core dependency to soft dependency in ``sktime 0.18.0``.
  To ensure functioning of setups of ``sktime`` code dependent on ``numba`` based estimators
  going forward, ensure to install ``numba`` in the environment explicitly,
  or install the ``all_extras`` soft dependency set which will continue to contain ``numba``.
  Besides this, ``numba`` dependent estimators will function identically as before.
* ``sktime``'s base module will move to a new core dependency, ``skbase``, from ``sktime 0.18.0``.
  This will not impact functionality or imports directly from ``sktime``, or any usage.

Forecasting
^^^^^^^^^^^

* Forecasters' ``predict_proba`` pre-0.17.0 ``tensorflow`` based return will
  be replaced by ``BaseDistribution`` object based return.
  This will be phased out in two minor cycles as follows.
* until 0.18.0, forecasters' ``predict_proba`` will return ``BaseDistribution``
  by default only in cases where calling ``predict_proba`` would have raised an error,
  prior to 0.17.0, i.e., on python 3.11 and when ``tensorflow-probability``
  is not present in the python environment.
* until 0.18.0, ``BaseDistribution`` return can be enforced by setting the new argument
  ``legacy_interface=False`` in ``predict_proba``. This is useful for handling deprecation.
* from 0.18.0, the default for ``legacy_interface`` will be set to ``False``.
* from 0.19.0, the ``legacy_interface`` argument will be removed from ``predict_proba``,
  together with the option to return ``tensorflow-probability`` based returns.

Transformations
^^^^^^^^^^^^^^^

* ``DateTimeFeatures``: the default value of the ``keep_original_columns``
  parameter has changed to ``False``
* ``FourierFeatures``: the default value of the ``keep_original_columns``
  parameter has changed to ``False``

Testing framework
^^^^^^^^^^^^^^^^^

* in ``check_estimator`` and ``run_tests``, the ``return_exceptions`` argument has been
  removed. It is now fully replaced by ``raise_exceptions`` (its logical negation),
  which has been available since 0.16.0.

Enhancements
~~~~~~~~~~~~

BaseObject, BaseEstimator
^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] tag manager mixin (:pr:`3630`) :user:`fkiraly`
* [ENH] Estimator config interface (:pr:`3822`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] nullable dtypes - ensure nullable columns are coerced to ``float`` dtype in ``pandas`` conversions (:pr:`4245`) :user:`fkiraly`
* [ENH] ``is_equal_index`` metadata element in checks and examples (:pr:`4312`) :user:`fkiraly`
* [ENH] granular control of mtype metadata computation, avoid computation when not needed (:pr:`4389`) :user:`fkiraly`, :user:`hoesler`
* [ENH] turn off all unnecessary input checks in current base class boilerplate (:pr:`4390`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] factor out column ensemble functionality from ``_ColumnEnsembleForecaster`` to new base mixin (:pr:`4231`) :user:`fkiraly`
* [ENH] ``ForecastKnownValues`` forecaster that forecasts prescribed known or expert forecast values (:pr:`4243`) :user:`fkiraly`
* [ENH] Improve vectorized metric calculation, deprecate ``VectorizedDF.get_iloc_indexer`` (:pr:`4228`) :user:`hoesler`
* [ENH] ``MeanAbsoluteError`` - ``evaluate_by_index`` (:pr:`4302`) :user:`fkiraly`
* [ENH] ``BaseForecastingErrorMetric`` internal interface cleanup (:pr:`4305`) :user:`fkiraly`
* [ENH] probabilistic forecasting rework part 1 - backend agnostic probability distributions (:pr:`4190`) :user:`fkiraly`
* [ENH] probabilistic forecasting rework part 2 - distribution forecast metrics log-loss, CRPS (:pr:`4276`) :user:`fkiraly`
* [ENH] probabilistic forecasting rework part 3 - forecasters (:pr:`4290`) :user:`fkiraly`
* [ENH] probabilistic forecasting rework part 4 - evaluation and tuning (:pr:`4367`) :user:`fkiraly`
* [ENH] informative error messages for forecasting pipeline constructors, for ``steps`` arg (:pr:`4371`) :user:`fkiraly`

Parameter estimators
^^^^^^^^^^^^^^^^^^^^

* [ENH] interface the seasonal package as a parameter estimator for seasonality (:pr:`4215`) :user:`blazingbhavneek`
* [ENH] parameter estimators for stationarity - ADF and KPSS (:pr:`4247`) :user:`fkiraly`
* [ENH] ``PluginParamsForecaster`` to accept any estimator, conformal tuned fast example (:pr:`4412`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] MrSQM classifier - direct interface (:pr:`4337`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] transformer interfacing ``numpy.fft`` for simple fourier transform (:pr:`4214`) :user:`blazingbhavneek`
* [ENH] ``sktime`` native column ensemble transformer (:pr:`4232`) :user:`fkiraly`
* [ENH] conditional transform (:pr:`4248`) :user:`fkiraly`
* [ENH] kinematic feature transformer (:pr:`4261`) :user:`fkiraly`
* [ENH] refactoring segmentation transformers to use ``pandas`` native data types (:pr:`4267`) :user:`fkiraly`
* [ENH] remove test for output values in ``test_FeatureUnion_pipeline`` (:pr:`4316`) :user:`fkiraly`
* [ENH] Hodrick-Prescott filter transformer (``statsmodels`` interface) (:pr:`4342`) :user:`ken-maeda`
* [ENH] turn ``BKFilter`` into a direct ``statsmodels`` interface (:pr:`4346`) :user:`fkiraly`
* [ENH] Christiano-Fitzgerald filter transformer (``statsmodels`` interface) (:pr:`4402`) :user:`ken-maeda`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] additional test parameter sets for performance metrics (:pr:`4246`) :user:`fkiraly`
* [ENH] test for ``get_test_params``, and reserved parameters (:pr:`4279`) :user:`fkiraly`
* [ENH] cleaned up probabilistic forecasting tests for quantile and interval predictions (:pr:`4393`) :user:`fkiraly`, :user:`yarnabrina`
* [ENH] cover list input case for ``test_predict_interval`` ``coverage`` and ``test_predict_quantiles`` ``alpha`` in forecaster contract tests (:pr:`4394`) :user:`yarnabrina`

Maintenance
~~~~~~~~~~~

* [MNT] address deprecation of ``pandas.DataFrame.iteritems`` (:pr:`4271`) :user:`fkiraly`
* [MNT] Fixes linting issue ``B016 Cannot raise a literal`` in ``distances`` module (:pr:`4284`) :user:`SamiAlavi`
* [MNT] add soft dependencies on python 3.11 that are 3.11 compatible (:pr:`4269`) :user:`fkiraly``
* [MNT] integrate parameter estimators with ``check_estimator`` (:pr:`4287`) :user:`fkiraly`
* [MNT] addressing ``pytest`` failure - downgrade ``dash`` to <2.9.0 (:pr:`4353`) :user:`fkiraly`
* [MNT] resolve circular imports in ``forecasting.base`` (:pr:`4329`) :user:`fkiraly`
* [MNT] isolating ``scipy`` imports, part 1 (:pr:`4005`) :user:`fkiraly`
* [MNT] Remove restrictions on branch for workflow that autodetect and updates ``CONTRIBUTORS.md`` (:pr:`4323`) :user:`achieveordie`
* [MNT] carry out forgotten deprecation for ``ContractableBOSS`` ``typed_dict`` parameter (:pr:`4331`) :user:`fkiraly`
* [MNT] except forecasters failing proba prediction tests (previously masked by buggy tests) (:pr:`4364`) :user:`fkiraly`
* [MNT] split up ``transformations.compose`` into submodules (:pr:`4368`) :user:`fkiraly`
* [MNT] replace emergency ``dash`` bound by exclusion of failing version 2.9.0 (:pr:`4415`) :user:`fkiraly`
* [MNT] remove soft dependency import warnings in modules and documented requirements to add these (:pr:`4398`) :user:`fkiraly`
* [MNT] dockerized tests (:pr:`4285`) :user:`fkiraly`, :user:`lmmentel`
* [MNT] Fix linting issues in transformations module (:pr:`4291`) :user:`SamiAlavi`
* [MNT] Fixes linting issues in ``base``, ``networks``, ``registry`` modules (:pr:`4310`) :user:`SamiAlavi`
* [MNT] resolve circular imports in ``forecasting.base`` (:pr:`4329`) :user:`fkiraly`
* [MNT] Linting ``test_croston.py``  (:pr:`4334`) :user:`ShivamPathak99`
* [MNT] except forecasters failing proba prediction tests (previously masked by buggy tests) (:pr:`4364`) :user:`fkiraly`
* [MNT] Auto-fixing linting issues (:pr:`4317`) :user:`SamiAlavi`
* [MNT] Fix linting issues in ``clustering`` module (:pr:`4318`) :user:`SamiAlavi`
* [MNT] Fix linting issues in ``forecasting`` module (:pr:`4319`) :user:`SamiAlavi`
* [MNT] Fixes linting issues in ``annotation`` module (:pr:`4309`) :user:`SamiAlavi`
* [MNT] Fix linting issues in ``series_as_features``, ``tests``, ``dist_kernels``, ``benchmarking`` modules (:pr:`4321`) :user:`SamiAlavi`
* [MNT] Fixes linting issues in ``classification`` module (:pr:`4311`) :user:`SamiAlavi`
* [MNT] Fix linting issues in ``performance_metrics`` module (:pr:`4320`) :user:`SamiAlavi`
* [MNT] Fix linting issues in ``utils`` module (:pr:`4322`) :user:`SamiAlavi`
* [MNT] replace author names by GitHub ID in author fields (:pr:`4340`) :user:`SamiAlavi`
* [MNT] address deprecation warnings from dependencies (:pr:`4423`) :user:`fkiraly`
* [MNT] 0.17.0 deprecation & change actions (:pr:`4424`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] direct documentation links to ``sktime.net`` addresses (:pr:`4241`) :user:`fkiraly`
* [DOC] improved reducer docstring formatting (:pr:`4160`) :user:`fkiraly`
* [DOC] improve docstring for ``VectorizedDF.items`` and ``.__iter__`` (:pr:`4223`) :user:`fkiraly`
* [DOC] direct documentation links to ``sktime.net`` addresses (:pr:`4241`) :user:`fkiraly`
* [DOC] update transformer extension template docstrings, reference to ``Hierarchical`` (:pr:`4250`) :user:`fkiraly`
* [DOC] API reference for parameter estimator module (:pr:`4244`) :user:`fkiraly`
* [DOC] add missing docstrings in ``PlateauFinder`` module (:pr:`4255`) :user:`ShivamPathak99`
* [DOC] docstring improvements for ``ColumnConcatenator`` (:pr:`4272`) :user:`JonathanBechtel`
* [DOC] Small docstring fixes in reducer module and tests (:pr:`4274`) :user:`danbartl`
* [DOC] clarified requirement for ``get_test_params`` in extension templates (:pr:`4289`) :user:`fkiraly`
* [DOC] developer guide for local testing (:pr:`4285`) :user:`fkiraly`
* [DOC] extension template for parameter estimators (:pr:`4288`) :user:`fkiraly`
* [DOC] refresh discord invite to new server (:pr:`4297`) :user:`fkiraly`
* [DOC] Update ``CONTRIBUTORS.md`` to most recent (:pr:`4358`) :user:`fkiraly`
* [DOC] improved method docstrings for transformers (:pr:`4328`) :user:`fkiraly`
* [DOC] ``MeanAbsoluteError`` docstring (:pr:`4302`) :user:`fkiraly`
* [DOC] updated ``dtw_distance`` docstring example to include import (:pr:`4324`) :user:`JonathanBechtel`
* [DOC] fix typo: Transforemd → Transformed (:pr:`4366`) :user:`kgeis`
* [DOC] ``TimeSeriesKMeans`` - correct ``init_algorithm`` default in docstring  (:pr:`4347`) :user:`marcosousapoza`
* [DOC] add missing import statements in numba distance docstrings (:pr:`4376`) :user:`JonathanBechtel`
* [DOC] guide for adding cython based estimators (:pr:`4388`) :user:`fkiraly`
* [DOC] add docstring example for ``ForecastX`` forecasting only some exogeneous variables (:pr:`4392`) :user:`fkiraly`
* [DOC] improvements to "invitation to contribute" paragraph in documentation (:pr:`4395`) :user:`abhisek7154`
* [DOC] README and docs update - tasks table, typos, lookup (:pr:`4414`) :user:`fkiraly`

Fixes
~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix level name handling in conversions ``nested_univ`` / ``pd-multiindex`` (:pr:`4270`) :user:`fkiraly`
* [BUG] fix incorrect inference of ``is_equally_spaced`` for unequal index ``pd-multiindex`` typed data (:pr:`4308`) :user:`noahleegithub`

Forecasting
^^^^^^^^^^^

* [BUG] fix ``Settingwithcopywarning`` when using custom error metric in ``evaluate`` (:pr:`4294`) :user:`fkiraly`, :user:`marrov`
* [BUG] fix forecasting metrics' ``evaluate_by_index`` for hierarchical input (:pr:`4306`) :user:`fkiraly`, :user:`marrov`
* [BUG] pass user passed parameters to ``ForecastX`` to underlying estimators (:pr:`4391`) :user:`yarnabrina`
* [BUG] fix unreported probabilistic prediction bugs detected through #4393 (:pr:`4399`) :user:`fkiraly`
* [BUG] ensure forecaster ``cutoff`` has ``freq`` inferred if inferable, for single series (:pr:`4406`) :user:`fkiraly`
* [BUG] fix ``ValueError`` in ``VECM._predict_interval`` if multiple coverage values were passed (:pr:`4411`) :user:`yarnabrina`
* [BUG] temporarily skip ``test_predict_quantiles`` for ``VAR`` due to known sporadic bug #4420 (:pr:`4425`) :user:`yarnabrina`

Parameter estimators
^^^^^^^^^^^^^^^^^^^^

* [BUG] fix seasonality estimators for ``candidate_sp`` being ``int`` (:pr:`4360`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``WeightedEnsembleClassifier._predict_proba`` to work with ``pandas`` based mtypes (:pr:`4275`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix broken ``ComposableTimeSeriesRegressor`` (:pr:`4221`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [BUG] in forecasting test framework, fix ineffective assertion for correct time index check (:pr:`4361`) :user:`fkiraly`
* [BUG] Fix ``MockForecaster._predict_quantiles`` to ensure monotonicity of quantiles (:pr:`4397`) :user:`yarnabrina`
* [BUG] prevent discovery of abstract ``TimeSeriesLloyds`` by contract tests (:pr:`4225`) :user:`fkiraly`

Utilities
^^^^^^^^^

* [BUG] fix ``show_versions`` and add tests (:pr:`4421`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`abhisek7154`,
:user:`achieveordie`,
:user:`blazingbhavneek`,
:user:`danbartl`,
:user:`fkiraly`,
:user:`hoesler`,
:user:`JonathanBechtel`,
:user:`ken-maeda`,
:user:`kgeis`,
:user:`lmmentel`,
:user:`marcosousapoza`,
:user:`marrov`,
:user:`noahleegithub`,
:user:`SamiAlavi`,
:user:`ShivamPathak99`,
:user:`yarnabrina`


Version 0.16.1 - 2023-02-13
---------------------------

Highlights
~~~~~~~~~~

* Experimental python 3.11 support. Full python 3.11 support is planned with 0.17.0. (:pr:`4000`, :pr:`3631`, :pr:`4226`) :user:`fkiraly`
* Experimental benchmarking module based on ``kotsu``, forecasting sub-module (:pr:`2977`) :user:`alex-hh`, :user:`dbcerigo`
* substantial speed-ups for panel and hierarchical transformers and forecasters (:pr:`4193`, :pr:`4194`, :pr:`4195`, :pr:`4196`) :user:`hoesler`

Testing and feedback of python 3.11 support and the benchmarking module are appreciated.

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``sktime`` now supports python 3.11
* on python 3.11, ``numba`` is not a dependency, and a number of other packages are
  also not available as soft dependencies, mostly due to compatibility with 3.11.
* ``sktime`` and its test suite can now be used without ``numba`` installed,
  with the exception of estimators depending on ``numba``.
  ``numba`` is still a core dependency on python 3.7-3.10.
* ``numba`` will become a soft dependency, from a core dependency, in 0.18.0.
  Estimators dependent on ``numba`` will function exactly as before if ``numba``
  is present in the python environment.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Benchmarking
^^^^^^^^^^^^

* the ``kotsu``-based benchmarking module introduces a new design and syntax
  for benchmarking forecasters.

Forecasting
^^^^^^^^^^^

* forecasters will now consistently preserve the ``name`` attribute in ``pd.Series`` passed.
  Previously, named ``pd.Series`` were not fully supported.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies
^^^^^^^^^^^^

* ``numba`` will change from core dependency to soft dependency in ``sktime 0.18.0``.
  To ensure functioning of setups of ``sktime`` code dependent on ``numba`` based estimators
  going forward, ensure to install ``numba`` in the environment explicitly,
  or install the ``all_extras`` soft dependency set which will continue to contain ``numba``.
  Besides this, ``numba`` dependent estimators will function identically as before.

Enhancements
~~~~~~~~~~~~

Benchmarking
^^^^^^^^^^^^

* [ENH] Benchmarking interface v2 based on ``kotsu`` package (:pr:`2977`) :user:`alex-hh`, :user:`dbcerigo`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Improve vectorization performance (:pr:`4195`) :user:`hoesler`
* [ENH] Improve panel mtype check performance (:pr:`4196`) :user:`hoesler`, :user:`danbartl`

Forecasting
^^^^^^^^^^^

* [ENH] fixes for forecasters to retain ``name`` attribute in ``predict`` (:pr:`4161`) :user:`fkiraly`
* [ENH] improved/fixed ``scoring`` argument for forecasting tuners (:pr:`4178`) :user:`fkiraly`
* [ENH] test ``Prophet`` with ``pd.DatetimeIndex`` (:pr:`4183`) :user:`fkiraly`
* [ENH] faster test for forecasters' ``predict_residuals`` (:pr:`4156`) :user:`fkiraly`
* [ENH] test that forecasters preserve ``name`` attr of ``pd.Series`` (:pr:`4157`) :user:`fkiraly`
* [ENH] improved/fixed ``scoring`` argument for forecasting tuners (:pr:`4178`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] add native multi-index/hierarchical data support to ``Imputer`` (:pr:`4194`) :user:`hoesler`
* [ENH] Add panel support to ``ColSelect`` transformer (:pr:`4193`) :user:`hoesler`

Fixes
~~~~~

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Correct ``'StarlightCurves'`` data set identifier string, to 'StarLightCurves' (:pr:`4222`) :user:`NeuralNut`
* [BUG] fix race condition in ``tsfile`` tests (:pr:`4192`) :user:`hoesler`

Forecasting
^^^^^^^^^^^

* [BUG] fixes for forecasters to retain ``name`` attribute in ``predict`` (:pr:`4161`) :user:`fkiraly`
* [BUG] ensure ``pd.Series`` ``name`` attribute is preserved in conversion to/from ``pd.DataFrame`` and ``np.ndarray``, as ``Series`` scitype (:pr:`4150`) :user:`fkiraly`
* [BUG] ``AutoETS``, ``UnobservedComponents``: fix ``predict_interval`` for integer based index not starting at zero (:pr:`4180`) :user:`fkiraly`

Parameter estimation
^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``nlag`` logic in ``SeasonalityACF`` and ``SeasonalityACFqstat`` (:pr:`4171`) :user:`fkiraly`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``TimeSeriesDBSCAN`` and remove strict ``BaseClusterer`` abstracts (:pr:`4227`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] Fix merge conflicts and formatting in ``.all-contributorsrc`` (:pr:`4205`) :user:`fkiraly`
* [MNT] isolate ``numba`` - NB: does not make ``numba`` a soft dependency (:pr:`3631`) :user:`fkiraly`
* [MNT] isolate remaining ``numba`` references (:pr:`4226`) :user:`fkiraly`
* [MNT] python 3.11 compatibility, with ``numba`` as core dependency on 3.7-3.10 (:pr:`4000`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] Fix rendering of examples section in ``Lag`` docstring (:pr:`3960`) :user:`aiwalter`
* [DOC] improved docstring for ``dtw_distance`` (:pr:`4028`) :user:`fkiraly`, :user:`matthewmiddlehurst`
* [DOC] remove slack links in favour of discord (:pr:`4202`) :user:`fkiraly`
* [DOC] fix tables in transformer ``transform`` docstrings - change md to rst (:pr:`4199`) :user:`romanlutz`
* [DOC] remove gap between pandas and ``DataFrame`` | ``Series`` in classification notebook (#4200) (:pr:`4200`) :user:`romanlutz`
* [DOC] Fixed table in ``CI`` overview documentation (:pr:`4198`) :user:`pranavvp16`

Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`alex-hh`,
:user:`danbartl`,
:user:`dbcerigo`,
:user:`fkiraly`,
:user:`hoesler`,
:user:`matthewmiddlehurst`,
:user:`NeuralNut`,
:user:`pranavvp16`,
:user:`romanlutz`

Version 0.16.0 - 2023-01-30
---------------------------

Highlights
~~~~~~~~~~

* ``HierarchyEnsembleForecaster`` for level- or node-wise application of forecasters on panel/hierarchical data (:pr:`3905`) :user:`VyomkeshVyas`
* new transformer: ``BKFilter``, Baxter-King filter, interfaced from ``statsmodels`` (:pr:`4127`) :user:`klam-data`, :user:`pyyim``
* ``get_fitted_params`` of pipelines and other heterogeneous meta-estimators now supports parameter nesting (:pr:`4110`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``statsmodels`` is now a soft dependency. Estimators dependent on ``statsmodels``
  can be used exactly as before if ``statsmodels`` is present in the python environment.

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseEstimator
^^^^^^^^^^^^^

* The method ``get_fitted_params``, of all ``BaseEstimator`` descendants
  (any estimator with ``fit``), has a new boolean argument ``deep``, default ``True``.
  Similar to the argument of the same name of ``get_params``, this allows to control
  for composite estimators, whether to return fitted parameters with or
  without estimator nesting.

Forecasting
^^^^^^^^^^^

* all forecasters: the public ``cutoff`` attribute of forecasters has changed
  to ``pd.Index`` subtype, from index element. To update previously
  functional code, replace references to ``cutoff`` by ``cutoff[0]``.


Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies
^^^^^^^^^^^^

* ``statsmodels`` has changed from core dependency to soft dependency in ``sktime 0.16.0``.
  To ensure functioning of setups of ``sktime`` code dependent on ``statsmodels`` based estimators
  going forward, ensure to install ``statsmodels`` in the environment explicitly,
  or install the ``all_extras`` soft dependency set which will continue to contain ``statsmodels``.

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``check_is_scitype``: the ``msg_legacy_interface`` argument has now been removed.
  Future behaviour is as per the default of the argument, ``msg_legacy_interface=False``.

Forecasting
^^^^^^^^^^^

* all forecasters: the public ``cutoff`` attribute of forecasters has changed
  to ``pd.Index`` subtype, from index element. To update previously
  functional code, replace references to ``cutoff`` by ``cutoff[0]``.

Transformations
^^^^^^^^^^^^^^^

* ``Catch22``: the ``transform_single_feature`` method has been removed from the ``Catch22``
  transformer
* ``FourierFeatures``: in 0.17.0, the default value of the ``keep_original_columns``
  parameter will change to ``False``

Enhancements
~~~~~~~~~~~~

BaseEstimator
^^^^^^^^^^^^^

* [ENH] ``get_fitted_params`` for pipelines and other heterogeneous meta-estimators (:pr:`4110`) :user:`fkiraly`
* [ENH] ``deep`` argument for ``get_fitted_params`` (:pr:`4113`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] significantly speed up ``nested_univ`` (nested dataframe) check for non-nested data (:pr:`4130`) :user:`danbartl`
* [ENH] refactor - localize broadcasting in ``VectorizedDF`` (:pr:`4132`) :user:`fkiraly`
* [ENH] ``get_time_index`` rework to get faster run times for grouped data (:pr:`4141`) :user:`danbartl`

Forecasting
^^^^^^^^^^^

* [ENH] ``HierarchyEnsembleForecaster`` for level- or node-wise application of forecasters on panel/hierarchical data (:pr:`3905`) :user:`VyomkeshVyas`
* [ENH] second set of test parameters for ``ARIMA`` (:pr:`4099`) :user:`fkiraly`
* [ENH] Refactor/simplify ``sktime.forecasting.model_selection._split.BaseSplitter._split_vectorized`` (:pr:`4108`) :user:`mateuja`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``PoissonHMM`` estimator (:pr:`4126`) :user:`klam-data`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Reduce repetitive code in ``test_boss.py`` and add check for string datatype in _boss.py (:pr:`4100`) :user:`erjieyong`

Time series generators
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] added ``piecewise_multinomial`` (:pr:`4079`) :user:`JonathanBechtel`
* [ENH] added ``piecewise_poisson`` (:pr:`4121`) :user:`Pyyim`

Transformations
^^^^^^^^^^^^^^^

* [ENH] Add ``keep_original_columns`` option to ``FourierFeatures`` trafo (:pr:`4008`) :user:`KishManani`
* [ENH] Add ``BKFilter`` Transformer (:pr:`4127`) :user:`klam-data`, :user:`pyyim``

Maintenance
~~~~~~~~~~~

* [MNT] Automate updating CONTRIBUTORS.md (:pr:`3807`) :user:`achieveordie`
* [MNT] address ``pd.Series`` constructor ``dtype`` deprecation / ``FutureWarning`` - part 2 (:pr:`4111`) :user:`fkiraly`
* [MNT] 0.16.0 change/deprecation action - ``statsmodels`` as soft dependency (:pr:`3516`) :user:`fkiraly`
* [MNT] emergency fix for precommit CI failure - remove ``isort`` (:pr:`4164`) :user:`fkiraly`
* [MNT] isolate ``statsmodels`` in ``HierarchyEnsembleForecaster`` docstring (:pr:`4166`) :user:`fkiraly`
* [MNT] 0.16.0 deprecation action - change ``BaseForecaster.cutoff`` to ``pd.Index`` (:pr:`3678`) :user:`fkiraly`
* [MNT] isolate ``statsmodels`` in ``HierarchyEnsembleForecaster`` docstring - accidentally missing commit (:pr:`4168`) :user:`fkiraly`
* [MNT] 0.16.0 deprecation & change actions (:pr:`4138`) :user:`fkiraly`
* [MNT] Bump ``isort`` to ``5.12.0`` in ``pre-commit`` config (:pr:`4167`) :user:`snnbotchway`

Documentation
~~~~~~~~~~~~~

* [DOC] fixes table of contents in ``01_forecasting.ipynb`` tutorial (:pr:`4120`) :user:`fkiraly`
* [DOC] improved docstring for ``AutoETS`` (:pr:`4116`) :user:`fkiraly`
* [DOC] Added Paul Yim, Kevin Lam, and Margaret Gorlin to contributor list (:pr:`4122`) :user:`Pyyim`
* [DOC] Fix broken link to the user guide in the glossary (:pr:`4125`) :user:`romanlutz`

Fixes
~~~~~

BaseObject
^^^^^^^^^^

* [BUG] fix faulty ``BaseObject.__eq__`` and ``deep_equals`` if an attribute or nested structure contains ``float`` (:pr:`4109`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] fix ``get_fitted_params`` for forecaster tuners, missing ``best_forecaster`` etc (:pr:`4102`) :user:`fkiraly`
* [BUG] fix ``get_fitted_params`` in case of vectoriztion for forecasters (:pr:`4105`) :user:`fkiraly`
* [BUG] fix erroneous ``int`` coercion of ``TrendForecaster`` and ``PolynomialTrendForecaster`` on ``DatetimeIndex`` (:pr:`4133`) :user:`fkiraly`
* [BUG] Remove unnecessary ``freq`` error in ``_RecursiveReducer`` (:pr:`4124`) :user:`danbartl`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Diagnose and fix sporadic failures in the test suite due to ``MemoryError`` (:pr:`4036`) :user:`achieveordie`
* [BUG] fix - Callbacks cause deep learning estimators to fail (:pr:`4095`) :user:`aaronrmm`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix ``get_fitted_params`` in case of vectoriztion for transformers (:pr:`4105`) :user:`fkiraly`
* [BUG] Fix ``OptionalPassthrough`` ``X_inner_mtype`` tag (:pr:`4115`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`aaronrmm`,
:user:`achieveordie`,
:user:`danbartl`,
:user:`erjieyong`,
:user:`fkiraly`,
:user:`JonathanBechtel`,
:user:`KishManani`,
:user:`klam-data`,
:user:`mateuja`,
:user:`Pyyim`,
:user:`romanlutz`,
:user:`snnbotchway`,
:user:`VyomkeshVyas`

Version 0.15.1 - 2023-01-12
---------------------------

Highlights
~~~~~~~~~~

* substantial speed-ups of boilerplate for panel and hierarchical data,
  may result in 10-50x overall speed improvement on large panel/hierarchical data (:pr:`3935`, :pr:`4061`) :user:`danbartl`
* dunders for time series distances and kernels, for arithmetic composition and pipelining (:pr:`3949`) :user:`fkiraly`
* pipelines and dunders for time series clustering (:pr:`3967`) :user:`fkiraly`
* new estimators: DBSCAN clustering for time series; kernel support vector classifier for time series kernels (:pr:`3950`, :pr:`4003`) :user:`fkiraly`, :user:`josuedavalos`
* notes and troubleshooting guide for installing ``sktime`` under macOS with ARM processors (:pr:`4010`) :user:`dainelli98`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* the ``python_dependencies`` tag now allows full PEP 440 specifier strings for specifying package dependencies

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* new mtypes for time series, panels and hierarchical data that can be used when ``dask`` is installed:
  ``dask_series``, ``dask_panel``, ``dask_hierarchical``. These can be used in estimators now.
  End-to-end integration with ``dask`` is not yet available, but on the roadmap.

Distances, kernels
^^^^^^^^^^^^^^^^^^

* pairwise transformers now possess a method ``transform_diag`` which returns the diagonal of the distance/kernel matrix
* pairwise panel transformers can be composed with each other using arithmetic operations, which will result
  in the corresponding arithmetic combination of transformers, e.g., sum of distances
* pairwise panel transformers can be composed with simple transformers using the ``*`` dunder,
  which will result in a pipeline of first applying the simple transformer, then the pairwise transformer

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* time series clusterers can now be used with ``make_pipeline`` and the ``*`` dunder to
  build linear pipelines with time series transformers

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

* in ``check_estimator`` and ``run_tests``, the ``return_exceptions`` argument has been deprecated,
  and will be replaced with ``raise_exceptions`` (its logical negation) in 0.17.0.
  Until 0.17.0, both arguments will work, with non-defaults being overriding.

Enhancements
~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``dask`` mtypes - part 1, ``Series`` (:pr:`3554`) :user:`fkiraly`
* [ENH] ``dask`` mtypes - part 2, ``Panel`` and ``Hierarchical`` (:pr:`4011`) :user:`fkiraly`
* [ENH] speed up mtype check for ``pandas`` based mtypes with ``pd.PeriodIndex`` (:pr:`3991`) :user:`fkiraly`
* [ENH] improve performance of ``pandas`` based panel and hierarchical mtype checks (:pr:`3935`) :user:`danbartl`
* [ENH] Speed up hierarchical checks and unify with panel approach (:pr:`4061`) :user:`danbartl`

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [ENH] generalize ``AggrDist`` and ``FlatDist`` to allow arbitrary callables, including ``sklearn`` kernel functions (:pr:`3956`) :user:`fkiraly`
* [ENH] ``transform_diag`` method for pairwise transformers, for computing diagonal of distance/kernel matrix (:pr:`3957`) :user:`fkiraly`
* [ENH] wrapper to convert kernels to distances and distances to kernels (:pr:`3958`) :user:`fkiraly`
* [ENH] dunders for time series distances and kernels (:pr:`3949`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] add global forecasting (pooling) options to ``DirectTabularRegressionForecaster`` and ``DirectTimeSeriesRegressionForecaster`` (:pr:`3688`) :user:`danbartl`
* [ENH] forecasting benchmark function ``evaluate`` to accept list of scorers (:pr:`3883`) :user:`aiwalter`
* [ENH] add contract test for hierarchical forecasting (:pr:`3969`) :user:`fkiraly`
* [ENH] extend ``Prophet`` to allow ``pd.PeriodIndex`` (:pr:`3995`) :user:`fkiraly`
* [ENH] improve handling of ``scitype`` in ``make_reduction`` (:pr:`4022`) :user:`fkiraly`
* [ENH] ``hcrystalball`` forecaster adapter (:pr:`4040`) :user:`MichalChromcak`

Pipelines
^^^^^^^^^

* [ENH] ``sklearn`` to ``sktime`` pipeline adapter (:pr:`3970`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] kernel support vector classifier for time series kernels (:pr:`3950`) :user:`fkiraly`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] clustering pipelines and dunders (:pr:`3967`) :user:`fkiraly`
* [ENH] DBSCAN clustering for time series (:pr:`4003`) :user:`fkiraly`, :user:`josuedavalos`

Transformations
^^^^^^^^^^^^^^^

* [ENH] "typical length" constant in transformer scenarios (:pr:`3892`) :user:`fkiraly`
* [ENH] change ``DateTimeFeatures`` trafo to work with multi-index data and add option to drop columns. (:pr:`3996`) :user:`KishManani`
* [ENH] time bin aggregation transformer (:pr:`3997`) :user:`fkiraly`
* [ENH] enable ``TimeSince`` trafo to transform multiindex dataframes natively (:pr:`4006`) :user:`KishManani`
* [ENH] make ``TimeSince`` trafo faster by changing period diff calculation (:pr:`4018`) :user:`KishManani`
* [ENH] clean up ``Detrender``, extend to forecasters which require forecasting horizon in ``fit`` (:pr:`4053`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] update ``_check_soft_dependencies`` to allow PEP 440 specifier strings for version bounds (:pr:`3925`) :user:`fkiraly`
* [ENH] allow tuples/lists of package identifier strings in ``_check_soft_dependencies`` (:pr:`3955`) :user:`fkiraly`
* [ENH] ``_check_estimator_deps`` to also allows list or tuple of ``BaseObject``-s (:pr:`4002`) :user:`fkiraly`
* [ENH] extend ``sklearn_scitype`` to infer the scitype correctly from composites (:pr:`4021`) :user:`fkiraly`
* [ENH] Improve error messages in ``test_estimator_tags`` test (:pr:`4014`) :user:`fkiraly`
* [ENH] in ``check_estimator`` and ``run_tests`` replace ``return_exceptions`` arg  with ``raise_exceptions``, with deprecation (:pr:`4030`) :user:`fkiraly`
* [ENH] add test parameter sets to increase number of test parameter sets per estimator to 2 or larger (:pr:`4043`) :user:`fkiraly`

Visualisations
^^^^^^^^^^^^^^

* [ENH] Implementing plot title for ``plot_series`` (:pr:`4038`) :user:`arnavrneo`

Maintenance
~~~~~~~~~~~

* [MNT] carry out accidentally missed deprecation action for 0.15.0: in ``WEASEL`` and ``BOSS``, remove ``type_dict`` and update default ``alphabet_size=2`` (:pr:`4025`) :user:`xxl4tomxu98`
* [MNT] move ``badrmarani`` contrib to chronological order (:pr:`4029`) :user:`fkiraly`
* [MNT] skip :pr:`4033` related failures until fixed (:pr:`4034`) :user:`fkiraly`
* [MNT] skip ``LSTMFCNClassifier`` tests due to unfixed failure on ``main`` (:pr:`4037`) :user:`fkiraly`
* [MNT] explicit lower version bound on ``scipy`` (:pr:`4019`) :user:`fkiraly`
* [MNT] fix ``_check_soft_dependencies`` breaking for PEP 440 specifiers without class reference (:pr:`4044`) :user:`fkiraly`
* [MNT] downwards compatibility fixes for minimal dependency set (:pr:`4041`) :user:`fkiraly`
* [MNT] address ``pd.Series`` constructor ``dtype`` deprecation / ``FutureWarning`` (:pr:`4031`) :user:`fkiraly`
* [MNT] isolate ``statsmodels``, recent instances (:pr:`4035`) :user:`fkiraly`
* [MNT] address ``pandas`` ``astype`` deprecation / ``FutureWarning`` in ``TrendForecaster`` (:pr:`4032`) :user:`fkiraly`
* [MNT] explicit use of ``min_periods`` args inside ``WindowSummarizer`` to address deprecation message (:pr:`4052`, :pr:`4074`) :user:`arnavrneo`

Documentation
~~~~~~~~~~~~~

* [DOC] complete docstring for ``ForecastingPipeline`` (:pr:`3840`) :user:`darshitsharma`
* [DOC] updates to distances API reference page (:pr:`3852`) :user:`MatthewMiddlehurst`, :user:`fkiraly`
* [DOC] improve ``Detrender`` docstring (:pr:`3948`) :user:`fkiraly`
* [DOC] add some missing entries in API reference (:pr:`3998`) :user:`fkiraly`
* [DOC] API ref for ``pipeline`` module (:pr:`3970`) :user:`fkiraly`
* [DOC] fix the build tag in README (:pr:`4007`) :user:`badrmarani`
* [DOC] warning, notes, and troubleshooting for installing ``sktime`` with macOS ARM (:pr:`4010`) :user:`dainelli98`
* [DOC] ``all_estimators`` reference on all estimator pages (:pr:`4027`) :user:`fkiraly`, :user:`MatthewMiddlehurst`
* [DOC] remove ``make_reduction`` scitype arg in examples (:pr:`4020`) :user:`fkiraly`
* [DOC] more details on code quality and linting (:pr:`4063`) :user:`miraep8`
* [DOC] update list of core devs (:pr:`4085`) :user:`fkiraly`
* [DOC] section on new tests in ``PULL_REQUEST_TEMPLATE`` (:pr:`4093`) :user:`Aarthy153`

Fixes
~~~~~

Distances, kernels
^^^^^^^^^^^^^^^^^^

* [BUG] fix tag logic in ``AggrDist`` and ``FlatDist`` (:pr:`3971`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] fix ``StatsForecastAutoARIMA_.predict`` incorrect in-sample start index (:pr:`3942`) :user:`tianjiqx`
* [BUG] fix ``statsmodels`` estimators when exogenous ``X`` is passed with more indices than ``fh`` (:pr:`3972`) :user:`adoherty21`
* [BUG] fix ``ReconcilerForecaster`` when not used in a pipeline with ``Aggregator`` (:pr:`3980`) :user:`ciaran-g`
* [BUG] fix logic bug in ``ForecastX`` predictions (:pr:`3987`) :user:`aiwalter`, :user:`fkiraly`
* [BUG] fix ``Prophet`` not working with non-integer forecast horizon (:pr:`3995`) :user:`fkiraly`
* [BUG] fix dropped column index in ``BaggingForecaster`` (:pr:`4001`) :user:`fkiraly`
* [BUG] fix ``TrendForecaster`` if ``regressor`` is not boolean coercible (:pr:`4047`) :user:`fkiraly`
* [BUG] fix mutation of ``regressor`` in ``PolynomialTrendForecaster._fit`` (:pr:`4057`) :user:`fkiraly`
* [BUG] fix ``ConformalIntervals`` update when ``sample_frac`` argument is not None (:pr:`4083`) :user:`bethrice44`

Governance
^^^^^^^^^^

* [GOV] code of conduct update - decision making on finances and resource allocation (:pr:`3674`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] constructor of any DL estimator to pass non-default values to underlying ``Network`` object (:pr:`4075`) :user:`achieveordie`
* [BUG] Fix BOSS based classifiers truncating class names to single character length (:pr:`4096`) :user:`erjieyong`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix default ``BaseClusterer._predict_proba`` for all mtypes (:pr:`3985`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] constructor of any DL estimator to pass non-default values to underlying ``Network`` object (:pr:`4075`) :user:`achieveordie`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix ``TimeSince`` check of inconsistency between ``time_index`` and ``start`` (:pr:`4015`) :user:`KishManani`
* [BUG] fix multivariate and hierarchical behaviour of ``Detrender`` (:pr:`4053`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [BUG] fix ``_check_soft_dependencies`` breaking for PEP 440 specifiers without class reference (:pr:`4044`) :user:`fkiraly`

Visualisations
^^^^^^^^^^^^^^

* [BUG] ``plot_cluster_algorithm``: fix error ``predict_series is undefined`` if ``X`` is passed as ``np.ndarray`` (:pr:`3933`) :user:`hakim89`

Contributors
~~~~~~~~~~~~

:user:`Aarthy153`,
:user:`achieveordie`,
:user:`adoherty21`,
:user:`aiwalter`,
:user:`arnavrneo`,
:user:`badrmarani`,
:user:`bethrice44`,
:user:`ciaran-g`,
:user:`dainelli98`,
:user:`danbartl`,
:user:`darshitsharma`,
:user:`erjieyong`,
:user:`fkiraly`,
:user:`hakim89`,
:user:`josuedavalos`,
:user:`KishManani`,
:user:`MatthewMiddlehurst`,
:user:`MichalChromcak`,
:user:`miraep8`,
:user:`patrickzib`,
:user:`tianjiqx`,
:user:`xxl4tomxu98`

Version 0.15.0 - 2022-12-22
---------------------------

Highlights
~~~~~~~~~~~~

* ``MLflow`` custom flavor for ``sktime`` forecasting (:pr:`3912`, :pr:`3915`) :user:`benjaminbluhm`
* compatibility with most recent versions of core dependencies ``sktime 1.2.0``and ``numpy 1.24`` (:pr:`3922`) :user:`fkiraly`
* ``TimeBinner`` transformation for temporal bin aggregation (:pr:`3745`) :user:`kcc-lion`
* E-Agglo estimator for hierarchical agglomerative cluster estimation (:pr:`3430`) :user:`KatieBuc`
* week-end dummy ``is_weekend`` in ``DateTimeFeatures`` transformation (:pr:`3844`) :user:`KishManani`
* deep learning classifiers migrated from ``sktime-dl`` to ``sktime``: ResNet, LSTM-FCN (:pr:`3714`, :pr:`3881`) :user:`nilesh05apr`, :user:`solen0id`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``sktime`` is now compatible with ``numpy 1.24``, bound is relaxed to ``<1.25``
* ``sktime`` is now compatible with ``sklearn 1.2.0``, bound is relaxed to ``<1.3.0``
* ``pycatch22`` is no longer a soft dependency of ``sktime``, due to installation issues.
  ``pycatch22`` based transformers are still functional if the dependency is installed in the python environment.
* ``statsmodels`` will change from core dependency to soft dependency in ``sktime 0.16.0``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

Comparison by equality for any ``sktime`` object now compares identity of parameters,
as obtained via ``get_params``, with recursive application if objects/estimators are nested.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Dependencies
^^^^^^^^^^^^

* ``statsmodels`` will change from core dependency to soft dependency in ``sktime 0.16.0``.
  To ensure functioning of setups of ``sktime`` code dependent on ``statsmodels`` based estimators
  after the deprecation period, ensure to install ``statsmodels`` in the environment explicitly,
  or install the ``all_extras`` soft dependency set which will continue to contain ``statsmodels``.

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``datatypes.check_is_scitype``: 2nd return argument (only returned if ``return_metadata=True``)
will be changed from ``list`` to ``dict`` format (see docstring).
The ``list`` format is deprecated since 0.14.0, and replaced by ``dict`` in 0.15.0.
The format is determined by temporary additional arg ``msg_legacy_interface``, which will be
the default has now changed to ``False`` (``dict`` format).
The ``msg_legacy_interface`` argument and the option to return the legacy ``list`` format will be removed in 0.16.0.

Forecasting
^^^^^^^^^^^

* ``ExpandingWindowSplitter`` had ``start_with_window`` argument removed. From now on, ``initial_window=0`` should be used instead of ``start_with_window=False``.
* the row transformers, ``SeriesToSeriesRowTransformer`` and ``SeriesToPrimitivesRowTransformer`` have been removed.
  Row/instance vectorization functionality is natively supported by ``sktime`` since 0.11.0 and does not need to be added by these wrappers anymore.
  Both transformers will be removed in 0.15.0. To migrate, simply remove the row transformer wrappers.
  In some rarer, ambiguous vectorization cases (e.g., using wrapped functions that are vectorized, such as ``np.mean``),
  ``FunctionTransformer`` may have to be used instead of ``SeriesToPrimitivesRowTransformer``.
* change to public ``cutoff`` attribute delayed to 0.16.0:
  public ``cutoff`` attribute of forecasters will change to ``pd.Index`` subtype, from index element.

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* Delayed: the base class of ``ProbabilityThresholdEarlyClassifier`` will be changed to ``BaseEarlyClassifier`` in 0.16.0.
  This will change how classification safety decisions are made and returned, see ``BaseEarlyClassifier`` or ``TEASER`` for the new interface.

Transformations
^^^^^^^^^^^^^^^

* ``transformations.series.compose`` has been removed in favour of ``transformations.compose``.
  All estimators in the former have been moved to the latter.
* The default of ``default_fc_parameters`` in ``TSFreshFeatureExtractor`` and ``TSFreshRelevantFeatureExtractor``
  has beenchanged from ``"efficient"`` to ``"comprehensive"``.

Testing framework
^^^^^^^^^^^^^^^^^

* The general interface contract test ``test_methods_do_not_change_state`` has been renamed to ``test_non_state_changing_method_contract``

Enhancements
~~~~~~~~~~~~

MLOps & Deployment
~~~~~~~~~~~~~~~~~~

* [ENH] MLflow custom flavor for ``sktime`` forecasting (:pr:`3912`) :user:`benjaminbluhm`

BaseObject
^^^^^^^^^^

* [ENH] equality dunder for ``BaseObject`` to compare blueprint (:pr:`3862`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] Check for frequency in hierarchical data, provide utility function to set frequency for hierarchical data (:pr:`3729`) :user:`danbartl`
* [ENH] forecasting pipeline ``get_fitted_params`` (:pr:`3863`) :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] E-Agglo estimator for hierarchical agglomerative cluster estimation (:pr:`3430`) :user:`KatieBuc`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Migrate LSTM-FCN classifier  from ``sktime-dl`` to ``sktime`` (:pr:`3714`) :user:`solen0id`
* [ENH] Migrate ``ResNetClassifier`` from ``sktime-dl`` to ``sktime`` (:pr:`3881`) :user:`nilesh05apr`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``DummyRegressor`` for time series regression (:pr:`3968`) :user:`badrmarani`

Transformations
^^^^^^^^^^^^^^^

* [ENH] ``TimeBinner`` transformation for temporal bin aggregation (:pr:`3745`) :user:`kcc-lion`
* [ENH] Add ``is_weekend`` option to ``DateTimeFeatures`` trafo (:pr:`3844`) :user:`KishManani`
* [ENH] Add multiplicative option to ``Detrender`` (:pr:`3931`) :user:`KishManani`

Visualisations
^^^^^^^^^^^^^^

* [ENH] Add support for plotting intervals in ``plot_series`` (:pr:`3825`) :user:`chillerobscuro`
* [ENH] Add ``colors`` argument to ``plot_series`` (:pr:`3908`) :user:`chillerobscuro`

Fixes
~~~~~

Forecasting
^^^^^^^^^^^

* [BUG] in ``ConformalIntervals``, fix update of residuals matrix for sliding window splitter (:pr:`3914`) :user:`bethrice44`
* [BUG] fix ``start_with_window`` deprecation in ``ExpandingWindowSplitter`` (:pr:`3953`) :user:`fkiraly`
* [BUG] fix ``EnsembleForecaster`` erroneous broadcasting and attribute clash (:pr:`3964`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix unreported ``set_params`` bug in ``ClassifierPipeline`` and ``RegressorPipeline`` (:pr:`3857`) :user:`fkiraly`
* [BUG] fixes KNN estimators' ``kneighbors`` methods to work with all mtypes (:pr:`3927`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix unreported ``set_params`` bug in ``ClassifierPipeline`` and ``RegressorPipeline`` (:pr:`3857`) :user:`fkiraly`
* [BUG] fixes KNN estimators' ``kneighbors`` methods to work with all mtypes (:pr:`3927`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] ``ClearSky`` doesn't raise error for range indexes and when ``X`` has no set frequency (:pr:`3872`) :user:`ciaran-g`
* [BUG] ``sklearn 1.2.0`` compatibility - fix invalid elbow variable selection shrinkage parameter passed to ``sklearn`` ``NearestCentroid`` (:pr:`3921`) :user:`fkiraly`

Visualisations
^^^^^^^^^^^^^^

* [BUG] fix soft dependency check in ``plotting.plot_correlations`` (:pr:`3887`) :user:`dsanr`


Documentation
~~~~~~~~~~~~~

* [DOC] fixed rendering in dependencies doc (:pr:`3846`) :user:`templierw`
* [DOC] update transformers extension section in transformers tutorial (:pr:`3860`) :user:`fkiraly`
* [DOC] tidying Rocket docstrings (:pr:`3860`) :user:`TonyBagnall`
* [DOC] added post-processing in pipelines to forecasting tutorial (:pr:`3878`) :user:`nshahpazov`
* [DOC] changing import path for ``plot_cluster_algorithm`` (:pr:`3945`) :user:`GianFree`

Maintenance
~~~~~~~~~~~

* [MNT] Additional project urls in ``pyproject.toml`` (#3864) :user:`lmmentel`
* [MNT] ``sklearn 1.2.0`` compatibility - remove private ``_check_weights`` import in ``KNeighborsTimeSeriesClassifier`` and -``Regressor`` (:pr:`3918`) :user:`fkiraly`
* [MNT] ``sklearn 1.2.0`` compatibility - cover ``BaseForest`` parameter change (:pr:`3919`) :user:`fkiraly`
* [MNT] ``sklearn 1.2.0`` compatibility - decouple ``sklearn.base._pprint`` (:pr:`3923`) :user:`fkiraly`
* [MNT] ``sklearn 1.2.0`` compatibility - remove ``normalize=False`` args from ``RidgeClassifierCV`` (:pr:`3924`) :user:`fkiraly`
* [MNT] ``sklearn 1.2.0`` compatibility - ``ComposableTimeSeriesForest`` reserved attribute fix (:pr:`3926`) :user:`fkiraly`
* [MNT] remove ``pycatch22`` as a soft dependency (:pr:`3917`) :user:`fkiraly`
* [MNT] Update ``sklearn`` compatibility to ``1.2.x``, version bound to ``<1.3`` (:pr:`3922`) :user:`fkiraly`
* [MNT] bump ``numpy`` version bound to ``<1.25`` and fix compatibility issues (:pr:`3915`) :user:`aquemy`, :user:`fkiraly`
* [MNT] ``0.15.0`` deprecation actions (:pr:`3952`) :user:`fkiraly`
* [MNT] skip sporadic ``ResNetClassifier`` failures (:pr:`3974`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`aiwalter`,
:user:`aquemy`,
:user:`badrmarani`,
:user:`benjaminbluhm`,
:user:`bethrice44`,
:user:`chillerobscuro`,
:user:`ciaran-g`,
:user:`danbartl`,
:user:`dsanr`,
:user:`fkiraly`,
:user:`GianFree`,
:user:`KatieBuc`,
:user:`kcc-lion`,
:user:`KishManani`,
:user:`lmmentel`,
:user:`nilesh05apr`,
:user:`nshahpazov`,
:user:`solen0id`,
:user:`templierw`,
:user:`TonyBagnall`

Version 0.14.1 - 2022-11-30
---------------------------

Highlights
~~~~~~~~~~

* dedicated notebook tutorial for transformers and feature engineering - stay tuned for more at pydata global 2022! (:pr:`1705`) :user:`fkiraly`
* documentation & step-by-step guide to add a new dataset loader (:pr:`3805`) :user:`templierw`
* new transformer: ``Catch22Wrapper``, direct interface for ``pycatch22`` (:pr:`3431`) :user:`MatthewMiddlehurst`
* new transformer: ``TimeSince`` for feature engineering, time since fixed date/index (:pr:`3810`) :user:`KishManani`
* permutation wrapper ``Permute`` for tuning of estimator order in forecatsing pipelines (:pr:`3689`) :user:`aiwalter` :user:`fkiraly`
* all soft dependencies are now isolated in tests, all tests now run with minimal dependencies (:pr:`3760`) :user:`fkiraly`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* dunder method for variable subsetting exogeneous data: ``my_forecaster[variables]`` will create a ``ForecastingPipeline``
  that subsets the exogeneous data to ``variables``

Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] default ``get_params`` / ``set_params`` for ``_HeterogenousMetaEstimator`` & [BUG] fix infinite loop in ``get_params`` for ``FeatureUnion``, with hoesler (:pr:`3708`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] direct reducer prototype rework based on feedback (:pr:`3382`) :user:`fkiraly`
* [ENH] forecast default update warning to point to stream forecasting wrappers (:pr:`3410`) :user:`fkiraly`
* [ENH] getitem / square brackets dunder for forecasting (:pr:`3740`) :user:`fkiraly`
* [ENH] Add test for global forecasting case (:pr:`3728`) :user:`danbartl`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``Catch22Transformer`` update and ``Catch22Wrapper`` for ``pycatch22`` (:pr:`3431`) :user:`MatthewMiddlehurst`
* [ENH] ``MinirocketMultivariateVariable`` transformer, miniROCKET for unequal length time series (:pr:`3786`) :user:`michaelfeil`
* [ENH] slightly speed up the tests for ``ComposableTimeSeriesForestClassifier`` (:pr:`3762`) :user:`TonyBagnall`
* [ENH] Warning rather than error for TDE small series (:pr:`3767`) :user:`MatthewMiddlehurst`
* [ENH] Add some ``get_test_params`` values to deep learning classifiers and regressors (:pr:`3761`) :user:`TonyBagnall`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Add some ``get_test_params`` values to deep learning classifiers and regressors (:pr:`3761`) :user:`TonyBagnall`

Transformations
^^^^^^^^^^^^^^^

* [ENH] better error message on transform output check fail (:pr:`3724`) :user:`fkiraly`
* [ENH] second test case for ``FeatureUnion``, construction without names (:pr:`3792`) :user:`fkiraly`
* [ENH] permutation wrapper ``Permute`` for tuning of pipeline sequence (:pr:`3689`) :user:`aiwalter` :user:`fkiraly`
* [ENH] ``fit_transform`` for ``TSFreshRelevantFeatureExtractor`` (:pr:`3785`) :user:`MatthewMiddlehurst`
* [ENH] ``TimeSince`` transformer for feature engineering, time since fixed date/index (:pr:`3810`) :user:`KishManani`

Governance
^^^^^^^^^^

* [GOV] Add :user:`achieveordie` as a core developer (:pr:`3851`) :user:`achieveordie`

Fixes
~~~~~

Data loaders
^^^^^^^^^^^^

* [BUG] remove test and add warning to ``load_solar`` (:pr:`3771`) :user:`ciaran-g`

Forecasting
^^^^^^^^^^^

* [BUG] fix ``ColumnEnsembleForecaster`` for hierarchical ``X`` (:pr:`3768`) :user:`RikStarmans` :user:`fkiraly`
* [BUG] decouple forecasting pipeline module from registry (:pr:`3799`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] ``keras`` import quick-fix (:pr:`3744`) :user:`ltsaprounis`
* [BUG] in ``TemporalDictionaryEnsemble``, set ``Parallel`` ``prefer="threads"``, fixes #3788 (:pr:`3808`) :user:`TonyBagnall`
* [BUG] in ``DummyClassifier``, fix incorrectly set ``capability:multivariate`` tag (:pr:`3858`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix behaviour of `FourierFeatures` with `pd.DatetimeIndex` (:pr:`3606`) :user:`eenticott-shell`
* [BUG] fix infinite loop in ``get_params`` for ``FeatureUnion`` (:pr:`3708`) :user:`hoesler` :user:`fkiraly`
* [BUG] ``SupervisedIntervals`` bugfixes and clean up (:pr:`3727`) :user:`MatthewMiddlehurst`
* [BUG] Reduce size of ``MultiRocket`` test example to avoid sporadic ``MemoryError`` in testing (:pr:`3813`) :user:`TonyBagnall`
* [BUG] fix return index for transformers' ``Primitives`` output in row vectorization case (:pr:`3839`) :user:`fkiraly`
* [BUG] in ``Reconciler``, fix summation matrix bug for small hierarchies with one unique ID in outer index (:pr:`3859`) :user:`ciaran-g`

Testing framework
^^^^^^^^^^^^^^^^^

* [BUG] Update ``test_deep_estimator_full`` to incorporate new versions of ``tensorflow`` / ``keras`` (:pr:`3820`) :user:`achieveordie`

Documentation
~~~~~~~~~~~~~

* [DOC] transformers tutorial (:pr:`1705`) :user:`fkiraly`
* [DOC] Update documentation for Greedy Gaussian Segmentation (:pr:`3739`) :user:`lmmentel`
* [DOC] Compose and deep learning classifier doc tidy (:pr:`3756`) :user:`TonyBagnall`
* [DOC] added new slack link (:pr:`3747`) :user:`hadifawaz1999`
* [DOC] Updates documentation for channel selection (:pr:`3770`) :user:`haskarb`
* [DOC] Update File Format Specifications page to show list of hyperlinked formats (:pr:`3775`) :user:`achieveordie`
* [DOC] Examples webpage (:pr:`3653`) :user:`MatthewMiddlehurst`
* [DOC] Update CC and CoC and active core-devs lists in ``team.rst`` (:pr:`3733`) :user:`GuzalBulatova`
* [DOC] Improve ShapeletTransformClassifier docstring (:pr:`3737`) :user:`MatthewMiddlehurst`
* [DOC] Improve sklearn classifier docstrings (:pr:`3754`) :user:`MatthewMiddlehurst`
* [DOC] Add missing estimators to classification API page (:pr:`3742`) :user:`MatthewMiddlehurst`
* [DOC] Updates to regression API reference (:pr:`3751`) :user:`TonyBagnall`
* [DOC] Fixed doc typo in ``RocketClassifier`` docstring (:pr:`3759`) :user:`matt-wisdom`
* [DOC] Include section on unequal length data in classification notebook (:pr:`3809`) :user:`MatthewMiddlehurst`
* [DOC] documentation on workflow of adding a new dataset loader (:pr:`3805`) :user:`templierw`
* [DOC] add defaults in ``ScaledLogitTransformer`` docstring (:pr:`3845`) :user:`fkiraly`
* [DOC] Added ``ForecastByLevel`` to API docs (:pr:`3837`) :user:`aiwalter`
* [DOC] Update CONTRIBUTORS.md (:pr:`3781`) :user:`achieveordie`
* [DOC] Docstring improvements to ``TSFreshRelevantFeatureExtractor`` (:pr:`3785`) :user:`MatthewMiddlehurst`


Maintenance
~~~~~~~~~~~

* [MNT] Converted ``setup.py`` to ``pyproject.toml``. Depends on ``setuptools>61.0.0`` (:pr:`3723`) :user:`jorenham` :user:`wolph`
* [MNT] decouple forecasting pipeline module from registry (:pr:`3799`) :user:`fkiraly`
* [MNT] temporary skip of new failure ``test_deep_estimator_full[keras-adamax]`` (:pr:`3817`) :user:`fkiraly`
* [MNT] isolate soft dependencies in tests (:pr:`3760`) :user:`fkiraly`
* [MNT] fix ``pyproject.toml`` broken string (:pr:`3797`) :user:`TonyBagnall`
* [MNT] exclude ``TapNet`` from tests (:pr:`3812`) :user:`TonyBagnall`
* [MNT] test soft dependency isolation in non-suite tests (:pr:`3750`) :user:`fkiraly`
* [MNT] Address ``ContinuousIntervalTree`` and ``RandomShapeletTransform`` deprecation warnings (:pr:`3796`) :user:`MatthewMiddlehurst`
* [MNT] isolate ``statsmodels``, part 4: isolating ``statsmodels`` in non-suite tests (:pr:`3821`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`aiwalter`,
:user:`ciaran-g`,
:user:`danbartl`,
:user:`eenticott-shell`,
:user:`fkiraly`,
:user:`hadifawaz1999`,
:user:`haskarb`,
:user:`hoesler`,
:user:`jorenham`,
:user:`KishManani`,
:user:`lmmentel`,
:user:`matt-wisdom`,
:user:`MatthewMiddlehurst`,
:user:`michaelfeil`,
:user:`RikStarmans`,
:user:`templierw`,
:user:`TonyBagnall`,
:user:`wolph`

Version 0.14.0 - 2022-11-05
---------------------------

Highlights
~~~~~~~~~~

* serialization and deserialization of all ``sktime`` objects via ``save`` method & ``base.load`` (:pr:`3336`, :pr:`3425`) :user:`achieveordie` :user:`fkiraly`
* documented format specification for ``.ts`` files (:pr:`3380`) :user:`achieveordie`
* new forecaster: modular/configurable Theta forecaster (:pr:`1300`) :user:`GuzalBulatova`
* new probabilistic prediction adder for forecasters: squaring residuals (:pr:`3378`) :user:`kcc-lion`
* forecasting ``evaluate`` now supports hierarchical and panel data and parallelism via ``dask`` and ``joblib`` (:pr:`3511`, :pr:`3542`) :user:`topher-lo` :user:`fkiraly`
* ``get_fitted_params`` now supported for all estimators via defaults (:pr:`3645`) :user:`fkiraly`

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

BaseObject & BaseEstimator
^^^^^^^^^^^^^^^^^^^^^^^^^^

* all objects and estimators (``BaseObject`` descendants) now possess a ``save`` method for serialization to memory or file.
  Serialized objects can be deserialized by ``base.load``.
  Interface contracts on ``save`` and ``load`` are now tested by the standard test suite (e.g., ``check_estimator``).
* all fittable objects ("estimators", ``BaseEstimator`` descendants) now have a functioning default implementation of ``get_fitted_params``.
  Interface contracts on ``get_fitted_params`` are now tested by the standard test suite (e.g., ``check_estimator``).
* the extender contract for ``get_fitted_params`` has changed. For new implementations of ``sktime`` estimators,
  developers should implement ``_get_fitted_params`` rather than ``get_fitted_params`` directly, similar to ``fit`` and ``_fit``.
  The extension templates have been updated accordingly. Estimators following the old extension contract are still compatible
  for the time being and will remain compatible at least until 0.15.0.

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* ``ExpandingWindowSplitter`` parameter ``start_with_window`` is deprecated and will be removed in 0.15.0.
  For continued functionality of ``start_with_window=True``, use ``start_with_window=0`` instead.
  Other values of ``start_with_window`` will behave as in the case ``start_with_window=False``.
* Isolated ``pd.timedelta`` elements should no longer be passed to splitters and ``ForecastingHorizon``,
  as ``pandas`` has deprecated ``freq`` for ``pd.delta``.
  Exceptions will be raised in corner cases where ``freq`` as not been passed and cannot be inferred.
* change to public ``cutoff`` attribute delayed to 0.15.0:
  public ``cutoff`` attribute of forecasters will change to ``pd.Index`` subtype, from index element.

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* The base class of ``ProbabilityThresholdEarlyClassifier`` will be changed to ``BaseEarlyClassifier`` in 0.15.0.
  This will change how classification safety decisions are made and returned, see ``BaseEarlyClassifier`` or ``TEASER`` for the new interface.

Transformations
^^^^^^^^^^^^^^^

* The default of ``default_fc_parameters`` in ``TSFreshFeatureExtractor`` and ``TSFreshRelevantFeatureExtractor``
  will change from ``"efficient"`` to ``"comprehensive"`` in 0.15.0.

Testing framework
^^^^^^^^^^^^^^^^^

* The name of the test ``test_methods_do_not_change_state`` will change to
  ``test_non_state_changing_method_contract`` in 0.15.0.
  For a safe transition in a case where the old name
  has been used as part of an argument in ``check_estimator``, use
  both the new and the old name (in a list) in test/fixture exclusion or inclusion.

Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] ``get_args`` default handling for keys not present (:pr:`3595`) :user:`fkiraly`
* [ENH] improve base class test docstrings and clone test (:pr:`3555`) :user:`fkiraly`
* [ENH] ``get_fitted_params`` for nested ``sklearn`` components (:pr:`3645`) :user:`fkiraly`
* [ENH] Serialization and deserialization of estimators (:pr:`3336`) :user:`fkiraly`
* [ENH] Serialization and deserialization of deep learning estimators (:pr:`3425`) :user:`achieveordie`

Data loaders
^^^^^^^^^^^^

* [ENH] support for ``@targetlabel`` identifier for ``.ts`` files in ``load_from_tsfile`` (:pr:`3436`) :user:`achieveordie`
* [ENH] refactor/integrate ``_contrib`` - ``datasets`` (:pr:`3518`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ``dask`` conversion adapters for multi-indexed ``pandas.DataFrame`` (:pr:`3513`) :user:`fkiraly`
* [ENH] refactor mtype conversion extension utils into one location (:pr:`3514`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] modular/configurable Theta forecaster (:pr:`1300`) :user:`GuzalBulatova`
* [ENH] global/local recursive reduction prototype (:pr:`3333`) :user:`fkiraly`
* [ENH] Squaring residuals estimator (:pr:`3378`) :user:`kcc-lion`
* [ENH] extend recursive strategy in ``make_reduction`` to allow global pooling on panel data  (:pr:`3451`) :user:`danbartl`
* [EHN] Parallelized ``evaluate`` with ``{joblib, dask}`` (:pr:`3511`) :user:`topher-lo`
* [ENH] use ``statsmodels`` ``append`` in ``_StatsModelsAdapter._update`` (:pr:`3527`) :user:`chillerobscuro`
* [ENH] extend ``evaluate`` to hierarchical and panel data (:pr:`3542`) :user:`fkiraly`
* [ENH] ``numpy`` integer support for ``ColumnEnsembleForecaster`` (:pr:`3557`) :user:`fkiraly`
* [ENG] forecast-by-level wrapper (:pr:`3585`) :user:`fkiraly`
* [ENH] multivariate test case for ``EnsembleForecaster`` (:pr:`3637`) :user:`fkiraly`
* [ENH] extend ``ColumnEnsembleForecaster`` to allow application of multivariate forecasters (:pr:`3504`) :user:`fkiraly`
* [ENH] add forecaster test case with string columns (:pr:`3506`) :user:`fkiraly`
* [ENH] extend forecasting grid/random search to hierarchical and panel data (:pr:`3548`) :user:`fkiraly`
* [ENH] Make ``EnsembleForecaster`` work with multivariate data (:pr:`3623`) :user:`AnH0ang`
* [ENH] ``ExpandingWindowSplitter`` fix for ``initial_window=0`` and deprecating ``"start_with_window"`` (:pr:`3690`) :user:`chillerobscuro`

Parameter estimation
^^^^^^^^^^^^^^^^^^^^

* [ENH] fixed parameter setter estimator (:pr:`3639`) :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Information Gain Temporal Segmentation Estimator (:pr:`3399`) :user:`lmmentel`
* [ENH] Segmentation metrics (:pr:`3403`) :user:`lmmentel`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] TapNet DL Model for classification (:pr:`3386`) :user:`achieveordie`
* [ENH] refactor/integrate ``_contrib`` - ``diagram_code`` (:pr:`3519`) :user:`fkiraly`
* [ENH] fast test parameters for ``TapNet`` estimators and docstring/interface cleanup (:pr:`3544`) :user:`achieveordie`
* [ENH] more relevant parameters to ``CNNRegressor`` for user flexibility (:pr:`3561`) :user:`achieveordie`
* [ENH] allow ``KNeighborsTimeSeriesClassifier`` to handle distances between unequal length series (:pr:`3654`) :user:`fkiraly`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] TapNet DL Model for regression from ``sktime-dl`` (:pr:`3481`) :user:`achieveordie`
* [ENH] allow ``KNeighborsTimeSeriesRegressor`` to handle distances between unequal length series(:pr:`3654`) :user:`fkiraly`


Transformations
^^^^^^^^^^^^^^^

* [ENH] test that ``TruncationTransformer`` preserves index and column names in ``pd-multiindex`` (:pr:`3535`) :user:`fkiraly`
* [ENH] replace inplace sort by non-inplace sort in ``Reconciler`` (:pr:`3553`) :user:`fkiraly`
* [ENH] ``SupervisedIntervals`` transformer and cleaned ``numba`` functions (:pr:`3622`) :user:`MatthewMiddlehurst`
* [ENH] ``TSFreshFeatureExtractor`` cleanup, tests, and docstring (:pr:`3636`) :user:`kcc-lion`
* [ENH] Option to fit ``Clearsky`` transformer in parallel (:pr:`3652`) :user:`ciaran-g`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] tests for ``get_fitted_params`` interface contract by estimator (:pr:`3590`) :user:`fkiraly`

Governance
^^^^^^^^^^

* [GOV] add :user:`GuzalBulatova` to CC (:pr:`3505`) :user:`GuzalBulatova`
* [GOV] add :user:`miraep8` to core developers (:pr:`3610`) :user:`miraep8`
* [GOV] new CC observers role, update to role holders list (:pr:`3505`) :user:`GuzalBulatova`
* [GOV] minor clarifications of governance (:pr:`3581`) :user:`fkiraly`
* [GOV] clarifications on algorithm maintainer role (:pr:`3676`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] update docs on releasing conda packages (:pr:`3279`) :user:`lmmentel`
* [DOC] Add Format Specification for ``.ts`` files. (:pr:`3380`) :user:`achieveordie`
* [DOC] clarifications on deprecation notes (:pr:`3411`) :user:`fkiraly`
* [DOC] Update CONTRIBUTORS.md (:pr:`3503`) :user:`shagn`
* [DOC] ``sklearn`` usage examples in classifier notebook (:pr:`3523`) :user:`MatthewMiddlehurst`
* [DOC] update extension templates and docstrings for ``_get_fitted_params`` (:pr:`3589`) :user:`fkiraly`
* [DOC] Replace ``sphinx-panels`` with ``sphinx-design`` (:pr:`3575`) :user:`MatthewMiddlehurst`
* [DOC] fixes outdated points of contact in code of conduct (:pr:`3593`) :user:`fkiraly`
* [DOC] fixes incorrect coc issue reporting link in issue tracker and remaining references in coc (:pr:`3594`) :user:`fkiraly`
* [DOC] Add API documentation for the annotation subpackage (:pr:`3603`) :user:`lmmentel`
* [DOC] invite to fall dev days on website landing page (:pr:`3607`) :user:`miraep8`
* [DOC] add recommendations for ``get_test_params`` in extension templates (:pr:`3635`) :user:`achieveordie`

Maintenance
~~~~~~~~~~~

* [MNT] 0.14.0 deprecation actions (:pr:`3677`) :user:`fkiraly`
* [MNT] Bump pre-commit action from 2 to 3 (:pr:`3576`) :user:`lmmentel`
* [MNT] Bump setup-python action from 2 to 4 (:pr:`3577`) :user:`lmmentel`
* [MNT] Remove ``ABCMeta`` inheritance from ``_HeterogeneousMetaEstimator`` (:pr:`3569`) :user:`fkiraly`
* [MNT] loosen ``scipy`` bound to <2.0.0 (:pr:`3587`) :user:`fkiraly`
* [MNT] Replace deprecated ``sphinx-panels`` with ``sphinx-design`` (:pr:`3575`) :user:`MatthewMiddlehurst`
* [MNT] Bump checkout action from 2 to 3 (:pr:`3578`) :user:`lmmentel`
* [MNT] temporarily remove stochastically failing tapnet from tests (:pr:`3624`) :user:`fkiraly`
* [MNT] replace ``ARIMA`` used in tests by reducer to remove soft dependency in tests (:pr:`3552`) :user:`fkiraly`
* [MNT] replace author names by GitHub ID in author fields, linting (:pr:`3628`) :user:`fkiraly`
* [ENH] isolate ``statsmodels`` imports (:pr:`3445`) :user:`fkiraly`
* [MNT] isolate ``statsmodels`` imports, part 2 (:pr:`3515`) :user:`fkiraly`
* [MNT] isolate ``statsmodels``, part 3: replace dependent estimators in test parameters (:pr:`3632`) :user:`fkiraly`
* [MNT] replace author names by GitHub ID in author fields, linting (:pr:`3628`) :user:`fkiraly`

Refactored
~~~~~~~~~~

* [ENH] refactor remaining ``get_fitted_params`` overrides to ``_get_fitted_params`` (:pr:`3591`) :user:`fkiraly`
* [BUG] fix ``get_fitted_params`` for non-conformant estimators (:pr:`3599`) :user:`fkiraly`

Fixes
~~~~~

BaseObject
^^^^^^^^^^

* [BUG] fix ``get_fitted_params`` default for unfittable components (:pr:`3598`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [BUG] fix bug with data loading from timeseriesclassification.com when ``extract_path`` is not ``None`` (:pr:`3021`) :user:`TonyBagnall`
* [BUG] fix error in writing datasets to file in ts format (:pr:`3532`) :user:`TonyBagnall`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``pd.concat`` in stratified resampling causing error in ``check_is_scitype`` (:pr:`3546`) :user:`TonyBagnall`
* [BUG] fix ``check_estimator`` exclude arguments not working for non-base scitype tests (:pr:`3566`) :user:`fkiraly`
* [BUG] fix erroneous asserts in input checkers (:pr:`3556`) :user:`fkiraly`
* [BUG] Exclude ``np.timedelta64`` from ``is_int`` check (:pr:`3627`) :user:`khrapovs`
* [BUG] fix ``get_cutoff`` for ``numpy`` format (:pr:`3442`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] fix ``ConformalIntervals`` update does not update ``residuals_matrix`` (:pr:`3460`) :user:`bethrice44`
* [BUG] Fix side effect of ``predict_residuals`` (:pr:`3475`) :user:`aiwalter`
* [BUG] Fix residuals formula in ``NaiveForecaster.predict_var`` for non-null ``window_length`` (:pr:`3495`) :user:`topher-lo`
* [BUG] fix ``ColumnEnsembleForecaster`` for ``str`` index (:pr:`3504`) :user:`canbooo` :user:`fkiraly`
* [BUG] Fix pipeline tags for NaN values (:pr:`3549`) :user:`aiwalter`
* [BUG] fix conditional ``requires-fh-in-fit`` tag in ``EnsembleForecaster`` (:pr:`3642`) :user:`fkiraly`

Parameter estimation
^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix ``PluginParamsForecaster`` docstring and add dict use example (:pr:`3643`) :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fixing tags typo in ``BaseHmmLearn`` (:pr:`3563`) :user:`guzalbulatova` :user:`miraep8`

Time series clustering
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Pass all average params to kmeans (:pr:`3486`) :user:`chrisholder`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``KNeighborsTimeSeriesClassifier`` tag handling dependent on distance component (:pr:`3654`) :user:`fkiraly`
* [BUG] Add missing ``get_test_params`` to ``TapNet`` estimators (:pr:`3541`) :user:`achieveordie`
* [BUG] ``numba`` / ``np.median`` interaction raises error for large data sets run with ``n_jobs>1`` (:pr:`3602`) :user:`TonyBagnall`
* [BUG] bug in the interaction between ``numba`` and ``np.zeros`` identified in #2397 (:pr:`3618`) :user:`TonyBagnall`
* [BUG] various small bugfixes (:pr:`3706`) :user:`MatthewMiddlehurst`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fixed msm alignment path (:pr:`3484`) :user:`chrisholder`
* [BUG] TWE alignment path fix and refactor (:pr:`3485`) :user:`chrisholder`
* [BUG] Fix typo in ``set_tags`` call in ``AggrDist.__init__`` (:pr:`3562`) :user:`aiwalter`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix ``KNeighborsTimeSeriesRegressor`` tag handling dependent on distance component (:pr:`3654`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] ``RandomShapeletTransform``: floor the maximum number of shapelets to number of classes (:pr:`3564`) :user:`TonyBagnall`
* [BUG] ``ClearSky`` transformer: fix missing value problem after transform (:pr:`3579`) :user:`ciaran-g`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`aiwalter`,
:user:`AnH0ang`,
:user:`arampuria19`,
:user:`bethrice44`,
:user:`canbooo`,
:user:`chillerobscuro`,
:user:`chrisholder`,
:user:`ciaran-g`,
:user:`danbartl`,
:user:`fkiraly`,
:user:`GuzalBulatova`,
:user:`kcc-lion`,
:user:`khrapovs`,
:user:`lmmentel`,
:user:`MatthewMiddlehurst`,
:user:`miraep8`,
:user:`shagn`,
:user:`TonyBagnall`,
:user:`topher-lo`

Version 0.13.4 - 2022-09-27
---------------------------

Maintenance release - moved ``sktime`` repository to ``sktime`` org from ``alan-turing-institute`` org (:pr:`2926`)

Forks and links should be redirected, governance remains unchanged.

In case of any problems, please contact us via the `issue tracker <https://github.com/sktime/sktime/issues>`_ or `discussion forum <https://github.com/sktime/sktime/discussions>`_.

Version 0.13.3 - 2022-09-25
---------------------------

Highlights
~~~~~~~~~~~~

* new DL based time series classifiers: ``FCNClassifier``, ``MLPClassifier`` (:pr:`3232`, :pr:`3233`) :user:`AurumnPegasus`
* new transformers: Fourier features, DOBIN basis features (:pr:`3373`, :pr:`3374`) :user:`KatieBuc`, :user:`ltsaprounis`
* new annotation estimators: GGS, HIDAlgo, STRAY (:pr:`2744`, :pr:`3158`, :pr:`3338`) :user:`lmmentel`, :user:`KatieBuc`
* annotation: ``hmmlearn`` interface (:pr:`3362`) :user:`miraep8`
* fully documented tags in forecaster and transformer extension templates (:pr:`3334`, :pr:`3440`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``sktime`` is now compatible with ``pmdarima 2.0.0``, bound is relaxed to ``<3.0.0``
* ``sktime`` is now compatible with ``pandas 1.5.0``, bound is relaxed to ``<1.6.0``

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``datatypes.check_is_scitype``: 2nd return argument (only returned if ``return_metadata=True``)
will be changed from ``list`` to ``dict`` format (see docstring).
``list`` format will be deprecated from 0.14.0, and replaced by ``dict`` in 0.15.0.
The format will be determined by temporary additional arg ``msg_legacy_interface``, which will be
introduced in 0.14.0, default changed to ``False`` in 0.15.0, and removed in 0.16.0.

Enhancements
~~~~~~~~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] support for ``xarray`` ``DataArray`` & mtypes (:pr:`3255`) :user:`benHeid`
* [ENH] avoid metadata computation in ``scitype`` utility (:pr:`3357`) :user:`fkiraly`
* [ENH] ``check_is_scitype`` error message return changed to ``dict`` (:pr:`3466`) :user:`fkiraly`
* [ENH] soft dependency handling for mtypes (:pr:`3408`) :user:`fkiraly`
* [ENH] Optimize ``from_3d_numpy_to_nested`` converter function (:pr:`3339`) :user:`paulbauriegel`
* [ENH] simplify ``convert_to_scitype`` logic, fix export and docstring omissions in scitype converter module (:pr:`3358`) :user:`fkiraly`

Data loaders
^^^^^^^^^^^^

* [ENH] test for correct return type of ``load_basic_motions`` (:pr:`3458`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] ``pmdarima 2.0.0`` compatibility fix - use absolute index in return (:pr:`3302`) :user:`fkiraly`
* [ENH] global/local setting for ``DirectReductionForecaster`` (:pr:`3327`) :user:`fkiraly`
* [ENH] consistent ``sp`` handling in parameter estimators and ``AutoARIMA`` (:pr:`3367`) :user:`fkiraly`
* [ENH] enable default ``get_fitted_params`` for forecasters and delegated estimators (:pr:`3381`) :user:`fkiraly`
* [ENH] prevent vectorization in forecaster multiplexer (:pr:`3391`) :user:`fkiraly`
* [ENH] prevent vectorization in update wrappers and ``ForecastX`` (:pr:`3393`) :user:`fkiraly`
* [ENH] added missing data input check in forecasters (:pr:`3405`) :user:`fkiraly`
* [ENH] Add parallel ``fit`` and ``predict_residuals`` for calculation of ``residuals_matrix`` in ``ConformalIntervals`` (:pr:`3414`) :user:`bethrice44`
* [ENH] predictive variance and quantiles for naive forecaster (:pr:`3435`) :user:`topher-lo`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Greedy Gaussian Segmentation (:pr:`2744`) :user:`lmmentel`
* [ENH] HIDAlgo annotation (:pr:`3158`) :user:`KatieBuc`
* [ENH] ``hmmlearn`` interface (:pr:`3362`) :user:`miraep8`
* [ENH] STRAY anomaly detection (:pr:`3338`) :user:`KatieBuc`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Dictionary classifiers speedup (:pr:`3216`, :pr:`3360`) :user:`patrickzib`
* [ENH] new classifier: ``MLPClassifier`` (:pr:`3232`) :user:`AurumnPegasus`
* [ENH] new classifier: ``FCNClassifier`` (:pr:`3233`) :user:`AurumnPegasus`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENG] Weights in scipy distance (:pr:`1940`) :user:`stepinski`
* [ENH] distance features transformer (:pr:`3356`) :user:`fkiraly`
* [ENH] signature kernel from (Kiraly et al, 2016) (:pr:`3355`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] option to keep column names in ``Lag`` (:pr:`3343`) :user:`fkiraly`
* [ENH] ``BaseTransformer`` data memory - enabled by tag (:pr:`3307`) :user:`fkiraly`
* [ENH] Fourier features transformer (:pr:`3374`) :user:`ltsaprounis`
* [ENH] prevent vectorization in tramsformer multiplexer (:pr:`3391`) :user:`fkiraly`
* [ENH] added ``scale``, ``offset`` parameters to ``LogTransformer`` (:pr:`3354`) :user:`bugslayer-332`
* [ENH] ``pandas 1.5.0`` compatibility fix: use ``infer_freq`` in ``Lag`` if no ``freq`` passed or specified (:pr:`3456`) :user:`fkiraly`
* [ENH] refactor inheritance of ``PAA``, ``SAX``, ``SFA`` (:pr:`3308`) :user:`fkiraly`
* [ENH] DOBIN basis transformation (:pr:`3373`) :user:`KatieBuc`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] testing transformers with ``transform`` data different from ``fit`` data (:pr:`3341`) :user:`fkiraly`
* [ENH] reduce legacy logic in test framework and refactor to scenarios (:pr:`3342`) :user:`fkiraly`
* [ENH] second param sets for selected estimators (:pr:`3428`) :user:`fkiraly`

Fixes
~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] ensure ``nested_univ`` metadata inference passes for scalar columns present (:pr:`3463`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] Fix default conformal intervals ``initial_window`` parameter (:pr:`3383`) :user:`bethrice44`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fixing HMM last read bug (:pr:`3366`) :user:`miraep8`
* [BUG] Fix for hmm sporadic test failure (:pr:`3396`) :user:`miraep8`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fixes missing ``super.__init__`` call in ``MLPNetwork`` (:pr:`3350`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fixes incorrect warning condition in ``InvertTransform`` (:pr:`3352`) :user:`fkiraly`
* [BUG] ensure ``Differencer`` always inverts properly (:pr:`3346`) :user:`fkiraly`, :user:`ilkersigirci`

Maintenance
~~~~~~~~~~~

* [MNT] skip ``CNNClassifier`` doctest (:pr:`3305`) :user:`fkiraly`
* [MNT] Retry url request after HTTPError (:pr:`3242`) :user:`khrapovs`
* [MNT] skip ``ClearSky`` doctest to avoid ``load_solar`` crash (:pr:`3376`) :user:`fkiraly`
* [MNT] skip sporadic failure in testing ``HMM`` (:pr:`3395`) :user:`fkiraly`
* [MNT] isolate soft dependency in ``MLPClassifier`` doctest (:pr:`3409`) :user:`fkiraly`
* [MNT] Small refactoring changes (:pr:`3418`) :user:`lmmentel`
* [MNT] replaces deprecated ``pandas`` ``is_monotonic`` by ``is_monotonic_increasing`` (:pr:`3455`) :user:`fkiraly`
* [MNT] update ``test_interpolate`` to be ``pandas 1.5.0`` compatible (:pr:`3467`) :user:`fkiraly`
* [MNT] ``pandas 1.5.0`` compatibility (:pr:`3457`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] updated extension templates - tags explained, soft dependencies (:pr:`3334`) :user:`fkiraly`
* [DOC] API reference for ``dists_kernels`` module (:pr:`3312`) :user:`fkiraly`
* [DOC] fix notebook/example symlinks (:pr:`3379`) :user:`khrapovs`
* [DOC] Some tips on getting virtual environments to work (:pr:`3331`) :user:`miraep8`
* [DOC] changed wrong docstring default value of ``start_with_window`` in ``SlidingWindowSplitter`` to actual default value (:pr:`3340`) :user:`bugslayer-332`
* [DOC] Correct minor typos in ``examples/AA_datatypes_and_datasets.ipynb`` (:pr:`3349`) :user:`achieveordie`
* [DOC] updated extension templates - transformer tags explained (:pr:`3377`) :user:`fkiraly`
* [DOC] correcting and clarifying ``BaseSplitter`` docstrings (:pr:`3440`) :user:`fkiraly`
* [DOC] Fix docstring of TransformerPipeline (:pr:`3401`) :user:`aiwalter`
* [DOC] Expired slack link under "Where to ask questions" (:pr:`3449`) :user:`topher-lo`
* [DOC] Instructions for how to skip tests for new soft dependencies. (:pr:`3416`) :user:`miraep8`
* [DOC] replace legacy estimator overview with links (:pr:`3407`) :user:`fkiraly`
* [DOC] Update core dev list (:pr:`3415`) :user:`aiwalter`
* [DOC] Expired slack link under "Where to ask questions" (:pr:`3449`) :user:`topher-lo`
* [DOC] Added example to ``plot_series`` & fixed example for ``plot_lags`` (:pr:`3400`) :user:`shagn`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`aiwalter`,
:user:`AurumnPegasus`,
:user:`benHeid`,
:user:`bethrice44`,
:user:`bugslayer-332`,
:user:`fkiraly`,
:user:`ilkersigirci`,
:user:`KatieBuc`,
:user:`khrapovs`,
:user:`lmmentel`,
:user:`ltsaprounis`,
:user:`miraep8`,
:user:`patrickzib`,
:user:`paulbauriegel`,
:user:`shagn`,
:user:`stepinski`,
:user:`topher-lo`

Version 0.13.2 - 2022-08-23
---------------------------

Highlights
~~~~~~~~~~

* new forecaster: ``statsmodels`` ``ARDL`` interface (:pr:`3209`) :user:`kcc-lion`
* new transformer: channel/variable selection (Dhariyal et al 2021) for multivariate time series classification (:pr:`3248`) :user:`haskarb`
* new dunders: ``trafo ** forecaster`` = apply to exogeneous data; ``-trafo`` = ``OptionalPassthrough``; ``~trafo`` = invert (:pr:`3243`, :pr:`3273`, :pr:`3274`) :user:`fkiraly`
* pairwise transformations (time series distances, kernels) are now fully integrated with the ``check_estimator`` utility (:pr:`3254`) :user:`fkiraly`

Dependency changes
~~~~~~~~~~~~~~~~~~

* ``pmdarima`` is bounded ``<2.0.0`` until compatibility issues are resolved

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

Forecasting
^^^^^^^^^^^

* dunder method for pipelining transformers to exogeneous data: ``my_trafo ** my_forecaster`` will create a ``ForecastingPipeline``
  Note: ``**`` has precedence over ``*`` (apply to endogeneous data)
* the default value for the ``ignores-exogeneous-X`` tag is set to the safer value ``False``.
  This does not affect ``sktime`` forecasters, but may affect ``sktime`` compatible forecasters
  in which an explicit setting of the tag has been omitted, in that ``X`` is now passed to all internal functions ``_fit``, ``predict``, etc.
  This is breaking only under the condition that (a) the tag has been erroneously omitted, (b) the internal functions are broken,
  i.e., will cause an exception only if the error (a) was masking a bug (b).

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* dunder method for pipelining ordinary transformers with pairwise transformers: ``my_trafo ** distance``
  will create a ``PwTrafoPanelPipeline``, same as "apply ``my_trafo.fit_transform`` to both inputs first, then apply ``distance``"

Transformations
^^^^^^^^^^^^^^^

* dunder method for applying ``OptionalPassthrough``: ``-my_trafo`` is the same as ``OptionalPassthrough(my_trafo)``
* dunder method for inverting transformer: ``~my_trafo`` has ``transform`` and ``inverse_transform`` switched

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Transformations
^^^^^^^^^^^^^^^

* deprecated: ``transformations.series.compose`` is deprecated in favour of ``transformations.compose``.
  All estimators in the former are moved to the latter, and will no longer be accessible in ``transformations.series.compose`` from 0.15.0.
* deprecated: the row transformers, ``SeriesToSeriesRowTransformer`` and ``SeriesToPrimitivesRowTransformer`` have been deprecated.
  Row/instance vectorization functionality is natively supported by ``sktime`` since 0.11.0 and does not need to be added by these wrappers anymore.
  Both transformers will be removed in 0.15.0. To migrate, simply remove the row transformer wrappers.
  In some rarer, ambiguous vectorization cases (e.g., using wrapped functions that are vectorized, such as ``np.mean``),
  ``FunctionTransformer`` may have to be used instead of ``SeriesToPrimitivesRowTransformer``.


Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] robustify ``BaseObject.set_tags`` against forgotten ``__init__`` (:pr:`3226`) :user:`fkiraly`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] treat non nested cols in conversion ``nested_univ`` to ``pd-multiindex`` (:pr:`3250`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [ENH] ``statsmodels`` ``ARDL`` interface (:pr:`3209`) :user:`kcc-lion`
* [ENH] ``**`` dunder for applying transformers to exogeneous data in forecasters (:pr:`3243`) :user:`fkiraly`
* [ENH] test ``pd.Series`` with name attribute in forecasters (:pr:`3297`, :pr:`3323`) :user:`fkiraly`
* [ENH] set default ``ignores-exogeneous-X`` to ``False`` (:pr:`3260`) :user:`fkiraly`
* [ENH] forecasting pipeline test case with ``Detrender`` (:pr:`3270`) :user:`fkiraly`
* [ENH] test hierarchical forecasters with hierarchical data (:pr:`3321`) :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Data generator for annotation - normal multivariate mean shift (:pr:`3114`) :user:`KatieBuc`

Time series distances and kernels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] MSM distance clean-up (:pr:`2964`) :user:`chrisholder`
* [ENH] panel distance from flattened tabular distance (:pr:`3249`) :user:`fkiraly`
* [ENH] test class integration for pairwise transformers (:pr:`3254`) :user:`fkiraly`
* [ENH] expose edit distances as sklearn compatible objects (:pr:`3251`) :user:`fkiraly`
* [ENH] pipeline composition for pairwise panel transformers (:pr:`3263`) :user:`fkiraly`
* [ENH] arithmetic combinations of distances/kernel transformers (:pr:`3264`) :user:`fkiraly`
* [ENH] constant distance dummy (:pr:`3266`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] channel selection (Dhariyal et al 2021) for multivariate time series classification (:pr:`3248`) :user:`haskarb`
* [ENH] channel selection (Dhariyal et al 2021) - compatibility with arbitrary distance (:pr:`3256`) :user:`fkiraly`
* [ENH] in ``Lag``, make column naming consistent between single-lag and multi-lag case (:pr:`3261`) :user:`KishManani`
* [ENH] deprecate ``transformations.series.compose`` in favour of ``transformations.compose`` (:pr:`3271`) :user:`fkiraly`
* [ENH] inversion of transformer wrapper and dunder (:pr:`3274`) :user:`fkiraly`
* [ENH] correctness test for ``OptionalPassthrough`` (:pr:`3276`) :user:`aiwalter`
* [ENH] ``OptionalPassthrough`` wrapping via ``neg`` dunder (:pr:`3273`) :user:`fkiraly`
* [ENH] refactor of ``OptionalPassthrough`` as a delegator (:pr:`3272`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] test ``super.__init__`` call in objects and estimators (:pr:`3309`) :user:`fkiraly`

Governance
^^^^^^^^^^

* [GOV] ``sktime`` as a "library", not a "curated selection" (:pr:`3155`) :user:`fkiraly`


Fixes
~~~~~

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] Fix ``write_ndarray_to_tsfile`` for ``classLabel = False`` (:pr:`3303`) :user:`paulbauriegel`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix failure of some conversions in ``_load_provided_dataset()`` (:pr:`3231`) :user:`achieveordie`
* [BUG] fix recurring instances of forgotten list comprehension brackets inside ``np.all`` (:pr:`3245`) :user:`achieveordie`, :user:`fkiraly`
* [BUG] fix ``_enforce_infer_freq`` private utility for short time series (:pr:`3287`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] Delay trimming in ``ForecastingGridSearchCV until`` after transforming (:pr:`3132`) :user:`miraep8`
* [BUG] Fix tag in ``DirectReductionForecaster`` (:pr:`3257`) :user:`KishManani`
* [BUG] ensure that forecasters do not add ``pd.Series.name`` attribute (:pr:`3290`) :user:`fkiraly`
* [BUG] removes superfluous ``UserWarning`` in ``AutoETS.fit`` if ``auto=True`` and ``additive_only=True`` #3311 (:pr:`3317`) :user:`chillerobscuro`
* [BUG] fix ``ColumnEnsembleForecaster`` for hierarchical input (:pr:`3324`) :user:`fkiraly`
* [BUG] fix bug where default forecaster ``_update`` empties converter store (:pr:`3325`) :user:`fkiraly`
* [BUG] (temporary fix) remove hierarchical datatypes from recursive reduction forecasters (:pr:`3326`) :user:`fkiraly`

Parameter estimation
^^^^^^^^^^^^^^^^^^^^

* [BUG] fixed concat dunder for ``ParamFitterPipeline`` (:pr:`3262`) :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [BUG] ClaSP Segmentation fixes (:pr:`3217`) :user:`patrickzib`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

Transformations
^^^^^^^^^^^^^^^

* [BUG] fix ``Deseasonalizer._update`` (:pr:`3268`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] Deprecation of row transformers (:pr:`2370`) :user:`fkiraly`
* [MNT] add soft dependency tag to ``CNNClassifier`` (:pr:`3252`) :user:`fkiraly`
* [MNT] bound ``pmdarima < 2.0.0`` (:pr:`3301`) :user:`fkiraly`
* [MNT] fix merge accident that deleted ``DtwDist`` export (:pr:`3304`) :user:`fkiraly`
* [MNT] move transformers in ``transformations.series.compose`` to ``transformations.compose`` (:pr:`3310`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`aiwalter`,
:user:`chillerobscuro`,
:user:`chrisholder`,
:user:`fkiraly`,
:user:`haskarb`,
:user:`KatieBuc`,
:user:`kcc-lion`,
:user:`KishManani`,
:user:`miraep8`,
:user:`patrickzib`,
:user:`paulbauriegel`

Version 0.13.1 - 2022-08-11
---------------------------

Highlights
~~~~~~~~~~

* forecasting reducers constructed via ``make_reduction`` now fully support global/hierarchical forecasting (:pr:`2486`) :user:`danbartl`
* forecasting metric classes now fully support hierarchical data and hierarchy averaging via ``multilevel`` argument (:pr:`2601`) :user:`fkiraly`
* probabilistic forecasting functionality for ``DynamicFactor``, ``VAR`` and ``VECM`` (:pr:`2925`, :pr:`3105`) :user:`AurumnPegasus`, :user:`lbventura`
* ``update`` features for ``AutoARIMA``, ``BATS``, ``TBATS``, and forecasting tuners (:pr:`3055`, :pr:`3068`, :pr:`3086`) :user:`fkiraly`, :user:`jelc53`
* new transformer: ``ClearSky`` transformer for solar irradiance time series (:pr:`3130`) :user:`ciaran-g`
* new transformer: ``Filter`` transformer for low/high-pass and band filtering, interfaces ``mne`` ``filter_data`` (:pr:`3067`) :user:`fkiraly`, :user:`sveameyer13`

Dependency changes
~~~~~~~~~~~~~~~~~~

* new soft dependency ``mne``, from ``Filter`` transformer
* new developer dependency ``pytest-randomly``

Core interface changes
~~~~~~~~~~~~~~~~~~~~~~

All Estimators
^^^^^^^^^^^^^^

* ``get_fitted_params`` now has a private implementer interface ``_get_fitted_params``, similar to ``fit`` / ``_fit`` etc
* the undocumented ``_required_parameters`` parameter is no longer required (to be present in certain estimators)

Forecasting
^^^^^^^^^^^

* forecasting metric classes now fully support hierarchical data and hierarchy averaging via ``multilevel`` argument

Parameter estimation
^^^^^^^^^^^^^^^^^^^^

* new estimator type - parameter estimators, base class ``BaseParamFitter``

Deprecations and removals
~~~~~~~~~~~~~~~~~~~~~~~~~

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* ``ProbabilityThresholdEarlyClassifier`` has been deprecated and will be replaced by an early classifier of the same name in version 0.15.0.
    Interfaces will not be downwards compatible.

Enhancements
~~~~~~~~~~~~

BaseObject
^^^^^^^^^^

* [ENH] remove custom ``__repr__`` from ``BaseTask``, inherit from ``BaseObject`` (:pr:`3049`) :user:`fkiraly`
* [ENH] default implementation for ``get_fitted_params`` and nested fitted params interface (:pr:`3077`) :user:`fkiraly`
* [ENH] remove ``_required_parameters`` interface point from ``BaseObject`` (:pr:`3152`) :user:`fkiraly`

Data sets and data loaders
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] ensure unique instance index in ``sktime`` datasets (:pr:`3029`) :user:`fkiraly`
* [ENH] Rework of data loaders (:pr:`3109`) :user:`achieveordie`

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] add check for unique column indices to mtype checks (:pr:`2971`) :user:`fkiraly`
* [ENH] Adapter from ``pd-multiindex`` to ``gluonts`` ``ListDataset`` (:pr:`2976`) :user:`TNTran92`
* [ENH] add check for non-duplicate indices in ``nested_univ`` mtype (:pr:`3029`) :user:`fkiraly`
* [BUG] Remove redundant computations in ``datatypes._utilities.get_cutoff`` (:pr:`3070`) :user:`shchur`

Forecasting
^^^^^^^^^^^

* [ENH] Reworked ``make_reduction`` for global forecasting (:pr:`2486`) :user:`danbartl`
* [ENH] flexible ``update`` behaviour of forecasting tuners (:pr:`3055`) :user:`fkiraly`
* [ENH] flexible ``update`` behaviour of ``AutoARIMA`` (:pr:`3068`) :user:`fkiraly`
* [ENH] Reducer prototype rework - experimental (:pr:`2833`) :user:`fkiraly`
* [ENH] better ``ForecastingHorizon`` construction error message (:pr:`3236`) :user:`fkiraly`
* [ENH] metrics rework part IV - hierarchical metrics (:pr:`2601`) :user:`fkiraly`
* [ENH] Reducer prototype rework - experimental (:pr:`2833`) :user:`fkiraly`
* [ENH] ``predict_interval`` capability for ``VECM`` (:pr:`2925`) :user:`AurumnPegasus`
* [ENH] "dont refit or update" option in ``evaluate`` (:pr:`2954`) :user:`fkiraly`
* [ENH] regular update for stream forecasting, and "no update" wrappers (:pr:`2955`) :user:`fkiraly`
* [ENH] Implement ``get_fitted_params`` for tuning forecasters  (:pr:`2975`) :user:`ZiyaoWei`
* [ENH] allow ``sp=None`` in the ``NaiveForecaster`` (:pr:`3043`) :user:`fkiraly`
* [MNT] remove custom ``__repr__`` from ``BaseSplitter`` (:pr:`3048`) :user:`fkiraly`
* [ENH] dedicated ``update`` for ``BATS`` and ``TBATS`` (:pr:`3086`) :user:`jelc53`
* [ENH] ``DynamicFactor`` ``predict_interval`` and ``predict_quantiles`` (:pr:`3105`) :user:`lbventura`
* [ENH] Added ``error_score`` to ``evaluate`` and forecasting tuners (:pr:`3135`) :user:`aiwalter`
* [ENH] Refactor ``CutoffSplitter`` using ``get_window`` function (:pr:`3145`) :user:`khrapovs`
* [ENH] Refactor ``SingleWindowSplitter`` using ``get_window`` function (:pr:`3146`) :user:`khrapovs`
* [ENH] Allow lists to be ``cutoff`` argument in ``CutoffSplitter`` (:pr:`3147`) :user:`khrapovs`
* [ENH] Adding ``VAR._predict_intervals`` (:pr:`3149`) :user:`lbventura`

Parameter estimation
^^^^^^^^^^^^^^^^^^^^

* [ENH] Parameter estimators and "plug in parameter" compositors (:pr:`3041`) :user:`fkiraly`

Time series annotation
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] HMM annotation estimator (:pr:`2855`) :user:`miraep8`
* [ENH] Data generator for annotation (:pr:`2996`) :user:`lmmentel`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [ENH] refactored ``KNeighborsTimeSeriesClassifier`` (:pr:`1998`) :user:`fkiraly`
* [ENH] move vector classifiers from ``_contrib`` to classification module (:pr:`2951`) :user:`MatthewMiddlehurst`
* [ENH] Various improvements to CNN Classifier base class (:pr:`2991`) :user:`AurumnPegasus`
* [ENH] weighted ensemble compositor for classifiers to allow users to build their own HIVE-COTE like ensembles (:pr:`3036`) :user:`fkiraly`
* [ENH] classifier ``fit_predict`` methods and default ``_predict`` (:pr:`3038`) :user:`fkiraly`
* [ENH] remove unused methods from ``ClassifierPipeline`` (:pr:`3042`) :user:`fkiraly`
* [ENH] refactor ``RocketClassifier`` to pipeline delegate (:pr:`3102`) :user:`fkiraly`
* [ENH] refactor ``Catch22Classifier`` to pipeline delegate (:pr:`3112`) :user:`fkiraly`
* [ENH] classifier runtime profiling utility (:pr:`3076`) :user:`fkiraly`
* [ENH] deprecate ``ProbabilityThresholdEarlyClassifier`` (:pr:`3133`) :user:`MatthewMiddlehurst`
* [ENH] classifier single class handling (:pr:`3140`) :user:`fkiraly`
* [ENH] classification evaluation utility (:pr:`3173`) :user:`TNTran92`

Time series regression
^^^^^^^^^^^^^^^^^^^^^^

* [ENH] Adding ``CNNRegressor`` and ``BaseDeepRegressor`` (:pr:`2902`) :user:`AurumnPegasus`
* [ENH] ``RocketRegressor`` (:pr:`3126`) :user:`fkiraly`
* [ENH] regressor pipelines, regressor delegators (:pr:`3126`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [ENH] refactored ``ColumnConcatenator``, rewrite using ``pd-multiindex`` inner mtype (:pr:`2379`) :user:`fkiraly`
* [ENH] ``__getitem__`` aka ``[ ]`` dunder for transformers, column subsetting (:pr:`2907`) :user:`fkiraly`
* [ENH] ``YtoX`` transformer to use transform endogeneous data as exogegneous (:pr:`2922`) :user:`fkiraly`
* [BUG] fixes ``RandomIntervalFeatureExtractor`` to have unique column names (:pr:`3001`) :user:`fkiraly`
* [BUG] fix for ``Differencer.inverse_transform`` not having access to data index ``freq`` (:pr:`3007`) :user:`fkiraly`
* [ENH] Refactor transformers in ``_deseasonalize`` module (:pr:`3040`) :user:`fkiraly`
* [ENH] ``Filter`` transformer from ``sktime-neuro`` (:pr:`3067`) :user:`fkiraly`
* [ENH] increase stateless scope of ``FunctionTransformer`` and ``TabularToSeriesAdaptor`` (:pr:`3087`) :user:`fkiraly`
* [ENH] ``ClearSky`` transformer for solar irradiance time series (:pr:`3130`) :user:`ciaran-g`
* [ENH] move simple ``ShapeletTransform`` from ``_contrib`` to ``transformations`` module (:pr:`3136`) :user:`fkiraly`

Testing framework
^^^^^^^^^^^^^^^^^

* [ENH] test that transformer output columns are unique (:pr:`2969`) :user:`fkiraly`
* [ENH] test estimator ``fit`` without soft dependencies (:pr:`3039`) :user:`fkiraly`
* [ENH] test all ``BaseObject`` descendants for sklearn compatibility (:pr:`3122`) :user:`fkiraly`
* [ENH] ``functools`` wrapper to preserve docstrings in estimators wrapped by ``make_mock_estimator`` (:pr:`3228`) :user:`ltsaprounis`
* [ENH] refactoring test params for ``FittedParamExtractor`` to ``get_test_params`` (:pr:`2995`) :user:`mariamjabara`
* [ENH] refactored test params for ``ColumnTransformer`` (:pr:`3008`) :user:`kcc-lion`
* [ENH] complete refactor of all remaining test params left in ``_config`` to ``get_test_params`` (:pr:`3123`) :user:`fkiraly`
* [ENH] partition design for test matrix to reduce test time to a third (:pr:`3137`) :user:`fkiraly`

Documentation
~~~~~~~~~~~~~

* [DOC] expanding content in testing section of "adding estimator" developer docs (:pr:`2544`) :user:`aiwalter`
* [DOC] add multivariate CNN example from ``sktime-dl`` (:pr:`3002`) :user:`tobiasweede`
* [DOC] parameter checking and move of ``super.__init__`` in extension templates (:pr:`3010`) :user:`fkiraly`
* [DOC] proba forecasting notebook from pydata Berlin 2022 (:pr:`3016`) :user:`ciaran-g`, :user:`eenticott-shell`, :user:`fkiraly`
* [DOC] added docstring example for ``make_reduction`` (:pr:`3054`) :user:`aiwalter`
* [DOC] fix typo in ``segmentation_with_clasp.ipynb`` (:pr:`3060`) :user:`soma2000-lang`
* [DOC] improve splitters docstrings (:pr:`3075`) :user:`khrapovs`
* [DOC] code quality docs expanded with instructions for local code quality checking set-up (:pr:`3089`) :user:`fkiraly`
* [DOC] added NumFOCUS to sponsors website (:pr:`3093`) :user:`aiwalter`
* [DOC] added Python 3.10 reference to installation docs (:pr:`3098`) :user:`aiwalter`
* [DOC] improvements on local linting/precommit setup developer documentation (:pr:`3111`) :user:`C-mmon`
* [DOC] changed sktime logo on README (:pr:`3143`) :user:`aiwalter`
* [DOC] clarifications in the ``Deseasonalizer`` docstring (:pr:`3157`) :user:`fkiraly`
* [DOC] fix references (:pr:`3170`) :user:`aiwalter`
* [DOC] added docstring examples and cleaning (:pr:`3174`) :user:`aiwalter`
* [DOC] added more detail to step 4 of high-level steps to implementing an es… (:pr:`3200`) :user:`kcc-lion`
* [DOC] improved ``STLForecaster`` docstring (:pr:`3203`) :user:`fkiraly`
* [DOC] added notebook cell output for notebooks shown in website (:pr:`3215`) :user:`aiwalter`
* [DOC] hierarchical forecasting notebook from pydata London 2022 (:pr:`3227`) :user:`danbartl`, :user:`fkiraly`
* [DOC] cleaned up user docs and tutorials page (:pr:`3240`) :user:`fkiraly`

Fixes
~~~~~

Data types, checks, conversions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix stray args in one ``from_multi_index_to_3d_numpy`` (:pr:`3239`) :user:`fkiraly`

Forecasting
^^^^^^^^^^^

* [BUG] fix forecaster default ``predict_quantiles`` for multivariate data (:pr:`3106`) :user:`fkiraly`
* [BUG] ``ExpandingWindowSplitter`` constructor ``sklearn`` conformace fix (:pr:`3121`) :user:`fkiraly`
* [BUG] fix override/defaulting of "prediction intervals" adders  (:pr:`3129`) :user:`bethrice44`
* [BUG] fix ``check_equal_time_index`` with numpy arrays as input (:pr:`3160`, :pr:`3167`) :user:`benHeid`
* [BUG] fix broken ``AutoEnsembleForecaster`` inverse variance method (:pr:`3208`) :user:`AnH0ang`
* [BUG] fixing bugs in metrics base classes and custom performance metric (:pr:`3225`) :user:`fkiraly`

Time series classification
^^^^^^^^^^^^^^^^^^^^^^^^^^

* [BUG] fix HIVE-COTE2 sporadic test failure (:pr:`3094`) :user:`MatthewMiddlehurst`
* [BUG] fixes to ``BaseClassifier._predict_proba`` default and ``SklearnClassifierPipeline`` in case ``predict_proba`` is not implemented (:pr:`3104`) :user:`fkiraly`
* [BUG] allowing single class case in sklearn classifiers (trees/forests) (:pr:`3204`) :user:`fkiraly`
* [BUG] skip check for no. estimators in contracted classifiers (:pr:`3207`) :user:`fkiraly`

Transformations
^^^^^^^^^^^^^^^

* [BUG] fixed inverse transform logic in transformer pipelines (:pr:`3085`) :user:`fkiraly`
* [BUG] fixed ``DateTimeFeatures`` inconsistent output type formats (:pr:`3223`) :user:`danbartl`
* [BUG] fixed ``Datetimefeatures`` ``day_of_year`` option not working (:pr:`3223`) :user:`danbartl`

Testing framework
^^^^^^^^^^^^^^^^^

* [BUG] address shadowing of ``object`` in ``_check_soft_dependencies`` (:pr:`3116`) :user:`fkiraly`
* [BUG] prevent circular imports in ``all_estimators`` (:pr:`3198`) :user:`fkiraly`

Maintenance
~~~~~~~~~~~

* [MNT] removal of pytest pyargs in CI/CD (:pr:`2928`) :user:`fkiraly`
* [MNT] fix broken Slack invite links (:pr:`3017`, :pr:`3066`) :user:`aiwalter`, :user:`Arvind644`
* [MNT] removed ``hcrystalball`` from ``all_extras`` dependency set (:pr:`3091`) :user:`aiwalter`
* [MNT] cleaning up CI workflow (:pr:`2896`) :user:`lmmentel`
* [MNT] add ``testdir`` to ``.gitignore`` (:pr:`3019`) :user:`Lovkush-A`
* [MNT] xfail known sporadic test failure #2368 (:pr:`3030``) :user:`fkiraly`
* [MNT] Update codecov github action from v2 to v3 (:pr:`3050`) :user:`miraep8`
* [MNT] bump MacOS GitHub actions host to MacOS-11 (:pr:`3107`) :user:`lmmentel`
* [MNT] temporarily exclude ``RandomShapeletTransform`` from tests (:pr:`3139`) :user:`fkiraly`
* [MNT] temporary fix for Mac CI failures: skip recurringly failing estimators (:pr:`3134`) :user:`fkiraly`
* [MNT] reduce test verbosity (:pr:`3074`) :user:`lmmentel`
* [MNT] isolate soft dependencies (:pr:`3081`) :user:`fkiraly`
* [MNT] reduce expected test time by making tests conditional on ``no-softdeps`` (:pr:`3092`) :user:`fkiraly`
* [MNT] temporarily exclude ``RandomShapeletTransform`` from tests (:pr:`3139`) :user:`fkiraly`
* [MNT] restrict changelog generator to changes to main branch (:pr:`3168`) :user:`lmmentel`
* [MNT] skip known failure case for ``VARMAX`` (:pr:`3178`) :user:`fkiraly`
* [MNT] added ``pytest-randomly`` (:pr:`3187`) :user:`aiwalter`
* [MNT] updated social links and badges, added LinkedIn badge (:pr:`3195`) :user:`aiwalter`
* [MNT] reactivate tests for ``TSFreshRelevantFeatureExtractor`` (:pr:`3196`) :user:`fkiraly`

Contributors
~~~~~~~~~~~~

:user:`achieveordie`,
:user:`aiwalter`,
:user:`AnH0ang`,
:user:`Arvind644`,
:user:`AurumnPegasus`,
:user:`benHeid`,
:user:`bethrice44`,
:user:`C-mmon`,
:user:`ciaran-g`,
:user:`danbartl`,
:user:`eenticott-shell`,
:user:`fkiraly`,
:user:`jelc53`,
:user:`kcc-lion`,
:user:`khrapovs`,
:user:`lbventura`,
:user:`lmmentel`,
:user:`Lovkush-A`,
:user:`ltsaprounis`,
:user:`mariamjabara`,
:user:`MatthewMiddlehurst`,
:user:`miraep8`,
:user:`shchur`,
:user:`soma2000-lang`,
:user:`sveameyer13`,
:user:`TNTran92`,
:user:`tobiasweede`,
:user:`ZiyaoWei`

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

* [ENH] extend ``ColumnSelect`` to work for scalar ``columns`` parameter (:pr:`2906`) :user:`fkiraly`
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

* [ENH] Extended sliding and expanding window splitters to allow timedelta forecasting horizon (:pr:`2551`) :user:`khrapovs`
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

* [BUG] fixing direct conversions from/to ``numpyflat`` mtype being overridden by indirect ones (:pr:`2517`) :user:`fkiraly`

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
* pipeline, tuning and evaluation compatibility for probabilistic forecasting (:pr:`2234`, :pr:`2318`) :user:`eenticott-shell` :user:`fkiraly`
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
* [DOC] typo fix constructor -> constructor in extension templates (:pr:`2348`) :user:`fkiraly`
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
* for extenders: detailed `"creating sktime compatible estimator" guide <https://www.sktime.net/en/stable/developer_guide/add_estimators.html>`_
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
*  [BUG] ``_make_hierarchical`` is renamed to ``_make_hierarchical`` (typo/bug) issue #2195 (:pr:`2196`) :user:`Vasudeva-bit`
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
* [ENH] Test enhancements documentation (:pr:`1922`) :user:`fkiraly`
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
* [MNT] Move all the CI/CD workflows over to github actions and drop azure pipelines and appveyor (:pr:`1620`, :pr:`1920`) :user:`lmemntel`
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
https://github.com/sktime/sktime/compare/v0.8.1...v0.9.0


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
* [MNT] Deprecate manylinux2010 (:pr:`1379`) :user:`mloning`
* [MNT] Added pre-commit hook to sort imports (:pr:`1465`) :user:`aiwalter`
* [MNT] add :code:`max_requirements`, bound statsmodels (:pr:`1479`) :user:`fkiraly`
* [MNT] Hotfix tag scitype:y typo (:pr:`1449`) :user:`aiwalter`
* [MNT] Add :code:`pydocstyle` to precommit (:pr:`890`) :user:`mloning`

* [BUG] incorrect/missing weighted geometric mean in forecasting ensemble (:pr:`1370`) :user:`fkiraly`
* [BUG] :pr:`1469`: stripping names of index X and y  (:pr:`1493`) :user:`boukepostma`
* [BUG] W-XXX frequency bug from :pr:`866` (:pr:`1409`) :user:`xiaobenbenecho`
* [BUG] Pandas.NA for unpredictable insample forecasts in AutoARIMA (:pr:`1442`) :user:`IlyasMoutawwakil`
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
* [DOC] major overhaul of :code:`sktime`'s `online documentation <https://www.sktime.net/en/latest/>`_
* [DOC] `searchable, auto-updating estimators register <https://www.sktime.net/en/latest/estimator_overview.html>`_ in online documentation (:pr:`930` :pr:`1138`) :user:`afzal442` :user:`mloning`
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

* introduction of m(achine)types and scitypes for defining in-memory format conventions across all modules, see `in-memory data types tutorial <https://github.com/sktime/sktime/blob/main/examples/AA_datatypes_and_datasets.ipynb>`_
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
* [DOC] naive foreasting docstring edits (:pr:`1333`) :user:`AreloTanoh`
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
