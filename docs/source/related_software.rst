.. _related_software:

================
Related Software
================

The Python ecosystem contains numerous packages that can be used to store
and process time series data. The following list is by no means exhaustive.
If you miss anything, feel free to open a `PR <https://github.com/sktime/sktime/edit/main/docs/source/related_software.rst>`_.

**Table of contents**

* `Packages for machine learning, statistics or analytics with time series <#machine-learning>`_,
* `Other time series related frameworks or database systems <#time-series-databases-and-frameworks>`_.

For time series data containers, see `our wiki entry <https://github.com/sktime/sktime/wiki/Time-series-data-container>`_.

Machine learning
================

Libraries
---------

.. list-table::
   :header-rows: 1

   * - Project Name
     - Description
   * - `adtk <https://github.com/arundo/adtk>`_
     - Anomaly Detection Tool Kit, a Python package for unsupervised/rule-based time series anomaly detection
   * - `atspy <https://github.com/firmai/atspy>`_
     - Collection of forecasting models, wraps existing statistical and machine learning models for forecasting, automated benchmarking
   * - `Arrow <https://github.com/crsmithdev/arrow>`_
     - A sensible, human-friendly approach to creating, manipulating, formatting and converting dates, times, and timestamps
   * - `cesium <https://github.com/cesium-ml/cesium>`_
     - Time series platform with feature extraction aiming for non uniformly sampled signals
   * - `catch22 <https://github.com/chlubba/op_importance>`_
     - Feature selection pipeline for `hctsa <https://github.com/benfulcher/hctsa>`_ and the so-called catch22 feature set
   * - `crystalball <https://github.com/heidelbergcement/hcrystalball>`_
     - A python library for forecasting with scikit-learn like API
   * - `darts <https://github.com/unit8co/darts>`_
     - Python collection of time series forecasting tools, from preprocessing to models (uni-/multivariate, prophet, neural networks) and backtesting utilities
   * - `deeptime <https://github.com/deeptime-ml/deeptime>`_
     - Library for unsupervised learning with time series including dimensionality reduction, clustering, and Markov model estimation
   * - `deltapy <https://github.com/firmai/deltapy>`_
     - Collection of data augmentation tools, including feature extraction from time series, wraps existing toolkits like tsfresh
   * - `diviner <https://github.com/databricks/diviner>`_
     - Diviner by Databricks enables large-scale time series forecasting and serves as a wrapper around other open source forecasting libraries
   * - `dtaidistance <https://github.com/wannesm/dtaidistance>`_
     - Time series distances
   * - `dtw <https://github.com/pierre-rouanet/dtw>`_
     - Scipy-based dynamic time warping
   * - `Featuretools <https://github.com/Featuretools/featuretools>`_
     - Time series feature extraction, with possible conditionality on other variables with a pandas compatible relational-database-like data container
   * - `fecon235 <https://github.com/rsvp/fecon235>`_
     - Computational tools for financial economics
   * - `ffn <https://github.com/pmorissette/ffn>`_
     - financial function library
   * - `flint <https://github.com/twosigma/flint>`_
     - A Time Series Library for Apache Spark
   * - `GENDIS <https://github.com/IBCNServices/GENDIS>`_
     - Shapelet discovery by genetic algorithms
   * - `glm-sklearn <https://github.com/jcrudy/glm-sklearn>`_
     - scikit-learn compatible wrapper around the GLM module in `statsmodels <https://github.com/statsmodels/statsmodels>`_
   * - `gluon-ts <https://github.com/awslabs/gluon-ts>`_
     - Probabilistic forecasting and anomaly detection using deep learning by Amazon
   * - `greykite <https://github.com/linkedin/greykite>`_
     - A Time Series Library for rorecasting by LinkedIn which contains the interpretable "Silverkite" algorithm.
   * - `hctsa <https://github.com/benfulcher/hctsa>`_
     - Matlab based feature extraction which can be controlled from python
   * - `HMMLearn <https://github.com/hmmlearn/hmmlearn>`_
     - Hidden Markov Models with scikit-learn compatible API
   * - `kats <https://github.com/facebookresearch/kats>`_
     - A toolkit by Facebook for time series analysis, including detection, forecasting, feature extraction/embedding, multivariate analysis, etc.
   * - `khiva-python <https://github.com/shapelets/khiva-python>`_
     - A Time Series library with accelerated analytics on GPUS, it provides feature extraction and motif discovery among other functionalities.
   * - `lifelines <https://github.com/CamDavidsonPilon/lifelines>`_
     - Toolkit for survival analysis
   * - `linearmodels <https://github.com/bashtage/linearmodels/>`_
     - Add linear models including instrumental variable and panel data models that are missing from statsmodels.
   * - `loudML <https://github.com/regel/loudml>`_
     - Time series inference engine built on top of TensorFlow to forecast data, detect outliers, and automate your process using future knowledge.
   * - `matrixprofile-ts <https://github.com/target/matrixprofile-ts>`_
     - A Python library for detecting patterns and anomalies in massive datasets using the Matrix Profile
   * - `mcfly <https://mcfly.readthedocs.io/en/latest/>`_
     - Deep learning for time series classification with automated hyperparameter selection
   * - `neuralprophet <https://github.com/ourownstory/neural_prophet>`_
     - A Neural Network based model, inspired by Facebook Prophet and AR-Net, built on PyTorch
   * - `Nitime <https://github.com/nipy/nitime>`_
     - Time series analysis for neuroscience data
   * - `NoLiTSA <https://github.com/manu-mannattil/nolitsa>`_
     - Non-linear time series analysis
   * - `orbit <https://github.com/uber/orbit>`_
     - Bayesian time series forecasting package by uber
   * - `pomegranate <https://pomegranate.readthedocs.io/en/latest/index.html>`_
     - Probabilistic models ranging from individual probability distributions to compositional models such as Bayesian networks and hidden Markov models.
   * - `Pastas <https://github.com/pastas/pastas>`_
     - Time series analysis for hydrological data
   * - `prophet <https://github.com/facebook/prophet>`_
     - Time series forecasting for time series data that has multiple seasonality with linear or non-linear growth
   * - `pyDSE <https://github.com/blue-yonder/pydse>`_
     - ARMA models for dynamic system Estimation
   * - `PyEMMA <https://github.com/markovmodel/PyEMMA>`_
     - Analysis of extensive molecular dynamics simulations based on Markov models
   * - `PyFlux <https://github.com/RJT1990/pyflux>`_
     - Classical time series forecasting models
   * - `PyHubs <https://sourceforge.net/projects/pyhubs/>`_
     - Hubness-aware machine learning in Python including time series classification via dynamic time warping based KNN classification
   * - `PyOD <https://github.com/yzhao062/pyod>`_
     - Toolbox for outlier detection
   * - `pysf <https://github.com/alan-turing-institute/pysf>`_
     - A scikit-learn compatible machine learning library for supervised/panel forecasting
   * - `pmdarima <https://github.com/tgsmith61591/pyramid>`_
     - Port of R's auto.arima method to Python
   * - `pyts <https://github.com/johannfaouzi/pyts>`_
     - Contains time series preprocessing, transformation as well as classification techniques
   * - `ruptures <https://github.com/deepcharles/ruptures>`_
     - time series annotation: change point detection, segmentation
   * - `salesforce-merlion <https://github.com/salesforce/Merlion/>`_
     - Library from salesforce for forecasting, anomaly detection, and change point detection
   * - `scikit-fda <https://github.com/GAA-UAM/scikit-fda>`_
     - A Python library to perform Functional Data Analysis, compatible with scikit-learn, including representation, preprocessing, exploratory analysis and machine learning methods
   * - `scikit-multiflow <https://scikit-multiflow.github.io>`_
     - Extension of scikit-learn to supervised learning of streaming data (dynamic online learning), including regression/classification and change detection
   * - `scikit-survival <https://github.com/sebp/scikit-survival>`_
     - Survival analysis built on top of scikit-learn
   * - `seasonal <https://github.com/welch/seasonal>`_
     - Toolkit to estimate trends and seasonality in time series
   * - `seqlearn <https://github.com/larsmans/seqlearn>`_
     - Extends the scikit-learn pipeline concept to time series annotation
   * - `seglearn <https://github.com/dmbee/seglearn>`_
     - Extends the scikit-learn pipeline concept to time series data for classification, regression and forecasting
   * - `sktime <https://github.com/sktime/sktime>`_
     - A scikit-learn compatible library for learning with time series/panel data including time series classification/regression and (supervised/panel) forecasting
   * - `statsforecast <https://github.com/Nixtla/statsforecast>`_
     - StatsForecast by Nixtla offers a collection of widely used univariate time series forecasting models optimized for high performance using numba
   * - `statsmodels <https://github.com/statsmodels/statsmodels>`_
     - Contains a submodule for classical time series models and hypothesis tests
   * - `stumpy <https://github.com/TDAmeritrade/stumpy>`_
     - Calculates matrix profile for time series subsequence all-pairs-similarity-search
   * - `tbats <https://pypi.org/project/tbats/>`_
     - Package provides BATS and TBATS time series forecasting methods
   * - `tensorflow_probability.sts <https://github.com/tensorflow/probability/tree/main/tensorflow_probability/python/sts>`_
     - Bayesian Structural Time Series model in Tensorflow Probability
   * - `timechop <https://github.com/dssg/timechop>`_
     - Toolkit for temporal cross-validation, part of the Data Science for Social Good predictive analytics framework
   * - `Traces <https://github.com/datascopeanalytics/traces>`_
     - A library for unevenly-spaced time series analysis
   * - `ta-lib <https://github.com/mrjbq7/ta-lib>`_
     - Calculate technical indicators for financial time series (python wrapper around TA-Lib)
   * - `ta <https://github.com/bukosabino/ta>`_
     - Calculate technical indicators for financial time series
   * - `tseries <https://github.com/mhamilton723/tseries>`_
     - scikit-learn compatible time series regressor as a meta-estimator for forecasting
   * - `tsfresh <https://github.com/blue-yonder/tsfresh>`_
     - Extracts and filters features from time series, allowing supervised classificators and regressor to be applied to time series data
   * - `tslearn <https://github.com/rtavenar/tslearn>`_
     - Direct time series classifiers and regressors
   * - `tspreprocess <https://github.com/MaxBenChrist/tspreprocess>`_
     - Preprocess time series (resampling, denoising etc.), still WIP
   * - `alibi-detect <https://github.com/SeldonIO/alibi-detect>`_
     - Toolbox for Outlier, Adversarial and Drift detection


Specific model implementations
------------------------------

.. list-table::
   :header-rows: 1

   * - Project name
     - Description
   * - `ES-RNN forecasting algorithm <https://github.com/damitkwr/ESRNN-GPU>`_
     - Python implementation of the winning forecasting method of the M4 competition combining exponential smoothing with a recurrent neural network using PyTorch
   * - `Deep learning methods for time series classification <https://github.com/hfawaz/dl-4-tsc>`_
     - A collection of common deep learning architectures for time series classification
   * - `M4 competition <https://github.com/M4Competition>`_
     - Collection of statistical and machine learning forecasting methods
   * - `Microsoft forecasting <https://github.com/microsoft/forecasting>`_
     - Collection of forecasting models and best practices, interfaces existing libraries in Python and R
   * - `LSTM-Neural-Network-for-Time-Series-Prediction <https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction>`_
     - LSTM for forecasting model
   * - `LSTM_tsc <https://github.com/RobRomijnders/LSTM_tsc>`_
     - An LSTM for time series classification
   * - `shapelets-python <https://github.com/mohaseeb/shaplets-python>`_
     - Shapelet classifier based on a multi layer neural network
   * - `ROCKET <https://github.com/angus924/rocket>`_
     - Time series classification using random convolutional kernels
   * - `TensorFlow-Time-Series-Examples <https://github.com/hzy46/TensorFlow-Time-Series-Examples>`_
     - Time Series Prediction with tf.contrib.timeseries
   * - `UCR_Time_Series_Classification_Deep_Learning_Baseline <https://github.com/cauchyturing/UCR_Time_Series_Classification_Deep_Learning_Baseline>`_
     - Fully convolutional neural networks for state-of-the-art time series classification
   * - `WTTE-RNN <https://github.com/ragulpr/wtte-rnn/>`_
     - Time to event forecast by RNN based Weibull density estimation


Time series databases and frameworks
====================================

.. list-table::
   :header-rows: 1

   * - Project Name
     - Description
   * - `artic <https://github.com/manahl/arctic>`_
     - High performance datastore for time series and tick data
   * - `automl_service <https://github.com/crawles/automl_service>`_
     - Fully automated time series classification pipeline, deployed as a web service
   * - `cesium <https://github.com/cesium-ml/cesium>`_
     - Time series platform with feature extraction aiming for non uniformly sampled signals
   * - `thunder <https://github.com/thunder-project/thunder>`_
     - Scalable analysis of image and time series data in Python based on spark
   * - `whisper <https://github.com/graphite-project/whisper>`_
     - File-based time-series database format
   * - `FinTime <https://cs.nyu.edu/shasha/fintime.html>`_
     - Financial time series database framework, design, benchmarks
   * - `MNE <https://martinos.org/mne/stable/index.html>`_
     - Python software for exploring, visualizing, and analyzing neurophysiological time series data (MEG, EEG, etc)


Acknowledgements
================

Thanks to `Max Christ <https://github.com/MaxBenChrist/>`_ who started the list `here <https://github.com/MaxBenChrist/awesome_time_series_in_python/blob/main/README.md>`_.
