.. _detection_ref:

Time series detection tasks
===========================

The :mod:`sktime.detection` module contains algorithms and tools
for time series detection tasks, including:

* anomaly or outlier detection
* change point detection
* time series segmentation and segment detection

The tasks include unsupervised and semi-supervised variants, and can batch or
stream/online detection.

Composition
-----------

.. currentmodule:: sktime.detection.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    DetectorPipeline
    DetectorAsTransformer

Time Series Segmentation
------------------------

.. currentmodule:: sktime.detection.clasp

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClaSPSegmentation

.. currentmodule:: sktime.detection.eagglo

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EAgglo

.. currentmodule:: sktime.detection.hmm_learn.gaussian

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GaussianHMM

.. currentmodule:: sktime.detection.hmm_learn.gmm

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GMMHMM

.. currentmodule:: sktime.detection.ggs

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GreedyGaussianSegmentation

.. currentmodule:: sktime.detection.hmm

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HMM

.. currentmodule:: sktime.detection.igts

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    InformationGainSegmentation

.. currentmodule:: sktime.detection.hmm_learn.poisson

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PoissonHMM

.. currentmodule:: sktime.detection.stray

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    STRAY

.. currentmodule:: sktime.detection.clust

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClusterSegmenter

.. currentmodule:: sktime.detection.bs.BinarySegmentation

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BinarySegmentation

Time Series Anomaly Detection
-----------------------------

Window-based Anomaly Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: sktime.detection.lof

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SubLOF

Reduction to Tabular Anomaly Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: sktime.detection.adapters

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PyODDetector


Data Generation
---------------

.. automodule:: sktime.detection.datagen
    :no-members:
    :no-inherited-members:
