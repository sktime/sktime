.. _detection_ref:

Time series detection tasks
===========================

The :mod:`sktime.annotation` module contains algorithms and tools
for time series detection tasks, including:

* anomaly or outlier detection
* change point detection
* time series segmentation and segment detection

The tasks include unsupervised and semi-supervised variants, and can batch or
stream/online detection.


Time Series Segmentation
------------------------

.. currentmodule:: sktime.annotation.clasp

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClaSPSegmentation

.. currentmodule:: sktime.annotation.eagglo

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    EAgglo

.. currentmodule:: sktime.annotation.hmm_learn.gaussian

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GaussianHMM

.. currentmodule:: sktime.annotation.hmm_learn.gmm

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GMMHMM

.. currentmodule:: sktime.annotation.ggs

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    GreedyGaussianSegmentation

.. currentmodule:: sktime.annotation.hmm

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HMM

.. currentmodule:: sktime.annotation.igts

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    InformationGainSegmentation

.. currentmodule:: sktime.annotation.hmm_learn.poisson

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PoissonHMM

.. currentmodule:: sktime.annotation.stray

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    STRAY

.. currentmodule:: sktime.annotation.clust

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClusterSegmenter

Time Series Anomaly Detection
-----------------------------

Window-based Anomaly Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: sktime.annotation.lof

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SubLOF

Reduction to Tabular Anomaly Detection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: sktime.annotation.adapters

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PyODAnnotator


Data Generation
---------------

.. automodule:: sktime.annotation.datagen
    :no-members:
    :no-inherited-members:
