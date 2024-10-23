#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for annotator base type for time series stream.

    class name: BaseSeriesAnnotator

Scitype defining methods:
    fitting              - fit(self, X, Y=None)
    annotating           - predict(self, X)
    updating (temporal)  - update(self, X, Y=None)
    update&annotate      - update_predict(self, X)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()
"""

# todo 0.37.0: remove this base module
__all__ = ["BaseSeriesAnnotator"]

from sktime.detection.base import BaseDetector
from sktime.utils.warnings import warn


class BaseSeriesAnnotator(BaseDetector):
    """Base series annotator.

    Developers should set the task and learning_type tags in the derived class.

    task : str {"segmentation", "change_point_detection", "anomaly_detection"}
        The main annotation task:
        * If ``segmentation``, the annotator divides timeseries into discrete chunks
        based on certain criteria. The same label can be applied at multiple
        disconnected regions of the timeseries.
        * If ``change_point_detection``, the annotator finds points where the
        statistical properties of the timeseries change significantly.
        * If ``anomaly_detection``, the annotator finds points that differ significantly
        from the normal statistical properties of the timeseries.

    learning_type : str {"supervised", "unsupervised"}
        Annotation learning type:
        * If ``supervised``, the annotator learns from labelled data.
        * If ``unsupervised``, the annotator learns from unlabelled data.

    Notes
    -----
    Assumes "predict" data is temporal future of "fit"
    Single time series in both, no meta-data.

    The base series annotator specifies the methods and method
    signatures that all annotators have to implement.

    Specific implementations of these methods is deferred to concrete
    annotators.
    """

    def __init__(self):

        super().__init__()

        warn(
            "BaseSeriesAnnotator is deprecated as a base class for "
            "detection algorithms. Extension developers should use BaseDetector "
            "from sktime.detection.base instead.",
            obj=self,
        )
