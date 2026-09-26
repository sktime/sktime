#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class template for detector base type for time series streams.

    class name: BaseDetector

Scitype defining methods:
    pretraining          - pretrain(self, X, y=None)
    fitting              - fit(self, X, y=None)
    dtecting             - predict(self, X)
    updating (temporal)  - update(self, X, y=None)
    update&annotate      - update_predict(self, X, y=None)

Inspection methods:
    hyper-parameter inspection  - get_params()
    fitted parameter inspection - get_fitted_params()

State:
    fitted model/strategy   - by convention, any attributes ending in "_"
    fitted state flag       - check_is_fitted()
    estimator state         - state, one of "new", "pretrained", "fitted"
"""

__author__ = ["fkiraly", "tveten", "alex-jg3", "satya-pattnaik"]
__all__ = ["BaseDetector"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_estimator_deps

from sktime.base import BaseEstimator
from sktime.datatypes import check_is_error_msg, check_is_scitype, convert
from sktime.utils.adapters._safe_call import _method_has_arg
from sktime.utils.multiindex import flatten_multiindex
from sktime.utils.validation.series import check_series


class BaseDetector(BaseEstimator):
    """Base class for time series detectors.

    Developers should set the task and learning_type tags in the derived class.

    task : str {"segmentation", "change_point_detection", "anomaly_detection"}
        The main detection task:

        * If ``segmentation``, the detector divides timeseries into discrete chunks
        based on certain criteria. The same label can be applied at multiple
        disconnected regions of the timeseries.
        * If ``change_point_detection``, the detector finds points where the
        statistical properties of the timeseries change significantly.
        * If ``anomaly_detection``, the detector finds points that differ significantly
        from the normal statistical properties of the timeseries.

    learning_type : str {"supervised", "unsupervised", "semi_supervised"}
        Detection learning type:

        * If ``supervised``, the detector learns from labelled data.
        * If ``unsupervised``, the detector learns from unlabelled data.
        * If ``semi_supervised``, the detector learns from a combination of labelled
          and unlabelled data.

    Notes
    -----
    The base series detector specifies the methods and method
    signatures that all detectors have to implement.

    Specific implementations of these methods is deferred to concrete detectors.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # str or list of str, package soft dependencies
        # estimator tags
        # --------------
        "object_type": "detector",
        "learning_type": "None",  # supervised, unsupervised
        "task": "None",  # anomaly_detection, change_point_detection, segmentation
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:update": False,
        "capability:pretrain": False,
        "capability:variable_identification": False,
        #
        # todo: distribution_type does not seem to be used - refactor or remove
        "distribution_type": "None",
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
    }

    def __init__(self):
        self._state = "new"

        super().__init__()

        # this block has a double purpose:
        # - emit a warning if dependencies are not met, but allow instantiation
        # - if dependencies are met, call __post_init__ used by inheriting classes
        if _check_estimator_deps(self, severity="warning"):
            self.__post_init__()

    def __post_init__(self):
        """Post-init constructor logic, can be used by inheriting classes.

        This method should be used for:

        * parameter validation
        * initialization logic beyond self.param = param
        * any soft dependency imports in the constructor
        """
        pass

    @classmethod
    def _get_clone_plugins(cls):
        """Get clone plugins for BaseDetector.

        Overrides the default skbase clone behavior to preserve
        pretrained attributes when cloning detectors.

        The ``_PretrainedCloner`` plugin ensures that when a detector
        with pretrained state is cloned, the pretrained attributes are
        copied to the clone. Detectors without the ``capability:pretrain``
        tag are cloned as usual.

        Returns
        -------
        list
            List containing ``_PretrainedCloner`` plugin class.
        """
        # imported here and not at module level, as detection must not
        # import forecasting at module level, see the cross module import test
        from sktime.forecasting.base._clone_plugin import _PretrainedCloner

        return [_PretrainedCloner]

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated DetectorPipeline.

        Implemented for ``other`` being a transformer, otherwise returns
        ``NotImplemented``.

        Parameters
        ----------
        other: ``sktime`` transformer, must inherit from BaseTransformer
            otherwise, ``NotImplemented`` is returned

        Returns
        -------
        DetectorPipeline object,
            concatenation of ``other`` (first) with ``self`` (last).
            not nested, contains only non-DetectorPipeline ``sktime`` steps
        """
        from sktime.detection.compose import DetectorPipeline
        from sktime.transformations.adapt import TabularToSeriesAdaptor
        from sktime.transformations.base import BaseTransformer
        from sktime.utils.sklearn import is_sklearn_transformer

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformedTargetForecaster does the rest, e.g., dispatch on other
        if isinstance(other, BaseTransformer):
            self_as_pipeline = DetectorPipeline(steps=[self])
            return other * self_as_pipeline
        elif is_sklearn_transformer(other):
            return TabularToSeriesAdaptor(other) * self
        else:
            return NotImplemented

    @property
    def _is_fitted(self):
        """Internal fitted state for backward compatibility with skbase.

        Returns True only after ``fit``, not after ``pretrain``.
        For pretrain state checks, use ``self.state`` directly.

        Returns
        -------
        bool
            True if the detector has been fitted, False otherwise.
        """
        return self._state == "fitted"

    @_is_fitted.setter
    def _is_fitted(self, value):
        """Setter for backward compatibility.

        Parameters
        ----------
        value : bool
            If True, sets state to "fitted". If False, sets state to "new".
        """
        if value:
            self._state = "fitted"
        else:
            self._state = "new"

    @property
    def state(self):
        """State of the detector.

        Possible states for detectors are:

        * "new": post-init state
        * "pretrained": after ``pretrain`` has been called, until ``fit`` is called
        * "fitted": after ``fit`` has been called

        Returns
        -------
        str, one of {"new", "pretrained", "fitted"}
            State of the detector.
        """
        return self._state

    def pretrain(self, X, y=None):
        """Pretrain detector on a collection of time series.

        Pretrains the detector on Panel or Hierarchical data, i.e., multiple
        time series, before it is fitted to a single series via ``fit``.

        Only detectors with the ``capability:pretrain`` tag set to ``True``
        learn from the data. For all other detectors, ``pretrain`` is a no-op:
        the data is checked, nothing is stored, and only the state changes.

        For detectors with the ``capability:pretrain`` tag, if ``pretrain`` is
        called when the detector is already in the ``"pretrained"`` or
        ``"fitted"`` state, the internal ``_pretrain_update`` method is called
        instead of ``_pretrain``. By default, this pretrains again on the new
        data only, replacing the previous pretrained state.

        State change:
            Changes state to ``"pretrained"``.

        Writes to self:

            * Sets ``self.state`` to ``"pretrained"``.
            * If the ``capability:pretrain`` tag is ``True``, sets pretrained
              model attributes ending in ``"_"``, inspectable via
              ``get_pretrained_params``.

        Parameters
        ----------
        X : Panel or Hierarchical data in ``sktime`` compatible format
            Data to pretrain the detector on, must contain multiple time series.
            For instance, a ``pd.DataFrame`` with 2-level row ``MultiIndex``
            ``(instance, time)``, or a 3D ``np.ndarray``
            ``(instance, variable, time)``.
            Hierarchical data is flattened to Panel data, by joining all
            instance levels into one level, for instance, the instance
            ``("h0_0", "h1_0")`` becomes ``"h0_0__h1_0"``.

        y : optional, default=None
            Known events in ``X``, for detectors that learn from labels.
            Not checked or converted, passed on to ``_pretrain`` as is.
            Ignored by detectors without the ``capability:pretrain`` tag.

        Returns
        -------
        self : reference to self

        Raises
        ------
        TypeError
            If ``X`` is a single time series, or not valid Panel
            or Hierarchical data.

        See Also
        --------
        fit : Fit detector to a single series.
        get_pretrained_params : Retrieve parameters set during pretraining.
        state : Current state of the detector.
        """
        _check_estimator_deps(self)

        X, X_metadata = self._check_X_pretrain(X)

        # detectors without pretrain capability: no-op, only the state changes
        can_pretrain = self.get_tag(
            "capability:pretrain", tag_value_default=False, raise_error=False
        )
        if not can_pretrain:
            self._state = "pretrained"
            return self

        # pretrain accepts panel data independent of what fit and predict
        # support via X_inner_mtype, so panel mtypes are added here
        X_inner_mtype = self.get_tag("X_inner_mtype")
        if isinstance(X_inner_mtype, str):
            X_inner_mtype = [X_inner_mtype]
        X_inner = convert(
            X,
            from_type=X_metadata["mtype"],
            to_type=list(X_inner_mtype) + ["pd-multiindex"],
        )

        prior_attrs = {
            a for a in dir(self) if a.endswith("_") and not a.startswith("_")
        }

        if self._state == "new":
            self._pretrain(X=X_inner, y=y)
        else:
            self._pretrain_update(X=X_inner, y=y)

        if not hasattr(self, "_pretrained_attrs"):
            self._pretrained_attrs = []

        # track attributes set by this pretrain call
        new_attrs = [
            a
            for a in dir(self)
            if a.endswith("_")
            and not a.startswith("_")
            and a not in self._pretrained_attrs
            and a not in prior_attrs
        ]
        self._pretrained_attrs.extend(new_attrs)

        self._state = "pretrained"
        return self

    def get_pretrained_params(self, deep=True):
        """Get pretrained parameters of this detector.

        Returns a dictionary of attributes that were set during ``pretrain``.
        These are attributes ending in ``"_"`` that appeared on the detector
        after ``pretrain`` was called, and are tracked separately from the
        fitted parameters set by ``fit``.

        Returns an empty dict if the detector has not been pretrained,
        or does not have the ``capability:pretrain`` tag.

        Parameters
        ----------
        deep : bool, default=True
            Whether to return pretrained parameters of nested estimators.

            * If True, will return a dict of parameter name : value for this
              object, including pretrained parameters of nested estimators.
            * If False, will return a dict of parameter name : value for this
              object, but not include pretrained parameters of nested
              estimators.

        Returns
        -------
        params : dict
            Dictionary of pretrained parameter names mapped to their values.
            Keys are attribute names ending in ``"_"`` that were set during
            pretraining.

            If ``deep=True``, also contains keys/value pairs of nested
            estimators' pretrained parameters, indexed as
            ``[attrname]__[paramname]``.

        See Also
        --------
        pretrain : Pretrain detector on a collection of time series.
        get_fitted_params : Get parameters set during ``fit``.
        """
        if not hasattr(self, "_pretrained_attrs"):
            return {}

        params = {}
        for attr in self._pretrained_attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                params[attr] = value

                # Handle nesting: if value is an estimator with pretrained params
                if deep and hasattr(value, "get_pretrained_params"):
                    nested = value.get_pretrained_params(deep=True)
                    for nested_key, nested_val in nested.items():
                        params[f"{attr}__{nested_key}"] = nested_val

        return params

    def fit(self, X, y=None):
        """Fit to training data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Training data to fit model to (time series).

        y : pd.DataFrame with RangeIndex, optional.
            Known events for traininmg, in ``X``, if detector is supervised.

            Each row ``y`` is a known event.
            Can have the following columns:

            * ``"ilocs"`` - always. Values encode where/when the event takes place,
              via ``iloc`` references to indices of ``X``,
              or ranges to indices of ``X``, as below.
            * ``"label"`` - if the task, by tags, is supervised or semi-supervised
              segmentation with labels, or segment clustering.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column describe the event in a given row as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index at which the event takes place.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, interpreted as the range
              of indices over which the event takes place.

            Labels (if present) in the ``"labels"`` column indicate the type of event.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Creates fitted model that updates attributes ending in "_". Sets
        _is_fitted flag to True.
        """
        _check_estimator_deps(self)

        # input checks and conversions for X
        X_inner = self._check_X(X)

        # skip inner _fit if fit is empty
        # we also do not need to memorize data, since we do same in _update
        # basic checks (above) are still needed
        if self.get_tag("fit_is_empty", False):
            self._is_fitted = True
            return self

        self._X = X
        self._y = y

        if _method_has_arg(self._fit, "y"):
            self._fit(X=X_inner, y=y)  # X_inner is the converted X
        else:
            self._fit(X=X_inner)

        # this should happen last
        self._is_fitted = True

        return self

    def predict(self, X):
        """Create labels on test/deployment data.

        This method returns a list-like type specific to the detection task,
        e.g., segments for segmentation, anomalies for anomaly detection.

        The encoding varies by task and learning_type (tags), see below.

        For returns that are type consistent across tasks, see
        ``predict_points`` and ``predict_segments``.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with RangeIndex
            Detected or predicted events.

            Each row ``y`` is a detected or predicted event.
            Can have the following columns:

            * ``"ilocs"`` - always. Values encode where/when the event takes place,
              via ``iloc`` references to indices of ``X``,
              or ranges to indices of ``X``, as below.
            * ``"label"`` - if the task, by tags, is supervised or semi-supervised
              segmentation with labels, or segment clustering.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column describe the event in a given row as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index at which the event takes place.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, interpreted as the range
              of indices over which the event takes place.

            Labels (if present) in the ``"labels"`` column indicate the type of event.
        """
        self.check_is_fitted()

        X_inner = self._check_X(X)

        y = self._predict(X=X_inner)

        # deal with legacy return format with intervals in index
        y = self._coerce_to_df(y, columns=["ilocs"])

        return y

    def transform(self, X):
        """Create labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with same index as X
            Labels for sequence ``X``.

            * If ``task`` is ``"anomaly_detection"``, the values are integer labels.
              A value of 0 indicates that ``X``, at the same time index, has no anomaly.
              Other values indicate an anomaly.
              Most detectors will return 0 or 1, but some may return more values,
              if they can detect different types of anomalies.
              indicating whether ``X``, at the same
              index, is an anomaly, 0 for no, 1 for yes.
            * If ``task`` is ``"changepoint_detection"``, the values are integer labels,
              indicating labels for segments between changepoints.
              Possible labels are integers starting from 0.
            * If ``task`` is "segmentation", the values are integer labels of the
              segments. Possible labels are integers starting from 0.
        """
        y_sparse = self.predict(X)
        y_dense = self.sparse_to_dense(y_sparse, pd.RangeIndex(len(X)))
        y_dense = self._coerce_to_df(y_dense, columns=["labels"])
        return y_dense

    def transform_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to label (time series).

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Scores for sequence ``X``.
        """
        self.check_is_fitted()

        X_inner = self._check_X(X)

        return self._transform_scores(X_inner)

    def predict_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to label (time series).

        Returns
        -------
        scores : pd.DataFrame with same index as return of predict
            Scores for prediction of sequence ``X``.
        """
        self.check_is_fitted()

        X_inner = self._check_X(X)
        scores = self._predict_scores(X_inner)

        return pd.DataFrame(scores)

    def update(self, X, y=None):
        """Update model with new data and optional ground truth labels.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Training data to update model with (time series).
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        self.check_is_fitted()

        X_inner = self._check_X(X)

        # no update needed if fit is empty
        if self.get_tag("fit_is_empty", False):
            return self

        self._X = X_inner.combine_first(self._X)

        if y is not None:
            self._y = y.combine_first(self._y)

        if _method_has_arg(self._update, "y"):
            self._update(X=X_inner, y=y)
        else:
            self._update(X=X_inner)

        return self

    def update_predict(self, X, y=None):
        """Update model with new data and create labels for it.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Training data to update model with, time series.
        y : pd.DataFrame with RangeIndex, optional.
            Known events for training, in ``X``, if detector is supervised.

            Each row ``y`` is a known event.
            Can have the following columns:

            * ``"ilocs"`` - always. Values encode where/when the event takes place,
              via ``iloc`` references to indices of ``X``,
              or ranges to indices of ``X``, as below.
            * ``"label"`` - if the task, by tags, is supervised or semi-supervised
              segmentation with labels, or segment clustering.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column describe the event in a given row as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index at which the event takes place.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, interpreted as the range
              of indices over which the event takes place.

            Labels (if present) in the ``"labels"`` column indicate the type of event.

        Returns
        -------
        y : pd.DataFrame with RangeIndex
            Detected or predicted events.

            Each row ``y`` is a detected or predicted event.
            Can have the following columns:

            * ``"ilocs"`` - always. Values encode where/when the event takes place,
              via ``iloc`` references to indices of ``X``,
              or ranges to indices of ``X``, as below.
            * ``"label"`` - if the task, by tags, is supervised or semi-supervised
              segmentation, or segment clustering.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column describe the event in a given row as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index at which the event takes place.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, interpreted as the range
              of indices over which the event takes place.

            Labels (if present) in the ``"labels"`` column indicate the type of event.
        """
        X_inner = self._check_X(X)

        self.update(X=X, y=y)
        y = self.predict(X=X_inner)

        return y

    def fit_predict(self, X, y=None):
        """Fit to data, then predict it.

        Fits model to X and Y with given detection parameters
        and returns the detection labels produced by the model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed

        y : pd.DataFrame with RangeIndex, optional.
            Known events for training, in ``X``, if detector is supervised.

            Each row ``y`` is a known event.
            Can have the following columns:

            * ``"ilocs"`` - always. Values encode where/when the event takes place,
              via ``iloc`` references to indices of ``X``,
              or ranges to indices of ``X``, as below.
            * ``"label"`` - if the task, by tags, is supervised or semi-supervised
              segmentation with labels, or segment clustering.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column describe the event in a given row as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index at which the event takes place.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, interpreted as the range
              of indices over which the event takes place.

            Labels (if present) in the ``"labels"`` column indicate the type of event.

        Returns
        -------
        y : pd.DataFrame with RangeIndex
            Detected or predicted events.

            Each row ``y`` is a detected or predicted event.
            Can have the following columns:

            * ``"ilocs"`` - always. Values encode where/when the event takes place,
              via ``iloc`` references to indices of ``X``,
              or ranges to indices of ``X``, as below.
            * ``"label"`` - if the task, by tags, is supervised or semi-supervised
              segmentation with labels, or segment clustering.

            The meaning of entries in the ``"ilocs"`` column and ``"labels"``
            column describe the event in a given row as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              ``"ilocs"`` contains the iloc index at which the event takes place.
            * If ``task`` is ``"segmentation"``, ``"ilocs"`` contains left-closed
              intervals of iloc based segments, interpreted as the range
              of indices over which the event takes place.

            Labels (if present) in the ``"labels"`` column indicate the type of event.
        """
        # Non-optimized default implementation; override when a better
        # method is possible for a given algorithm.
        return self.fit(X, y=y).predict(X)

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it.

        Fits model to X and Y with given detection parameters
        and returns the detection labels made by the model.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed
        y : pd.Series or np.ndarray, optional (default=None)
            Target values of data to be predicted.

        Returns
        -------
        y : pd.DataFrame with same index as X
            Labels for sequence ``X``.

            * If ``task`` is ``"anomaly_detection"``, the values are integer labels.
              A value of 0 indicatesthat ``X``, at the same time index, has no anomaly.
              Other values indicate an anomaly.
              Most detectors will return 0 or 1, but some may return more values,
              if they can detect different types of anomalies.
              indicating whether ``X``, at the same
              index, is an anomaly, 0 for no, 1 for yes.
            * If ``task`` is ``"changepoint_detection"``, the values are integer labels,
              indicating labels for segments between changepoints.
              Possible labels are integers starting from 0.
            * If ``task`` is "segmentation", the values are integer labels of the
              segments. Possible labels are integers starting from 0.
        """
        y_sparse = self.fit_predict(X, y=y)

        # Handle both pandas and numpy inputs
        if hasattr(X, "index"):
            # X is pandas DataFrame or Series
            index = X.index
        else:
            # X is numpy array or other array-like without index
            # Create a default integer index
            index = pd.RangeIndex(len(X))

        y_dense = self.sparse_to_dense(y_sparse, index=index)
        y_dense = self._coerce_to_df(y_dense, columns=["labels"])
        return y_dense

    def _coerce_to_df(self, y, columns=None):
        """Coerce output to a DataFrame.

        Also deals with the following downwards cases:

        * IntervalIndex containing segments -> DataFrame with "ilocs" column
        """
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.DataFrame(y, columns=columns, dtype="int64")
        if isinstance(y.index, pd.IntervalIndex):
            if isinstance(y, pd.Series):
                y = pd.DataFrame(y.index, columns=columns)
            elif isinstance(y, pd.DataFrame):
                y_index = pd.DataFrame(y.index, columns=columns)
                y = y.reset_index(drop=True)
                y = pd.concat([y_index, y], axis=1)

        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y, columns=columns, dtype="int64")

        return y

    def _coerce_intervals_to_values(self, y):
        if not isinstance(y, (pd.Series, pd.DataFrame)):
            y = pd.Series(y, dtype="int64")
        if isinstance(y.index, pd.IntervalIndex):
            if isinstance(y, pd.Series):
                y = pd.Series(y.index)
        return y

    def _check_X(self, X):
        """Check input data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed

        Returns
        -------
        X : X_inner_mtype
            Data to be transformed
        """
        ALLOWED_SCITYPES = ["Series", "Panel"]
        X_valid, X_msg, X_metadata = check_is_scitype(
            X, scitype=ALLOWED_SCITYPES, return_metadata=[]
        )
        self._X_metadata = X_metadata
        if not X_valid:
            msg_start = (
                f"Unsupported input data type in {self.__class__.__name__}, input X"
            )
            allowed_msg = (
                "Allowed scitypes for X in detection are "
                f"{', '.join(ALLOWED_SCITYPES)}, "
                "for instance a pandas.DataFrame with sktime compatible time indices."
                " See the detection tutorial examples/07_detection.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb"
            )
            if not X_valid:
                check_is_error_msg(
                    X_msg,
                    var_name=msg_start,
                    allowed_msg=allowed_msg,
                    raise_exception=True,
                )

        X_inner_mtype = self.get_tag("X_inner_mtype")
        X_inner = convert(X, from_type=X_metadata["mtype"], to_type=X_inner_mtype)
        return X_inner

    def _check_X_pretrain(self, X):
        """Check input data to pretrain, and flatten Hierarchical data to Panel.

        Unlike ``_check_X``, does not write to self.

        Panel data is returned unchanged. Hierarchical data is flattened to
        Panel data, by joining all instance levels into one level with
        ``flatten_multiindex``, for instance, the instance ``("h0_0", "h1_0")``
        becomes ``"h0_0__h1_0"``.

        Parameters
        ----------
        X : object
            Input data to ``pretrain``.

        Returns
        -------
        X : Panel data
            ``X`` if it is Panel data, or ``X`` flattened to a ``pd.DataFrame``
            with 2-level row ``MultiIndex`` if it is Hierarchical data.
        X_metadata : dict
            Metadata of the returned ``X``, as returned by ``check_is_scitype``,
            contains the mtype of ``X`` in the ``"mtype"`` key.

        Raises
        ------
        TypeError
            If ``X`` is a single time series, or not valid Panel
            or Hierarchical data.
        """
        name = type(self).__name__

        if check_is_scitype(X, scitype="Series"):
            raise TypeError(
                f"{name}.pretrain requires Panel or Hierarchical data "
                "(multiple time series), but a single Series was passed. "
                "Use fit for a single series, or pass Panel data to pretrain, "
                "for instance a pd.DataFrame with 2-level row MultiIndex "
                "(instance, time)."
            )

        X_valid, X_msg, X_metadata = check_is_scitype(
            X, scitype=["Panel", "Hierarchical"], return_metadata=[]
        )

        # flatten all instance levels into one level, then continue as Panel
        if X_valid and X_metadata["scitype"] == "Hierarchical":
            X = convert(X, from_type=X_metadata["mtype"], to_type="pd_multiindex_hier")
            instances = flatten_multiindex(X.index.droplevel(-1))
            times = X.index.get_level_values(-1)
            flat_index = pd.MultiIndex.from_arrays(
                [instances, times], names=[None, times.name]
            )
            X = X.set_axis(flat_index, axis=0)
            X_valid, X_msg, X_metadata = check_is_scitype(
                X, scitype="Panel", return_metadata=[]
            )

        if not X_valid:
            check_is_error_msg(
                X_msg,
                var_name=f"Unsupported input data type in {name}.pretrain, input X",
                allowed_msg=(
                    "Allowed scitypes for X in pretrain are Panel and Hierarchical, "
                    "for instance a pd.DataFrame with 2-level row MultiIndex "
                    "(instance, time), or a 3D np.ndarray (instance, variable, time)."
                ),
                raise_exception=True,
            )

        return X, X_metadata

    def _pretrain(self, X, y=None):
        """Pretrain detector on a collection of time series, first call.

        private _pretrain containing the core logic, called from pretrain
        if the ``capability:pretrain`` tag is ``True``

        Writes to self:
            Sets pretrained model attributes ending in "_".

        Parameters
        ----------
        X : pd.DataFrame with 2-level row MultiIndex, or other Panel mtype
            Data to pretrain the detector on, a collection of time series.
        y : optional, default=None
            Known events in ``X``, as passed to ``pretrain``.

        Returns
        -------
        self :
            Reference to self.
        """
        # the default simply discards the data, i.e., no pretraining happens
        return self

    def _pretrain_update(self, X, y=None):
        """Pretrain detector on a collection of time series, if already pretrained.

        private _pretrain_update containing the core logic, called from pretrain
        if the ``capability:pretrain`` tag is ``True``, and the detector
        is already in the ``"pretrained"`` or ``"fitted"`` state

        Writes to self:
            Sets pretrained model attributes ending in "_".

        Parameters
        ----------
        X : pd.DataFrame with 2-level row MultiIndex, or other Panel mtype
            Data to pretrain the detector on, a collection of time series.
        y : optional, default=None
            Known events in ``X``, as passed to ``pretrain``.

        Returns
        -------
        self :
            Reference to self.
        """
        # the default calls _pretrain, i.e., the new data replaces the old
        return self._pretrain(X=X, y=y)

    def _fit(self, X, y=None):
        """Fit to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pd.DataFrame
            Training data to fit model to time series.
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.
        """
        raise NotImplementedError("abstract method")

    def _predict(self, X):
        """Create labels on test/deployment data.

        private _predict containing the core logic, called from predict

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.Series with RangeIndex
            Labels for sequence ``X``, in sparse format.
            Values are ``iloc`` references to indices of ``X``.

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the values are integer indices of the changepoints/anomalies.
            * If ``task`` is "segmentation", the values are ``pd.Interval`` objects.
        """
        raise NotImplementedError("abstract method")

    def _predict_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            Labels for sequence X exact format depends on detection type.
        """
        raise NotImplementedError("abstract method")

    def _transform_scores(self, X):
        """Return scores for predicted labels on test/deployment data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        scores : pd.DataFrame with same index as X
            Scores for sequence ``X``.
        """
        raise NotImplementedError("abstract method")

    def _update(self, X, y=None):
        """Update model with new data and optional ground truth labels.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            Training data to update model with time series
        y : pd.Series, optional
            Ground truth labels for training if detector is supervised.

        Returns
        -------
        self :
            Reference to self.

        Notes
        -----
        Updates fitted model that updates attributes ending in "_".
        """
        # default/fallback: re-fit to all data
        self._fit(self._X, self._y)

        return self

    def predict_segments(self, X):
        """Predict segments on test/deployment data.

        The main difference to ``predict`` is that this method always returns
        a ``pd.DataFrame`` with segments of interest, even if the task is not
        segmentation.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with RangeIndex

            ``pd.DataFrame`` with the following columns:

            * ``"ilocs"`` - always. Values are left-closed intervals with
              left/right values being ``iloc`` references to indices of ``X``,
              signifying segments.
            * ``"labels"`` - if the task, by tags, is supervised or semi-supervised
              segmentation, or segment clustering.

            The meaning of segments in the ``"ilocs"`` column and ``"labels"``
            column is as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the intervals are intervals between changepoints/anomalies, and
              potential labels are consecutive integers starting from 0.
            * If ``task`` is ``"segmentation"``, the values are segmentation labels.
        """
        self.check_is_fitted()
        X = check_series(X)

        task = self.get_tag("task")
        if task in ["anomaly_detection", "change_point_detection"]:
            y_pred_pts = self.predict_points(X)
            y_pred = self.change_points_to_segments(y_pred_pts, start=0, end=len(X))
        elif task == "segmentation":
            y_pred = self._predict_segments(X)

        y_pred = self._coerce_to_df(y_pred, columns=["ilocs"])
        return y_pred

    def predict_points(self, X):
        """Predict changepoints/anomalies on test/deployment data.

        The main difference to ``predict`` is that this method always returns
        a ``pd.DataFrame`` with points of interest, even if the task is not
        anomaly or change point detection.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        y : pd.DataFrame with RangeIndex

            ``pd.DataFrame`` with the following columns:

            * ``"ilocs"`` - always. Values are integers, ``iloc``
              references to indices of ``X``, signifying points of interest.
            * ``"labels"`` - if the task, by tags, is supervised or semi-supervised
              segmentation, or anomaly clustering.

            The meaning of segments in the ``"ilocs"`` column and ``"labels"``
            column is as follows:

            * If ``task`` is ``"anomaly_detection"`` or ``"change_point_detection"``,
              the values are integer indices of the changepoints/anomalies.
            * If ``task`` is ``"segmentation"``, the values are consecutive
              segment boundaries.

            The ``"labels"`` are potential labels for the points of interest.
        """
        self.check_is_fitted()
        X = check_series(X)

        task = self.get_tag("task")
        if task in ["anomaly_detection", "change_point_detection"]:
            y_pred = self._predict_points(X)
        elif task == "segmentation":
            y_pred_seg = pd.DataFrame(self.predict_segments(X))
            y_pred = self.segments_to_change_points(y_pred_seg)

        y_pred = self._coerce_to_df(y_pred, columns=["ilocs"])
        return y_pred

    def _predict_segments(self, X):
        """Predict segments on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            A series with an index of intervals. Each interval is the range of a
            segment and the corresponding value is the label of the segment.
        """
        return self._predict(X)

    def _predict_points(self, X):
        """Predict changepoints/anomalies on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series
            A series whose values are the changepoints/anomalies in X.
        """
        return self._predict(X)

    @staticmethod
    def sparse_to_dense(y_sparse, index):
        """Convert the sparse output from an detector to a dense format.

        Parameters
        ----------
        y_sparse : pd.Series
            * If ``y_sparse`` is a series with an index of intervals, it should
              represent segments where each value of the series is label of a segment.
              Unclassified intervals should be labelled -1. Segments must never have
              the label 0.
            * If the index of ``y_sparse`` is not a set of intervals, the values of the
              series should represent the indexes of changepoints/anomalies.
        index : array-like
            Larger set of indices which contains event indices in ``y_sparse``,
            to be used as the index of the returned series.

        Returns
        -------
        pd.Series
            A series with an index of ``index`` is returned.
            * If ``y_sparse`` is a series of changepoints/anomalies then the returned
              series is labelled 0 and 1 dependendy on whether the index is associated
              with an anomaly/changepoint. Where 1 means anomaly/changepoint.
            * If ``y_sparse`` is a series of segments then the returned series is
              labelled depending on the segment its indexes fall into. Indexes that
              fall into no segments are labelled -1.

        Examples
        --------
        >>> import pandas as pd
        >>> from sktime.detection.base import BaseDetector
        >>> y_sparse = pd.Series([2, 5, 7])  # Indices of changepoints/anomalies
        >>> index = range(0, 8)
        >>> BaseDetector.sparse_to_dense(y_sparse, index=index)
        0    0
        1    0
        2    1
        3    0
        4    0
        5    1
        6    0
        7    1
        dtype: int64
        >>> y_sparse = pd.Series(
        ...     [1, 2, 1],
        ...     index=pd.IntervalIndex.from_arrays(
        ...         [0, 4, 6], [4, 6, 10], closed="left"
        ...     )
        ... )
        >>> index = range(10)
        >>> BaseDetector.sparse_to_dense(y_sparse, index=index)
        0    1
        1    1
        2    1
        3    1
        4    2
        5    2
        6    1
        7    1
        8    1
        9    1
        dtype: int64
        """
        if not isinstance(y_sparse, pd.DataFrame):
            y_sparse = pd.DataFrame(y_sparse, dtype="int64")
        if not hasattr(y_sparse, "ilocs") or y_sparse.ilocs.dtype != "interval":
            # Anomaly/changepoint detection case
            y_dense = BaseDetector._sparse_points_to_dense(y_sparse, index)
            return y_dense
        else:
            # Segmentation case
            y_dense = BaseDetector._sparse_segments_to_dense(y_sparse, index)
            return y_dense

    @staticmethod
    def _sparse_points_to_dense(y_sparse, index):
        """Label the indexes in ``index`` if they are in ``y_sparse``.

        Parameters
        ----------
        y_sparse: pd.Series
            The values of the series must be the indexes of changepoints/anomalies.
        index: array-like
            Array of indexes that are to be labelled according to ``y_sparse``.

        Returns
        -------
        pd.Series
            A series with an index of ``index``. Its values are 1 if the index is in
            y_sparse and 0 otherwise.
        """
        y_dense = pd.Series(np.zeros(len(index)), index=index, dtype="int64")
        y_dense[y_sparse.values.flatten()] = 1
        return y_dense

    @staticmethod
    def _sparse_segments_to_dense(y_sparse, index):
        """Find the label for each index in ``index`` from sparse segments.

        Parameters
        ----------
        y_sparse : pd.Series
            A sparse representation of segments. The index must be the pandas interval
            datatype and the values must be the integer labels of the segments.
        index : array-like
            List of indexes that are to be labelled according to ``y_sparse``.

        Returns
        -------
        pd.Series
            A series with the same index as ``index`` where each index is labelled
            according to ``y_sparse``. Indexes that do not fall within any index are
            labelled -1.
        """
        if len(y_sparse) == 0:
            return pd.DataFrame(0, index=index, dtype="int64", columns=["labels"])

        seg_index = y_sparse.set_index("ilocs").index
        index_rg = pd.RangeIndex(len(index))

        if seg_index.is_overlapping:
            raise NotImplementedError(
                "Cannot convert overlapping segments to a dense format yet."
            )

        interval_ixs = seg_index.get_indexer(index_rg)

        if "labels" not in y_sparse.columns:
            y_dense = pd.DataFrame({"labels": interval_ixs}, index=index_rg)
            return y_dense
        else:
            y_dense = y_sparse.labels.loc[interval_ixs]
            y_dense = y_dense.reset_index(drop=True)
            return pd.DataFrame({"labels": y_dense}, index=index_rg)

    @staticmethod
    def dense_to_sparse(y_dense):
        """Convert the dense output from an detector to a sparse format.

        Parameters
        ----------
        y_dense : pd.Series
            * If ``y_sparse`` contains only 1's and 0's, the 1's represent change
              points or anomalies.
            * If ``y_sparse`` contains only contains integers greater than 0, it is an
              an array of segments.

        Returns
        -------
        pd.Series

            * If ``y_sparse`` is a series of changepoints/anomalies, a pandas series
              will be returned containing the indexes of the changepoints/anomalies
            * If ``y_sparse`` is a series of segments, a series with an interval
              datatype index will be returned. The values of the series will be the
              labels of segments.
        """
        if isinstance(y_dense, pd.DataFrame):
            y_dense = y_dense.iloc[:, 0]
        if not isinstance(y_dense, pd.Series):
            y_dense = pd.Series(y_dense, dtype="int64")
        if 0 in y_dense.values:
            # y_dense is a series of change points
            change_points = np.where(y_dense.values != 0)[0]
            return pd.Series(change_points, dtype="int64")
        else:
            segment_start_indexes = np.where(y_dense.diff() != 0)[0]
            segment_end_indexes = np.roll(segment_start_indexes, -1)

            # The final index is always the end of a segment
            segment_end_indexes[-1] = y_dense.index[-1]

            segment_labels = y_dense.iloc[segment_start_indexes].to_numpy()
            interval_index = pd.IntervalIndex.from_arrays(
                segment_start_indexes, segment_end_indexes, closed="left"
            )
            y_sparse = pd.Series(segment_labels, index=interval_index)

            # -1 represents unclassified regions so we remove them
            y_sparse = y_sparse.loc[y_sparse != -1]
            return y_sparse

    @staticmethod
    def _empty_sparse():
        """Return an empty sparse series in indicator format.

        Returns
        -------
        pd.DataFrame
            A empty DataFrame with a RangeIndex.
        """
        return pd.DataFrame(index=pd.RangeIndex(0), dtype="int64", columns=["ilocs"])

    @staticmethod
    def _empty_segments():
        """Return an empty sparse DataFrame in segmentation format.

        Returns
        -------
        pd.DataFrame
            A empty DataFrame with an IntervalIndex.
        """
        empty_segs = pd.DataFrame(
            pd.IntervalIndex([]),
            index=pd.RangeIndex(0),
            dtype="int64",
            columns=["ilocs"],
        )
        return empty_segs

    @staticmethod
    def change_points_to_segments(y_sparse, start=None, end=None):
        """Convert an series of change point indexes to segments.

        Parameters
        ----------
        y_sparse : pd.Series of int, sorted ascendingly
            A series containing the iloc indexes of change points.
        start : optional, default=0
            Starting point of the first segment.
            Must be before the first change point, i.e., < y_sparse[0].
        end : optional, default=y_sparse[-1] + 1
            End point of the last segment.
            Must be after the last change point, i.e., > y_sparse[-1].

        Returns
        -------
        pd.Series
            A series with an interval index indicating the start and end points of the
            segments. The values of the series are the labels of the segments.

        Examples
        --------
        >>> import pandas as pd
        >>> from sktime.detection.base import BaseDetector
        >>> change_points = pd.Series([1, 2, 5])
        >>> BaseDetector.change_points_to_segments(change_points, 0, 7)
        [0, 1)    0
        [1, 2)    1
        [2, 5)    2
        [5, 7)    3
        dtype: int64
        """
        if len(y_sparse) == 0:
            return BaseDetector._empty_segments()

        breaks = y_sparse.values

        if start is not None and start > breaks.min():
            raise ValueError("The start index must be before the first change point.")
        if end is not None and end < breaks.max():
            raise ValueError("The end index must be after the last change point.")

        if start is None:
            start = 0
        if end is None:
            end = breaks[-1] + 1

        breaks = np.insert(breaks, 0, start)
        breaks = np.append(breaks, end)

        index = pd.IntervalIndex.from_breaks(breaks, copy=True, closed="left")
        segments = pd.Series(0, index=index)

        in_range = index.left >= start

        number_of_segments = in_range.sum()
        segments.loc[in_range] = range(0, number_of_segments)

        return segments

    @staticmethod
    def segments_to_change_points(y_sparse):
        """Convert segments to change points.

        Parameters
        ----------
        y_sparse : pd.DataFrame
            A series of segments. The index must be the interval data type and the
            values should be the integer labels of the segments.

        Returns
        -------
        pd.Index
            An Index array containing the indexes of the start of each segment.

        Examples
        --------
        >>> import pandas as pd
        >>> from sktime.detection.base import BaseDetector
        >>> segments =  pd.DataFrame({
                "ilocs": pd.IntervalIndex.from_tuples([(0, 3), (3, 4), (4, 5),
                (5, 6), (6, 7), (7, 8), (8, 10), (10, 11), (11, 12), (12, 20)]),
                "labels": [0, 2, 1, 0, 2, 1, 0, 2, 1, 0]
            })
        >>> BaseDetector.segments_to_change_points(segments)
        Index([0, 3, 4, 5, 6, 7, 8, 10, 11, 12], dtype='int64')
        """
        if len(y_sparse) == 0:
            return BaseDetector._empty_sparse()
        change_points = y_sparse.set_index("ilocs").index.left
        return change_points
