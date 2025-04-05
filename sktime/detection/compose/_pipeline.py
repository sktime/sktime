"""Pipeline for time series detectors."""

from sktime.base import _HeterogenousMetaEstimator
from sktime.detection.base import BaseDetector
from sktime.registry import scitype


class DetectorPipeline(_HeterogenousMetaEstimator, BaseDetector):
    """Pipeline for time series anomaly, changepoint detection, segmentation.

    Parameters
    ----------
    steps : list of sktime transformers and detectors, or
        list of tuples (str, estimator) of ``sktime`` transformers or detectors.
        The list must contain exactly one forecaster.
        These are "blueprint" transformers resp forecasters,
        detector/transformer states do not change when ``fit`` is called.

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of ``sktime`` transformers or detectors
        clones of estimators in ``steps`` which are fitted in the pipeline
        is always in (str, estimator) format, even if ``steps`` is just a list
        strings not passed in ``steps`` are replaced by unique generated strings
        i-th transformer in ``steps_`` is clone of i-th in ``steps``
    estimator_ : estimator, reference to the first non-transformer in ``steps_``

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sktime.detection.lof import SubLOF
    >>> from sktime.transformations.series.detrend import Detrender
    >>>
    >>> n = 100
    >>> x = pd.Series(np.linspace(0, 5, n) + np.random.normal(0, 0.1, size=n))
    >>> x.at[50] = 100
    >>>
    >>> pipeline = Detrender() * SubLOF(n_neighbors=5, window_size=5, novelty=True)
    >>> pipeline.fit(x)
    DetectorPipeline(...)
    >>> y_hat = pipeline.transform(x)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "fkiraly",
        # estimator type
        # --------------
        "learning_type": "unsupervised",
    }

    # for default get_params/set_params from _HeterogenousMetaEstimator
    # _steps_attr points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_attr = "_steps"
    # if the estimator is fittable, _HeterogenousMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in a different attribute, _steps_fitted_attr
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _steps_fitted_attr = "steps_"

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_steps(steps, allow_postproc=False)

        tags_to_clone = ["learning_type", "task"]
        # we do not clone X-y-must-have-same-index, since transformers can
        #   create indices, and that behaviour is not tag-inspectable
        self.clone_tags(self.estimator_, tags_to_clone)

        # init must be called at the end so task is properly set
        super().__init__()

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
        TransformedTargetForecaster object,
            concatenation of ``other`` (first) with ``self`` (last).
            not nested, contains only non-DetectorPipeline ``sktime`` steps
        """
        from sktime.transformations.base import BaseTransformer
        from sktime.transformations.compose import TransformerPipeline

        _, ests = zip(*self.steps_)
        names = tuple(self._get_estimator_names(self.steps))
        if isinstance(other, TransformerPipeline):
            _, trafos_o = zip(*other.steps_)
            names_o = tuple(other._get_estimator_names(other.steps))
            new_names = names_o + names
            new_ests = trafos_o + ests
        elif isinstance(other, BaseTransformer):
            new_names = (type(other).__name__,) + names
            new_ests = (other,) + ests
        elif self._is_name_and_est(other, BaseTransformer):
            other_name = other[0]
            other_trafo = other[1]
            new_names = (other_name,) + names
            new_ests = (other_trafo,) + ests
        else:
            return NotImplemented

        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_ests)):
            return DetectorPipeline(steps=list(new_ests))
        else:
            return DetectorPipeline(steps=list(zip(new_names, new_ests)))

    @property
    def estimator_(self):
        """Return reference to the detector in the pipeline.

        Valid after _fit.
        """
        return self.steps_[-1][1]

    def _get_pipeline_scitypes(self, estimators):
        """Get list of scityes (str) from names/estimator list."""
        return [scitype(x[1], raise_on_unknown=False) for x in estimators]

    def _get_first_detector_index(self, estimators):
        """Get the index of the first forecaster in the list."""
        sts = self._get_pipeline_scitypes(estimators)
        for i, s in enumerate(sts):
            if s != "transformer":
                return i
        return -1

    def _check_steps(self, estimators, allow_postproc=False):
        """Check Steps.

        Parameters
        ----------
        estimators : list of estimators, or list of (name, estimator) pairs
        allow_postproc : bool, optional, default=False
            whether transformers after the detector are allowed

        Returns
        -------
        step : list of (name, estimator) pairs, estimators are cloned (not references)
            if estimators was a list of (str, estimator) tuples, then just cloned
            if was a list of estimators, then str are generated via _get_estimator_names

        Raises
        ------
        TypeError if names in ``estimators`` are not unique
        TypeError if estimators in ``estimators`` are not all forecaster or transformer
        TypeError if there is not exactly one forecaster in ``estimators``
        TypeError if not allow_postproc and forecaster is not last estimator
        """
        self_name = type(self).__name__
        if not isinstance(estimators, list):
            msg = (
                f"steps in {self_name} must be list of estimators, "
                f"or (string, estimator) pairs, "
                f"the two can be mixed; but, found steps of type {type(estimators)}"
            )
            raise TypeError(msg)

        estimator_tuples = self._get_estimator_tuples(estimators, clone_ests=True)
        names, estimators = zip(*estimator_tuples)

        # validate names
        self._check_names(names)

        ann_ind = self._get_first_detector_index(estimator_tuples)

        if not allow_postproc and ann_ind != len(estimators) - 1:
            TypeError(
                f"in {self_name}, last estimator must be a time series detector, "
                f"but found a transformer"
            )

        # Shallow copy
        return estimator_tuples

    def _iter_transformers(self, reverse=False, an_idx=-1):
        # exclude final forecaster
        steps = self.steps_[:an_idx]

        if reverse:
            steps = reversed(steps)

        for idx, (name, transformer) in enumerate(steps):
            yield idx, name, transformer

    def __len__(self):
        """Return the length of the Pipeline."""
        return len(self.steps)

    @property
    def named_steps(self):
        """Map the steps to a dictionary."""
        return dict(self._steps)

    @property
    def _steps(self):
        return self._get_estimator_tuples(self.steps, clone_ests=False)

    @_steps.setter
    def _steps(self, value):
        self.steps = value

    def _components(self, base_class=None):
        """Return references to all state changing BaseObject type attributes.

        This *excludes* the blue-print-like components passed in the __init__.

        Caution: this method returns *references* and not *copies*.
            Writing to the reference will change the respective attribute of self.

        Parameters
        ----------
        base_class : class, optional, default=None, must be subclass of BaseObject
            if None, behaves the same as ``base_class=BaseObject``
            if not None, return dict collects descendants of ``base_class``

        Returns
        -------
        dict with key = attribute name, value = reference to attribute
        dict contains all attributes of ``self`` that inherit from ``base_class``, and:
            whose names do not contain the string "__", e.g., hidden attributes
            are not class attributes, and are not hyper-parameters (``__init__`` args)
        """
        import inspect

        from sktime.base import BaseObject

        if base_class is None:
            base_class = BaseObject
        if base_class is not None and not inspect.isclass(base_class):
            raise TypeError(f"base_class must be a class, but found {type(base_class)}")
        # if base_class is not None and not issubclass(base_class, BaseObject):
        #     raise TypeError("base_class must be a subclass of BaseObject")

        fitted_estimator_tuples = self.steps_

        comp_dict = {name: comp for (name, comp) in fitted_estimator_tuples}
        return comp_dict

    def _fit(self, X, y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        y : pd.Series, optional
            ground truth detections for training if detector is supervised

        Returns
        -------
        self : returns a reference to self

        State change
        ------------
        creates fitted model (attributes ending in "_")
        """
        # skip transformers if X is ignored
        # condition 1 for ignoring X: X is None and required in fit of 1st transformer
        first_trafo = self.steps_[0][1]
        cond1 = len(self.steps_) > 1 and first_trafo.get_tag("requires_X")
        cond1 = cond1 and X is None

        # X ignored = condition 1
        skip_trafos = cond1
        self.skip_trafos_ = skip_trafos

        # If X is ignored, just ignore the transformers and pass through to forecaster
        if not skip_trafos:
            # transform X
            for step_idx, name, transformer in self._iter_transformers():
                t = transformer.clone()
                X = t.fit_transform(X=X, y=y)
                self.steps_[step_idx] = (name, t)

        # fit detector
        name, detector = self.steps_[-1]
        f = detector.clone()
        f.fit(X=X, y=y)
        self.steps_[-1] = (name, f)

        return self

    def _transform(self, X=None, y=None):
        # If X is not given or ignored, just passthrough the data without transformation
        if not self.skip_trafos_:
            for _, _, transformer in self._iter_transformers():
                X = transformer.transform(X=X, y=y)
        return X

    def _predict(self, X):
        """Predict on test/deployment data.

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
        X = self._transform(X=X)
        return self.estimator_.predict(X)

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
        X = self._transform(X=X)
        return self.estimator_.predict_segments(X)

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
        X = self._transform(X=X)
        return self.estimator_.predict_points(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for detectors.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        import datetime

        from sklearn.preprocessing import StandardScaler

        from sktime.detection.lof import SubLOF
        from sktime.transformations.series.adapt import TabularToSeriesAdaptor
        from sktime.transformations.series.detrend import Detrender
        from sktime.transformations.series.exponent import ExponentTransformer

        lof = SubLOF(
            n_neighbors=5, window_size=datetime.timedelta(days=25), novelty=True
        )
        STEPS1 = [
            ("transformer", TabularToSeriesAdaptor(StandardScaler())),
            ("anomaly", lof),
        ]
        params1 = {"steps": STEPS1}

        STEPS2 = [
            ("transformer", ExponentTransformer()),
            ("anomaly", lof),
        ]
        params2 = {"steps": STEPS2}

        params3 = {"steps": [Detrender(), lof]}

        return [params1, params2, params3]


# todo 1.0.0 - remove alias, i.e., remove this line
AnnotatorPipeline = DetectorPipeline
