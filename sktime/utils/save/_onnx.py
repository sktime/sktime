# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""ONNX export/import for sktime estimators.

Useful when you train on one architecture (e.g. x86) and need to run
inference on another (e.g. ARM), since ONNX is architecture-independent.

Note: ONNX conversion is handled by ``skl2onnx``, which supports
sklearn-compatible estimators. Sktime estimators that wrap or expose
a sklearn-compatible interface (e.g. via ``_estimator`` attribute) can
be converted by passing the inner sklearn estimator directly.
"""

__author__ = ["ojuschugh1"]

from sktime.utils.dependencies import _check_soft_dependencies


def save_to_onnx(estimator, path=None, initial_types=None, target_opset=None, **kwargs):
    """Export a fitted sklearn-compatible estimator to ONNX format.

    Uses ``skl2onnx`` under the hood. Install with ``pip install skl2onnx onnx``.

    Note: ``skl2onnx.convert_sklearn`` supports sklearn-compatible estimators.
    For sktime estimators that internally wrap sklearn (e.g. classifiers with
    an ``_estimator`` attribute), pass the inner sklearn estimator directly.

    Parameters
    ----------
    estimator : fitted sklearn-compatible estimator
        Must be compatible with ``skl2onnx.convert_sklearn``.
    path : str, Path, or None, default=None
        Where to save. If None, returns the model in memory instead.
        A ``.onnx`` suffix is appended if not already present.
    initial_types : list of tuples, or None
        ONNX input spec, e.g. ``[('X', FloatTensorType([None, 10]))]``.
        Guessed from the estimator when None.
    target_opset : int or None
        ONNX opset to target. Defaults to skl2onnx's choice.
    **kwargs
        Forwarded to ``skl2onnx.convert_sklearn``.

    Returns
    -------
    onnx.ModelProto or None
        The model object when ``path=None``, otherwise None.

    Raises
    ------
    ImportError
        If ``skl2onnx`` or ``onnx`` are not installed.
    TypeError
        If ``estimator`` is not a fitted sklearn-compatible estimator.
    NotFittedError
        If the estimator hasn't been fitted yet (sktime estimators only).

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> from sktime.utils.save import save_to_onnx, load_from_onnx
    >>> import numpy as np
    >>> X, y = make_classification(n_samples=50, n_features=4, random_state=0)
    >>> X = X.astype(np.float32)
    >>> clf = LogisticRegression().fit(X, y)
    >>> onnx_model = save_to_onnx(clf)  # in-memory  # doctest: +SKIP
    >>> save_to_onnx(clf, path="my_model")  # saves my_model.onnx  # doctest: +SKIP
    >>> wrapper = load_from_onnx("my_model.onnx")  # doctest: +SKIP
    """
    # validate input before checking soft deps — gives clearer errors
    from sktime.base import BaseEstimator as SktimeBaseEstimator

    if isinstance(estimator, SktimeBaseEstimator):
        if not estimator.is_fitted:
            from sktime.exceptions import NotFittedError

            raise NotFittedError("Estimator must be fitted before exporting to ONNX.")

    if not hasattr(estimator, "fit"):
        raise TypeError(
            f"Expected a fitted sklearn-compatible estimator, got {type(estimator)}. "
            "The estimator must have a 'fit' method and be compatible with "
            "skl2onnx.convert_sklearn."
        )

    _check_soft_dependencies("skl2onnx", "onnx", severity="error")

    from skl2onnx import convert_sklearn

    if initial_types is None:
        initial_types = _infer_onnx_initial_types(estimator)

    try:
        onnx_model = convert_sklearn(
            estimator,
            initial_types=initial_types,
            target_opset=target_opset,
            **kwargs,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert {type(estimator).__name__} to ONNX: {e}\n"
            "skl2onnx only supports sklearn-compatible estimators. "
            "For sktime estimators, extract the inner sklearn model "
            "first (e.g. estimator._estimator or estimator.estimator_)"
            ", or pass initial_types manually."
        ) from e

    if path is None:
        return onnx_model

    from pathlib import Path

    import onnx

    path = Path(path)
    if path.suffix != ".onnx":
        path = path.with_suffix(".onnx")
    onnx.save(onnx_model, str(path))
    return None


def load_from_onnx(path_or_bytes, input_name=None, output_name=None):
    """Load an ONNX model from disk or memory and wrap it for prediction.

    Requires ``onnxruntime`` (``pip install onnxruntime``).

    Parameters
    ----------
    path_or_bytes : str, Path, bytes, or onnx.ModelProto
    input_name : str or None
        Name of the input node. Uses the first one if not given.
    output_name : str or None
        Name of the output node. Uses the first one if not given.

    Returns
    -------
    OnnxWrapper

    Raises
    ------
    ImportError
        If ``onnxruntime`` is not installed.
    FileNotFoundError
        If a file path was given but the file doesn't exist.

    Examples
    --------
    >>> from sktime.utils.save import load_from_onnx
    >>> wrapper = load_from_onnx("my_model.onnx")  # doctest: +SKIP
    >>> preds = wrapper.predict(X_test)  # doctest: +SKIP
    """
    _check_soft_dependencies("onnxruntime", severity="error")

    from pathlib import Path

    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        if not path.exists():
            # try appending .onnx before giving up
            path_ext = path.with_suffix(".onnx")
            if path_ext.exists():
                path = path_ext
            else:
                raise FileNotFoundError(f"ONNX file not found: {path_or_bytes}")
        with open(path, "rb") as f:
            model_bytes = f.read()
    elif hasattr(path_or_bytes, "SerializeToString"):
        model_bytes = path_or_bytes.SerializeToString()
    elif isinstance(path_or_bytes, bytes):
        model_bytes = path_or_bytes
    else:
        raise TypeError(
            f"Expected a path, bytes, or onnx.ModelProto, got {type(path_or_bytes)}"
        )

    return OnnxWrapper(model_bytes, input_name=input_name, output_name=output_name)


class OnnxWrapper:
    """Wraps an onnxruntime InferenceSession for easy predict calls.

    Parameters
    ----------
    model_bytes : bytes
    input_name : str or None
    output_name : str or None
    """

    def __init__(self, model_bytes, input_name=None, output_name=None):
        _check_soft_dependencies("onnxruntime", severity="error")

        import onnxruntime as rt

        self._model_bytes = model_bytes
        self._session = rt.InferenceSession(model_bytes)
        self._input_name = input_name or self._session.get_inputs()[0].name
        self._output_name = output_name or self._session.get_outputs()[0].name

    def predict(self, X):
        """Run inference and return predictions.

        Parameters
        ----------
        X : array-like or pd.DataFrame

        Returns
        -------
        np.ndarray
        """
        import numpy as np

        if hasattr(X, "values"):
            X = X.values
        X = np.array(X, dtype=np.float32)
        return self._session.run([self._output_name], {self._input_name: X})[0]

    def predict_proba(self, X):
        """Return class probabilities when the model exports them.

        Raises an error if the model does not have a probability output node.

        Parameters
        ----------
        X : array-like or pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)

        Raises
        ------
        ValueError
            If the ONNX model does not expose a probability output.
        """
        import numpy as np

        if hasattr(X, "values"):
            X = X.values
        X = np.array(X, dtype=np.float32)

        outputs = self._session.get_outputs()
        if len(outputs) < 2:
            raise ValueError(
                "This ONNX model does not have a probability output node. "
                "Use predict() instead, or export the model with probability "
                "outputs enabled."
            )

        # prefer an output whose name contains "prob" or "proba"; fall back to
        # outputs[1] which is the sklearn convention for label/probabilities
        output_names = [o.name for o in outputs]
        proba_name = next(
            (n for n in output_names if "prob" in n.lower()),
            output_names[1],
        )
        proba = self._session.run([proba_name], {self._input_name: X})[0]
        # sklearn classifiers return a list of dicts {class: probability}
        if isinstance(proba, list) and isinstance(proba[0], dict):
            classes = sorted(proba[0].keys())
            proba = np.array([[p[c] for c in classes] for p in proba])
        return proba

    def get_model_bytes(self):
        """Return the raw serialized ONNX model bytes."""
        return self._model_bytes

    def __repr__(self):
        inputs = [i.name for i in self._session.get_inputs()]
        outputs = [o.name for o in self._session.get_outputs()]
        return f"OnnxWrapper(inputs={inputs}, outputs={outputs})"


def _infer_onnx_initial_types(estimator):
    """Guess the ONNX input spec from common estimator attributes."""
    from skl2onnx.common.data_types import FloatTensorType

    n_features = None
    for attr in ["n_features_in_", "n_features_", "n_columns_"]:
        if hasattr(estimator, attr):
            n_features = getattr(estimator, attr)
            break

    if n_features is None:
        raise ValueError(
            "Could not infer the number of input features from the estimator. "
            "Please pass 'initial_types' explicitly to 'save_to_onnx', e.g. "
            "[('X', FloatTensorType([None, n_features]))]."
        )
    return [("X", FloatTensorType([None, n_features]))]
