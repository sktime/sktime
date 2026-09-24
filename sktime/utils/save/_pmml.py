# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""PMML export/import for sktime estimators.

PMML (Predictive Model Markup Language) is a vendor-neutral XML format
for sharing trained models across tools and languages. Useful when you
want to deploy without shipping a full Python/sktime stack.

PMML v4.4 has built-in support for ARIMA and Exponential Smoothing.
For other estimators the sklearn2pmml or nyoka backends are used.

See Also
--------
* PMML spec: http://dmg.org/pmml/v4-4-1/GeneralStructure.html
* sklearn2pmml: https://github.com/jpmml/sklearn2pmml
* nyoka: https://github.com/SoftwareAG/nyoka
"""

__author__ = ["ojuschugh1"]

from sktime.utils.dependencies import _check_soft_dependencies


def save_to_pmml(estimator, path, pmml_backend="sklearn2pmml", **kwargs):
    """Export a fitted estimator to a PMML file.

    Two backends are supported:

    * ``"sklearn2pmml"`` (default) — works for most sklearn-compatible
      estimators.  Requires ``pip install sklearn2pmml`` and a Java
      runtime.
    * ``"nyoka"`` — has native support for ARIMA and ExponentialSmoothing.
      Requires ``pip install nyoka``.

    Parameters
    ----------
    estimator : fitted sktime or sklearn-compatible estimator
    path : str or Path
        Output file path. A ``.pmml`` suffix is added if missing.
    pmml_backend : str, default="sklearn2pmml"
        Which exporter to use. ``"sklearn2pmml"`` or ``"nyoka"``.
    **kwargs
        Passed through to the backend export function.

    Returns
    -------
    None

    Raises
    ------
    ImportError
        If the chosen backend package is not installed.
    TypeError
        If ``estimator`` is not a fitted estimator.
    NotFittedError
        If the estimator hasn't been fitted yet (sktime estimators only).
    ValueError
        If ``pmml_backend`` is not one of the supported values.

    Examples
    --------
    >>> from sktime.datasets import load_basic_motions
    >>> from sktime.classification.feature_based import SummaryClassifier
    >>> from sktime.utils.save import save_to_pmml, load_from_pmml
    >>> X_train, y_train = load_basic_motions(split="TRAIN")  # doctest: +SKIP
    >>> clf = SummaryClassifier()  # doctest: +SKIP
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    >>> save_to_pmml(clf, path="my_model.pmml")  # doctest: +SKIP
    >>> loaded = load_from_pmml("my_model.pmml")  # doctest: +SKIP
    >>> preds = loaded.predict(X_test)  # doctest: +SKIP
    """
    from pathlib import Path

    from sktime.base import BaseEstimator as SktimeBase

    _supported = {"sklearn2pmml", "nyoka"}
    if pmml_backend not in _supported:
        raise ValueError(
            f"pmml_backend must be one of {_supported}, got: '{pmml_backend}'"
        )

    is_sktime = isinstance(estimator, SktimeBase)
    if not is_sktime and not hasattr(estimator, "fit"):
        raise TypeError(
            f"Expected a fitted estimator (sktime or sklearn-compatible), "
            f"got {type(estimator)}"
        )

    if is_sktime and not estimator.is_fitted:
        from sktime.exceptions import NotFittedError

        raise NotFittedError("Estimator must be fitted before exporting to PMML.")

    path = Path(path)
    if path.suffix != ".pmml":
        path = path.with_suffix(".pmml")

    if pmml_backend == "sklearn2pmml":
        _save_sklearn2pmml(estimator, path, **kwargs)
    else:
        _save_nyoka(estimator, path, **kwargs)


def _save_sklearn2pmml(estimator, path, **kwargs):
    """Export using the sklearn2pmml backend."""
    _check_soft_dependencies("sklearn2pmml", severity="error")

    from sklearn2pmml import sklearn2pmml
    from sklearn2pmml.pipeline import PMMLPipeline

    # sklearn2pmml needs a PMMLPipeline; wrap if not already one
    pipeline = (
        estimator
        if isinstance(estimator, PMMLPipeline)
        else PMMLPipeline([("estimator", estimator)])
    )

    try:
        sklearn2pmml(pipeline, str(path), **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"sklearn2pmml export failed for {type(estimator).__name__}: {e}"
        ) from e


def _save_nyoka(estimator, path, **kwargs):
    """Export using the nyoka backend."""
    _check_soft_dependencies("nyoka", severity="error")

    import nyoka

    cls_name = type(estimator).__name__
    module = type(estimator).__module__

    try:
        if "forecasting" in module:
            _nyoka_forecaster(estimator, path, nyoka, cls_name, **kwargs)
        else:
            _nyoka_sklearn(estimator, path, nyoka, **kwargs)
    except Exception as e:
        raise RuntimeError(f"nyoka export failed for {cls_name}: {e}") from e


def _nyoka_forecaster(estimator, path, nyoka, cls_name, **kwargs):
    """Try nyoka's native time-series exporters, then fall back."""
    if (
        hasattr(nyoka, "ExponentialSmoothingToPMML")
        and "ExponentialSmoothing" in cls_name
    ):
        pmml = nyoka.ExponentialSmoothingToPMML(estimator, **kwargs)
        with open(str(path), "w") as f:
            pmml.export(f, 0)
        return

    if hasattr(nyoka, "ArimaToPMML") and ("ARIMA" in cls_name or "Arima" in cls_name):
        pmml = nyoka.ArimaToPMML(estimator, **kwargs)
        with open(str(path), "w") as f:
            pmml.export(f, 0)
        return

    # not a recognised forecaster type, fall back to generic sklearn exporter
    _nyoka_sklearn(estimator, path, nyoka, **kwargs)


def _nyoka_sklearn(estimator, path, nyoka, **kwargs):
    """Fall back to nyoka's generic sklearn-compatible exporter."""
    if not hasattr(nyoka, "skl_to_pmml"):
        raise RuntimeError(
            "nyoka.skl_to_pmml not found — check your nyoka version "
            "or use pmml_backend='sklearn2pmml' instead."
        )
    nyoka.skl_to_pmml(estimator, [], [], str(path), **kwargs)


def load_from_pmml(path):
    """Load a PMML model from disk and wrap it for prediction.

    Requires ``pypmml`` (``pip install pypmml``).

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    PmmlWrapper

    Raises
    ------
    ImportError
        If ``pypmml`` is not installed.
    FileNotFoundError
        If the file doesn't exist.

    Examples
    --------
    >>> from sktime.utils.save import load_from_pmml
    >>> model = load_from_pmml("my_model.pmml")  # doctest: +SKIP
    >>> preds = model.predict(X_test)  # doctest: +SKIP
    """
    _check_soft_dependencies("pypmml", severity="error")

    from pathlib import Path

    path = Path(path)
    if not path.exists():
        # try adding .pmml before giving up
        path_ext = path.with_suffix(".pmml")
        if path_ext.exists():
            path = path_ext
        else:
            raise FileNotFoundError(f"PMML file not found: {path}")

    return PmmlWrapper(str(path))


class PmmlWrapper:
    """Wraps a pypmml Model for straightforward predict calls.

    Parameters
    ----------
    path : str
        Path to a ``.pmml`` file.
    """

    def __init__(self, path):
        _check_soft_dependencies("pypmml", severity="error")

        from pypmml import Model

        self._path = path
        self._model = Model.fromFile(path)

    def predict(self, X):
        """Run inference and return predictions.

        Parameters
        ----------
        X : pd.DataFrame or dict
            DataFrame columns must match the PMML model's input field
            names.  numpy arrays are not accepted because PMML field
            names cannot be reliably inferred from positional indices.

        Returns
        -------
        np.ndarray
        """
        import numpy as np
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            results = [self._model.predict(row.to_dict()) for _, row in X.iterrows()]
        elif isinstance(X, np.ndarray):
            raise TypeError(
                "PmmlWrapper.predict requires a pd.DataFrame with "
                "named columns matching the PMML model's input fields."
                " Convert your array first: "
                "pd.DataFrame(X, columns=feature_names)"
            )
        elif isinstance(X, dict):
            results = [self._model.predict(X)]
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        preds = []
        for r in results:
            if isinstance(r, dict) and r:
                # exclude probability(...) keys, pick the prediction field
                non_prob = sorted(
                    k
                    for k in r
                    if not (isinstance(k, str) and k.startswith("probability("))
                )
                key = non_prob[0] if non_prob else sorted(r)[0]
                preds.append(r[key])
            else:
                preds.append(r)
        return np.array(preds)

    def predict_proba(self, X):
        """Return class probabilities when the PMML model exports them.

        Falls back to ``predict`` when no probability outputs are found.

        Parameters
        ----------
        X : pd.DataFrame or dict
            DataFrame columns must match the PMML model's input field
            names.  numpy arrays are not accepted because PMML field
            names cannot be reliably inferred from positional indices.

        Returns
        -------
        np.ndarray
        """
        import numpy as np
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            rows = [row.to_dict() for _, row in X.iterrows()]
        elif isinstance(X, np.ndarray):
            raise TypeError(
                "PmmlWrapper.predict_proba requires a pd.DataFrame "
                "with named columns matching the PMML model's input "
                "fields.  Convert your array first: "
                "pd.DataFrame(X, columns=feature_names)"
            )
        elif isinstance(X, dict):
            rows = [X]
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        results = [self._model.predict(row) for row in rows]

        if results and isinstance(results[0], dict):
            # pypmml returns keys like "probability(0)", "probability(1)"
            prob_keys = sorted(k for k in results[0] if k.startswith("probability("))
            if prob_keys:
                return np.array([[r.get(k, 0.0) for k in prob_keys] for r in results])

        return self.predict(X)

    def get_model_info(self):
        """Return basic metadata about the loaded PMML model.

        Returns
        -------
        dict with keys ``"model_class"`` and ``"path"``
        """
        return {
            "model_class": type(self._model).__name__,
            "path": self._path,
        }

    def __repr__(self):
        return f"PmmlWrapper(path='{self._path}')"
