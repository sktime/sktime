"""Wrappers for tsai deep-learning classifiers exposed through sktime."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sktime.classification.base import BaseClassifier
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
# from sktime.utils.validation._dependencies import _check_soft_dependencies

import torch          # ← required for the custom loss

__author__ = ["timeseriesAI (oguiza)", "estiern", "AlexThomasMa"]


# -------------------------------------------------------------------------
# custom loss: CrossEntropy that always sees integer targets
# -------------------------------------------------------------------------
_ce = torch.nn.CrossEntropyLoss()

def _loss(pred, targ):
    """Wrapper around torch.nn.CrossEntropyLoss that casts targets to long."""
    return _ce(pred, targ.long())
# -------------------------------------------------------------------------


class _TsaiBaseClassifier(BaseClassifier):
    """Shared logic for any tsai network used as a classifier in sktime."""

    _tags = {
        "scitype:instancewise": True,
        "requires_y": True,
        "handles-tabular_X": False,
        "python_dependencies": "tsai",
    }

    def __init__(self, tsai_model_cls, n_epochs=5, bs=64, lr=1e-3, **fit_kwargs):
        self.tsai_model_cls = tsai_model_cls
        self.n_epochs = n_epochs
        self.bs = bs
        self.lr = lr
        self.fit_kwargs = fit_kwargs

        self._learn = None          # fastai Learner (tsai learner)
        self._classes = None        # np.ndarray of class labels

        super().__init__()

    # ------------------------------------------------------------------ #
    # fitting
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # _check_soft_dependencies("tsai", severity="error")
        from tsai.all import get_ts_dls, ts_learner

        # 1) Convert nested DataFrame -> 3-D NumPy array of shape (N, C, L)
        X_np = from_nested_to_3d_numpy(X)

        # 2) Encode y labels to integer indices
        classes, y_int = np.unique(y, return_inverse=True)
        self._classes = classes
        y_int = y_int.astype(np.int64)            # CrossEntropyLoss expects int64

        # 3) Build tsai DataLoaders
        dls = get_ts_dls(X_np, y_int, bs=self.bs, classification=True)

        # 4) Instantiate tsai model & fastai Learner
        n_classes = len(self._classes)
        model = self.tsai_model_cls(dls.vars, n_classes, dls.len)
        self._learn = ts_learner(dls, model, loss_func=_loss)
        self._learn.fit_one_cycle(self.n_epochs, self.lr, **self.fit_kwargs)

        return self

    # inference (direct model forward - no fastai helpers)
    # ------------------------------------------------------------------ #
    def _forward(self, X_np: np.ndarray):
        """
        Run a batch (N,C,L) through the tsai model and return
        soft-max probabilities – completely avoiding fastai’s
        `test_dl` / `Learner.predict` code-path.
        """
        import torch

        mdl      = self._learn.model.eval()
        device   = self._learn.dls.device
        xb       = torch.from_numpy(X_np).to(device).float()      # (N,C,L)
        with torch.no_grad():
            logits = mdl(xb)                                      # (N,C)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs                                              # (N,C)

    def predict(self, X: pd.DataFrame):
        X_np  = from_nested_to_3d_numpy(X)        # (N,C,L)
        probs = self._forward(X_np)               # (N,C)
        idx   = probs.argmax(axis=1)              # (N,)
        return self._classes[idx]

    def predict_proba(self, X: pd.DataFrame):
        X_np  = from_nested_to_3d_numpy(X)
        return self._forward(X_np)
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #


def _safe_import_tsai_model(path: str, name: str):
    """Internal helper so the import only happens if tsai is present."""
    import importlib
    return getattr(importlib.import_module(path), name)


class TsaiTSTClassifier(_TsaiBaseClassifier):
    """tsai Temporal Self-Attention (TST) classifier wrapped for sktime."""

    def __init__(self, **kwargs):
        TST = _safe_import_tsai_model("tsai.models.TST", "TST")  # noqa: N806
        super().__init__(TST, **kwargs)


class TsaiInceptionTimeClassifier(_TsaiBaseClassifier):
    """tsai InceptionTime classifier wrapped for sktime."""

    def __init__(self, **kwargs):
        InceptionTime = _safe_import_tsai_model(
            "tsai.models.InceptionTime", "InceptionTime"
        )
        super().__init__(InceptionTime, **kwargs)
