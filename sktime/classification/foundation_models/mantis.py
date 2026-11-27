# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Mantis foundation-model time series classifier."""

__author__ = ["vedantag17"]
__all__ = ["MantisClassifier"]

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.classification.base import BaseClassifier
from sktime.utils.singleton import _multiton


class MantisClassifier(BaseClassifier):
    """Time series classifier using the Mantis foundation model.

    Mantis [1]_ is a family of lightweight, calibrated foundation models
    designed for time series classification. This adapter wraps Mantis's
    ``MantisTrainer`` scikit-learn-like interface as an sktime ``BaseClassifier``.

    Two usage modes are supported:

    * **Fine-tuning** (default): the Mantis backbone and/or classification head
      are fine-tuned on the supplied training data.
    * **Zero-shot** (``fine_tuning_type="head"`` with very few epochs): only the
      linear classification head is trained on top of frozen Mantis embeddings.

    Parameters
    ----------
    checkpoint : str or None, default="paris-noah/MantisV2"
        Hugging Face checkpoint to load via ``network.from_pretrained``.
        Supported checkpoints:

        * ``"paris-noah/MantisV2"``  (MantisV2 backbone — **recommended**)
        * ``"paris-noah/MantisPlus"``  (MantisV1+ backbone)
        * ``"paris-noah/Mantis-8M"``  (MantisV1 backbone, smallest)

        Pass ``None`` to use a randomly initialized backbone (no pre-training).
    model_version : {"v2", "v1"}, default="v2"
        Mantis architecture family.

        * ``"v2"`` — use with ``"paris-noah/MantisV2"``.
        * ``"v1"`` — use with ``"paris-noah/Mantis-8M"`` or
          ``"paris-noah/MantisPlus"``.
    fine_tuning_type : {"full", "head", "adapter_head", "scratch"}, \
default="full"
        Which parameters to update during training:

        * ``"full"`` — fine-tune the entire network (best accuracy).
        * ``"head"`` — train only the classification head on frozen embeddings
          (fastest, suitable as a zero-shot baseline).
        * ``"adapter_head"`` — train a learnable adapter and the head.
        * ``"scratch"`` — train everything from random initialisation.
    seq_len : int, default=512
        Sequence length passed to Mantis (must be a multiple of 32).
        Input time series that differ in length are resized to this value via
        linear interpolation.
    num_epochs : int, default=100
        Number of fine-tuning epochs.
    batch_size : int, default=64
        Batch size used during training.
    predict_batch_size : int, default=256
        Batch size used during inference.
    base_learning_rate : float, default=2e-4
        Initial learning rate for the AdamW optimizer.
    learning_rate_adjusting : bool, default=True
        Whether to use the cosine learning-rate schedule built into Mantis.
    device : str, default="auto"
        Torch device.  ``"auto"`` resolves to ``"cuda"`` when a GPU is
        available, otherwise ``"cpu"``.
    ignore_deps : bool, default=False
        Skip soft-dependency checks (useful for testing without ``mantis-tsfm``
        installed).

    References
    ----------
    .. [1] Feofanov et al., "Mantis: Lightweight Calibrated Foundation Model
       for User-Friendly Time Series Classification", 2025.
       https://arxiv.org/abs/2502.15637

    Examples
    --------
    >>> from sktime.classification.foundation_models.mantis import MantisClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_type="numpy3d")
    >>> X_test, y_test = load_unit_test(split="test", return_type="numpy3d")
    >>> clf = MantisClassifier(num_epochs=5, batch_size=16)
    >>> clf.fit(X_train, y_train)  # doctest: +SKIP
    MantisClassifier(...)
    >>> y_pred = clf.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["vedantag17"],
        "maintainers": ["vedantag17"],
        "python_dependencies": ["mantis-tsfm>=1.0.0", "torch"],
        # estimator type
        # --------------
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "numpy1D",
        "capability:multivariate": True,
        "capability:missing_values": False,
        "property:randomness": "stochastic",
        "serialization:native_artifacts": (
            "_network_",
            "_fine_tuned_model_",
        ),
        "serialization:skip": ("_trainer_",),
        # CI and testing tags
        # -------------------
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_deepcopy_fitted",
            "test_deepcopy_fitted_predict",
        ],
    }

    def __init__(
        self,
        checkpoint="paris-noah/MantisV2",
        model_version="v2",
        fine_tuning_type="full",
        seq_len=512,
        num_epochs=100,
        batch_size=64,
        predict_batch_size=256,
        base_learning_rate=2e-4,
        learning_rate_adjusting=True,
        device="auto",
        ignore_deps=False,
    ):
        self.checkpoint = checkpoint
        self.model_version = model_version
        self.fine_tuning_type = fine_tuning_type
        self.seq_len = seq_len
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.predict_batch_size = predict_batch_size
        self.base_learning_rate = base_learning_rate
        self.learning_rate_adjusting = learning_rate_adjusting
        self.device = device
        self.ignore_deps = ignore_deps

        super().__init__()

    def _fit(self, X, y):
        """Fit the Mantis classifier to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_channels, series_length)
            Training time series.
        y : np.ndarray of shape (n_instances,)
            Class labels.

        Returns
        -------
        self
        """
        from sklearn.preprocessing import LabelEncoder

        if not self.ignore_deps:
            _check_soft_dependencies("mantis-tsfm>=1.0.0", "torch", severity="error")

        # Validate seq_len
        if self.seq_len < 1 or self.seq_len % 32 != 0:
            raise ValueError(
                "seq_len must be a positive multiple of 32 for Mantis. "
                f"Got seq_len={self.seq_len}."
            )

        # Resolve device
        self._device_ = self._check_device()

        # Encode labels to [0, n_classes-1]
        self._label_encoder_ = LabelEncoder()
        y_encoded = self._label_encoder_.fit_transform(y).astype(np.int64)

        # Resize X to seq_len
        X_resized = self._resize(X)

        # Store channel count for post-pickle architecture reconstruction
        self._n_channels_ = X.shape[1]

        # Load a fresh (fine-tunable) trainer — NOT from the multiton cache,
        # because the multiton is for the zero-shot/frozen forecaster pattern.
        # Each fit call gets its own fine-tuned model.
        self._trainer_ = self._build_trainer()

        self._trainer_.fit(
            X_resized,
            y_encoded,
            fine_tuning_type=self.fine_tuning_type,
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            base_learning_rate=self.base_learning_rate,
            learning_rate_adjusting=self.learning_rate_adjusting,
        )
        self._fit_fine_tuning_type_ = getattr(
            self._trainer_, "fine_tuning_type", self.fine_tuning_type
        )
        self._fine_tuned_model_ = getattr(self._trainer_, "fine_tuned_model", None)
        self._network_ = getattr(self._trainer_, "network", None)

        return self

    def _predict(self, X):
        """Predict class labels for test instances.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_channels, series_length)

        Returns
        -------
        y_pred : np.ndarray of shape (n_instances,)
        """
        self._ensure_trainer_loaded()
        X_resized = self._resize(X)
        y_encoded = self._trainer_.predict(
            X_resized, batch_size=self.predict_batch_size
        )
        return self._label_encoder_.inverse_transform(y_encoded.astype(int))

    def _predict_proba(self, X):
        """Predict class probabilities for test instances.

        Parameters
        ----------
        X : np.ndarray of shape (n_instances, n_channels, series_length)

        Returns
        -------
        proba : np.ndarray of shape (n_instances, n_classes)
        """
        self._ensure_trainer_loaded()
        X_resized = self._resize(X)
        return self._trainer_.predict_proba(
            X_resized, batch_size=self.predict_batch_size
        )

    def _resize(self, X):
        """Resize time series to ``self.seq_len`` via linear interpolation."""
        if X.shape[-1] == self.seq_len:
            return X.astype(np.float32)

        import torch
        import torch.nn.functional as F

        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_resized = F.interpolate(
            X_tensor, size=self.seq_len, mode="linear", align_corners=False
        )
        return X_resized.numpy()

    def _check_device(self):
        """Resolve the torch device string."""
        if not self.ignore_deps:
            _check_soft_dependencies("torch", severity="error")
        try:
            import torch

            if self.device == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            return self.device
        except ImportError:
            return "cpu"

    def _get_network_kwargs(self):
        """Return kwargs used to build the Mantis backbone."""
        return {
            "checkpoint": self.checkpoint,
            "model_version": self.model_version,
            "seq_len": self.seq_len,
            "device": self._device_,
        }

    def _build_trainer(self):
        """Instantiate a fresh MantisTrainer with the chosen backbone.

        For **feature-extraction / zero-shot** (``fine_tuning_type="head"``),
        the frozen backbone weights are reused from the multiton cache so that
        the heavy model is only downloaded and kept in RAM once.

        For all fine-tuning types that update backbone weights (``"full"``,
        ``"scratch"``, ``"adapter_head"``), an independent copy of the cached
        backbone is made so that fine-tuning one instance does not affect
        others that share the cache.

        We use ``torch.save`` / ``torch.load`` on the ``state_dict()`` rather
        than ``copy.deepcopy`` because the Mantis backbone contains
        ``staticmethod`` objects that standard pickle (and therefore
        ``deepcopy``) cannot handle.
        """
        import io

        import torch
        from mantis.trainer import MantisTrainer

        # Load the base network (potentially from cache)
        network = _CachedMantisNetwork(
            key=str(sorted(self._get_network_kwargs().items())),
            network_kwargs=self._get_network_kwargs(),
        ).load_network()

        # When fine-tuning backbone weights, work on an independent copy.
        # Build a fresh network of the same class, then transplant the weights
        # via state_dict so we never deepcopy unpicklable staticmethod objects.
        if self.fine_tuning_type in ("full", "scratch"):
            from mantis.architecture import MantisV1, MantisV2

            model_version = self.model_version
            seq_len = self.seq_len
            device = self._device_

            if model_version == "v1":
                network_copy = MantisV1(seq_len=seq_len, device=device)
            elif model_version == "v2":
                network_copy = MantisV2(device=device)
            else:
                raise ValueError(
                    f"model_version must be 'v1' or 'v2', got '{model_version}'."
                )

            # Copy weights via an in-memory byte buffer (avoids staticmethod)
            buf = io.BytesIO()
            torch.save(network.state_dict(), buf)
            buf.seek(0)
            network_copy.load_state_dict(torch.load(buf, weights_only=True))
            network = network_copy

        return MantisTrainer(device=self._device_, network=network)

    def _ensure_trainer_loaded(self):
        """Reload trainer wrapper from serialized native artifacts if needed."""
        if not hasattr(self, "_trainer_") or self._trainer_ is None:
            if not getattr(self, "is_fitted", False):
                return  # not fitted yet, nothing to restore

            trainer = self._build_trainer()
            network_artifact = getattr(self, "_network_", None)
            if network_artifact is not None:
                trainer.network = network_artifact

            model_artifact = getattr(self, "_fine_tuned_model_", None)
            if model_artifact is not None:
                trainer.fine_tuned_model = model_artifact
            trainer.fine_tuning_type = getattr(
                self, "_fit_fine_tuning_type_", self.fine_tuning_type
            )

            self._trainer_ = trainer

    def _create_torch_artifact(self, name):
        """Construct Mantis module architectures for deserialization."""
        if name == "_network_":
            return self._build_trainer().network

        if name == "_fine_tuned_model_":
            trainer = self._build_trainer()
            trainer.network = self._network_

            n_classes = len(self._label_encoder_.classes_)
            dummy_X = np.zeros(
                (n_classes, self._n_channels_, self.seq_len), dtype=np.float32
            )
            dummy_y = np.arange(n_classes, dtype=np.int64)
            trainer.fit(
                dummy_X,
                dummy_y,
                fine_tuning_type=self._fit_fine_tuning_type_,
                num_epochs=0,
                batch_size=n_classes,
                learning_rate_adjusting=False,
            )
            return trainer.fine_tuned_model

        raise ValueError(f"Unknown torch artifact {name!r}.")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return. If no special
            parameters are defined for a value, the ``"default"`` set is used.

        Returns
        -------
        params : dict or list of dict
        """
        # Use head-only fine-tuning with few epochs so tests finish quickly.
        # ``fine_tuning_type="full"`` deep-copies the MantisV2 network which
        # contains a non-picklable staticmethod; it is excluded from test
        # params until the upstream library resolves that limitation.
        # seq_len=64 gives 64/32=2 patches, satisfying MantisV2's internal
        # std() requirement (seq_len=32 collapses to a single patch → NaN).
        params1 = {
            "checkpoint": None,  # random init — no HF download in tests
            "model_version": "v2",
            "fine_tuning_type": "head",
            "seq_len": 64,
            "num_epochs": 2,
            "batch_size": 4,
            "predict_batch_size": 8,
            "learning_rate_adjusting": False,
            "device": "cpu",
        }
        params2 = {
            "checkpoint": None,
            "model_version": "v1",
            "fine_tuning_type": "head",
            "seq_len": 64,
            "num_epochs": 2,
            "batch_size": 4,
            "predict_batch_size": 8,
            "learning_rate_adjusting": False,
            "device": "cpu",
        }
        return [params1, params2]


@_multiton
class _CachedMantisNetwork:
    """Cached Mantis backbone (multiton keyed on model config).

    Storing the backbone in a shared cache means that the HF download and
    model initialization happen at most once per unique configuration, even
    when many ``MantisClassifier`` instances are created.

    The classifier takes a *deep-copy* of this cached network before updating
    its weights, so the cached version always stays in the pre-trained state.
    """

    def __init__(self, key, network_kwargs):
        self.key = key
        self.network_kwargs = network_kwargs
        self._network = None

    def load_network(self):
        """Load (or return cached) Mantis backbone."""
        if self._network is not None:
            return self._network

        from mantis.architecture import MantisV1, MantisV2

        model_version = self.network_kwargs["model_version"]
        seq_len = self.network_kwargs["seq_len"]
        device = self.network_kwargs["device"]
        checkpoint = self.network_kwargs["checkpoint"]

        if model_version == "v1":
            network = MantisV1(seq_len=seq_len, device=device)
        elif model_version == "v2":
            network = MantisV2(device=device)
        else:
            raise ValueError(
                f"model_version must be 'v1' or 'v2', got '{model_version}'."
            )

        if checkpoint is not None:
            network = network.from_pretrained(checkpoint)

        self._network = network
        return self._network
