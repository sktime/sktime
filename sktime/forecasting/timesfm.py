# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TimesFM (Time Series Foundation Model)."""

__author__ = ["rajatsen91", "geetu040", "Me-Priyank"]
# rajatsen91 for google-research/timesfm


import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.utils.singleton import _multiton

_PYTORCH_REPO_IDS = {"google/timesfm-1.0-200m": "google/timesfm-1.0-200m-pytorch"}


def _resolve_backend(backend, use_source_package):
    """Resolve automatic backend selection, via JAX or PyTorch."""
    if backend != "auto":
        return backend

    if use_source_package:
        import jax

        return "gpu" if jax.default_backend() == "gpu" else "cpu"

    import torch

    if torch.cuda.is_available():
        return "gpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TimesFMForecaster(BaseForecaster):
    """TimesFM (Time Series Foundation Model) for Zero-Shot Forecasting.

    TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model
    developed by Google Research for time-series forecasting. This method has been
    proposed in [2]_ and official code is given at [1]_.

    By default, TimesFM runs on the ``transformers`` implementation of TimesFM,
    ``TimesFmModelForPrediction``, with the weights of the PyTorch checkpoint
    published by Google Research, converted on load. Alternatively, TimesFM can be
    used directly from the source package ``timesfm``, in its JAX version.

    The class offers two flags for handling dependencies and source package behavior:

    - ``use_source_package``: Determines the source of the package code:
      False for the ``transformers`` implementation, with its default dependencies.
      True for the source package ``timesfm``.
    - ``ignore_deps``: If set, bypasses dependency checks entirely.
      This is for users who want to manage their environment manually.

    Parameters
    ----------
    context_len : int, optional (default=None)
        The length of the input context sequence.
        If set to None, the context length is automatically computed as the smallest
        multiple of ``input_patch_len`` that is larger than the length
        of the input time series `y`.
        It should be a multiple of ``input_patch_len`` (32).
        The maximum context length currently supported is 512, but this can be
        increased in future releases.
        The input time series can have any context length,
        and padding or truncation will
        be handled by the model's inference code if necessary.

    horizon_len : int, optional (default=128)
        The length of the forecast horizon.
        If set to None, the forecast horizon is dynamically determined based on the
        provided forecasting horizon `fh`, if available.
        This can be set to any value, although it
        is generally recommended to keep it less than or equal to ``context_len`` for
        optimal performance. The model will still function
        if ``horizon_len`` exceeds ``context_len``.

    freq : int, optional (default=0)
        The frequency category of the input time series.

        - 0: High frequency, long horizon time series (e.g., daily data and above).
        - 1: Medium frequency time series (e.g., weekly, monthly data).
        - 2: Low frequency, short horizon time series (e.g., quarterly, yearly data).

        You can treat this parameter as a free parameter depending
        on your specific use case,
        although it is recommended to follow these guidelines for optimal results.

    repo_id : str or None, optional (default="google/timesfm-1.0-200m")
        The identifier for the model repository on Hugging Face.
        The default model is the 200M parameter version.
        Unless ``use_source_package=True``, the repository must contain a PyTorch
        checkpoint ``torch_model.ckpt``, as ``"google/timesfm-1.0-200m-pytorch"`` does,
        and the default ``"google/timesfm-1.0-200m"`` is loaded from its PyTorch
        version, ``"google/timesfm-1.0-200m-pytorch"``.
        If None, the model is initialized from random weights and no checkpoint is
        downloaded or restored. Random initialization is supported for the
        ``transformers`` implementation, i.e. when ``use_source_package=False``.
    input_patch_len : int, optional (default=32)
        The fixed length of input patches that the model processes.
        This parameter is fixed to 32 for the 200M model and should not be changed.
    output_patch_len : int, optional (default=128)
        The fixed length of output patches that the model generates.
        This parameter is fixed to 128 for the 200M model and should not be changed.
    num_layers : int, optional (default=20)
        The number of layers in the model architecture.
        This parameter is fixed to 20 for the 200M model and should not be changed.
    model_dims : int, optional (default=1280)
        The dimensionality of the model.
        This parameter is fixed to 1280 for the 200M model and should not be changed.
    per_core_batch_size : int, optional (default=32)
        The batch size to be used per core during model inference.
    backend : str, optional (default="cpu")
        The computational backend to be used,
        which can be one of "auto", "cpu", "gpu", or "tpu". ``"auto"`` selects
        CUDA, then MPS, then CPU, as available to PyTorch, or, if
        ``use_source_package=True``, GPU when JAX has a GPU backend and CPU otherwise.
        ``"tpu"`` is supported only if ``use_source_package=True``.
        This setting is case-sensitive.
    verbose : bool, optional (default=False)
        Whether to print detailed logs during execution.
    broadcasting : bool, default=False
        if True, multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from ``predict``.
    use_source_package : bool, default=False
        If True, the model will be loaded directly from the source package ``timesfm``,
        in its JAX version, instead of the ``transformers`` implementation.
        This also enforces a version bound for ``timesfm`` to be <1.2.0,
        which requires python 3.10 and linux.
    ignore_deps : bool, default=False
        If True, dependency checks will be ignored, and the user is expected to handle
        the installation of required packages manually. If False, the class will enforce
        the default dependencies required for the ``transformers`` implementation or
        the pypi package, as described above, via ``use_source_package``.

    References
    ----------
    .. [1] https://github.com/google-research/timesfm
    .. [2] Das, A., Kong, W., Sen, R., & Zhou, Y. (2024).
    A decoder-only foundation model for time-series forecasting. CoRR.

    Examples
    --------
    >>> from sktime.forecasting.timesfm import TimesFMForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TimesFMForecaster(
    ...     context_len=32,
    ...     horizon_len=8,
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP

    >>> from sktime.forecasting.timesfm import TimesFMForecaster
    >>> from sktime.datasets import load_tecator
    >>>
    >>> # load multi-index dataset
    >>> y = load_tecator(
    ...     return_type="pd-multiindex",
    ...     return_X_y=False
    ... )
    >>> y.drop(['class_val'], axis=1, inplace=True)
    >>>
    >>> # global forecasting on multi-index dataset
    >>> forecaster = TimesFMForecaster(
    ...     context_len=32,
    ...     horizon_len=8,
    ... ) # doctest: +SKIP
    >>>
    >>> # fit sets the context, predict uses it
    >>> forecaster.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["rajatsen91", "geetu040", "Me-Priyank"],
        # rajatsen91 for google-research/timesfm
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers[torch]>=4.52.0"],
        # estimator type
        # --------------
        "category": ["deep_learning", "foundation_model"],
        "y_inner_mtype": "pd.Series",
        "capability:multivariate": False,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "capability:unequal_length": False,
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:specific": ["sktime.forecasting.tests.test_timesfm"],
    }

    def __init__(
        self,
        context_len=None,
        horizon_len=128,
        freq=0,
        repo_id="google/timesfm-1.0-200m",
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        per_core_batch_size=32,
        backend="cpu",
        verbose=False,
        broadcasting=False,
        use_source_package=False,
        ignore_deps=False,
    ):
        self.context_len = context_len
        self._context_len = None
        self.horizon_len = horizon_len
        self._horizon_len = None
        self._context_len = None
        self.freq = freq
        self.repo_id = repo_id
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.per_core_batch_size = per_core_batch_size
        self.backend = backend
        self.verbose = verbose
        self.broadcasting = broadcasting
        self.use_source_package = use_source_package
        self.ignore_deps = ignore_deps

        super().__init__()

    def __post_init__(self):
        """Post-initialization setup."""
        self._backend = _resolve_backend(self.backend, self.use_source_package)

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        if not self.ignore_deps:
            if self.use_source_package:
                # Use timesfm with a version bound if use_source_package is True
                # todo 1.3.0: Regularly check whether timesfm version can be updated
                # if changed, also needs to be changed in docstring
                self.set_tags(
                    python_dependencies=["timesfm<1.2.0"],
                    python_version=">=3.10,<3.11",
                    env_marker="sys_platform=='linux'",
                )
        else:
            # Ignore dependencies, leave the dependency set empty
            clear_deps = {
                "python_dependencies": None,
                "python_version": None,
                "env_marker": None,
            }
            self.set_tags(**clear_deps)

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.Series",
                    "X_inner_mtype": "pd.DataFrame",
                }
            )

    def __getstate__(self):
        """Return state for pickling, excluding unpickleable TimesFM model."""
        state = self.__dict__.copy()
        if "tfm" in state:
            state["tfm"] = None
        return state

    def __setstate__(self, state):
        """Restore state, TimesFM model will be reloaded on next use."""
        self.__dict__.update(state)

    def _fit(self, y, X=None, fh=None):
        if fh is None and self.horizon_len is None:
            raise ValueError(
                "Both 'fh' and 'horizon_len' cannot be None. Provide at least one."
            )
        elif fh is not None and self.horizon_len is not None:
            fh = fh.to_relative(self.cutoff)
            self._horizon_len = max(self.horizon_len, *fh._values.values)
        elif fh is not None:
            fh = fh.to_relative(self.cutoff)
            self._horizon_len = max(*fh._values.values)
        else:
            self._horizon_len = self.horizon_len

        if self.context_len is not None:
            self._context_len = self.context_len
        else:
            # Compute context_len as the smallest multiple of input_patch_len
            # that is larger than the length of y.
            context_multiple = (len(y) // self.input_patch_len) + 1
            self._context_len = context_multiple * self.input_patch_len

        self.context = y
        self.tfm = self._load_model()

    def _load_model(self):
        """Load TimesFM model via multiton cache."""
        return _CachedTimesFM(
            key=self._get_unique_timesfm_key(),
            timesfm_kwargs=self._get_timesfm_kwargs(),
            use_source_package=self.use_source_package,
            repo_id=self.repo_id,
        ).load_from_checkpoint()

    def _get_timesfm_kwargs(self):
        """Get the kwargs for TimesFM model."""
        return {
            "context_len": self._context_len,
            "horizon_len": self._horizon_len,
            "input_patch_len": self.input_patch_len,
            "output_patch_len": self.output_patch_len,
            "num_layers": self.num_layers,
            "model_dims": self.model_dims,
            "per_core_batch_size": self.per_core_batch_size,
            "backend": self._backend,
            "verbose": self.verbose,
        }

    def _get_unique_timesfm_key(self):
        """Get unique key for TimesFM model to use in multiton."""
        repo_id = self.repo_id
        use_source_package = self.use_source_package
        kwargs = self._get_timesfm_kwargs()
        kwargs_plus_repo_id = {
            **kwargs,
            "repo_id": repo_id,
            "use_source_package": use_source_package,
        }
        return str(sorted(kwargs_plus_repo_id.items()))

    def _predict(self, fh, X=None):
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        if max(fh._values.values) > self._horizon_len:
            raise ValueError(
                f"Error in {self.__class__.__name__}, the forecast horizon exceeds the"
                f" specified horizon_len of {self._horizon_len}. Change the horizon_len"
                " when initializing the model or try another forecasting horizon."
            )

        context = self.context
        context_np = np.expand_dims(context.values, axis=0)
        # context_np.shape: (batch_size, n_timestamps)

        if not hasattr(self, "tfm") or self.tfm is None:
            self.tfm = self._load_model()

        pred, _ = self.tfm.forecast(context_np)

        # converting pred datatype

        batch_size, n_timestamps = pred.shape

        index = (
            ForecastingHorizon(range(1, n_timestamps + 1))
            .to_absolute(self._cutoff)
            ._values
        )
        pred = pd.Series(
            # batch_size * num_timestamps
            pred.ravel(),
            index=index,
            name=context.name,
        )

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = pred.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        pred = pred.loc[dateindex]

        return pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        test_params = [
            {
                "context_len": None,
                "horizon_len": 128,
                "freq": 0,
                "verbose": False,
                "repo_id": None,
                "output_patch_len": 32,
                "num_layers": 1,
                "model_dims": 16,
                "per_core_batch_size": 1,
            },
            {
                "context_len": 64,
                "horizon_len": 32,
                "freq": 0,
                "verbose": False,
                "repo_id": None,
                "output_patch_len": 32,
                "num_layers": 1,
                "model_dims": 16,
                "per_core_batch_size": 1,
            },
        ]
        params_no_broadcasting = [
            dict(p, **{"broadcasting": True}) for p in test_params
        ]
        test_params.extend(params_no_broadcasting)
        return test_params


@_multiton
class _CachedTimesFM:
    """Cached TimesFM model, to ensure only one instance exists in memory.

    TimesFM is a zero shot model and immutable, hence there will not be
    any side effects of sharing the same instance across multiple uses.
    """

    def __init__(self, key, timesfm_kwargs, use_source_package, repo_id):
        self.key = key
        self.timesfm_kwargs = timesfm_kwargs
        self.repo_id = repo_id
        self.use_source_package = use_source_package
        self.tfm = None

    def load_from_checkpoint(self):
        if self.tfm is None:
            if self.use_source_package:
                from timesfm import TimesFm

                self.tfm = TimesFm(**self.timesfm_kwargs)
            else:
                self.tfm = _TimesFMTorch(**self.timesfm_kwargs)

            if self.repo_id is not None:
                self.tfm.load_from_checkpoint(repo_id=self.repo_id)
            else:
                if not hasattr(self.tfm, "init_from_random_weights"):
                    raise NotImplementedError(
                        "repo_id=None requires the transformers TimesFM "
                        "implementation. Set use_source_package=False."
                    )
                self.tfm.init_from_random_weights()

        return self.tfm


class _TimesFMTorch:
    """TimesFM 1.0 inference with ``transformers.TimesFmModelForPrediction``.

    Replaces ``TimesFm`` of the JAX implementation in ``sktime.libs.timesfm``, with
    its methods used by ``TimesFMForecaster``, ``load_from_checkpoint``,
    ``init_from_random_weights`` and ``forecast``, and its forecasting logic:
    autoregressive decoding of ``output_patch_len`` steps at a time, on the last
    ``context_len`` values, with point forecasts from the mean head, truncated at
    zero if all inputs are non-negative.

    Parameters
    ----------
    context_len : int
        Length of the context the model forecasts from.
    horizon_len : int
        Number of steps to forecast.
    input_patch_len : int
        Length of the input patches.
    output_patch_len : int
        Number of steps forecast per decoding step.
    num_layers : int
        Number of transformer layers.
    model_dims : int
        Dimension of the model.
    per_core_batch_size : int
        Number of series forecast per forward pass.
    backend : str
        ``"gpu"`` for CUDA, or the name of a PyTorch device.
    verbose : bool
        Whether to print the repository a checkpoint is loaded from.
    """

    def __init__(
        self,
        context_len,
        horizon_len,
        input_patch_len,
        output_patch_len,
        num_layers,
        model_dims,
        per_core_batch_size,
        backend,
        verbose,
    ):
        self.context_len = context_len
        self.horizon_len = horizon_len
        self.input_patch_len = input_patch_len
        self.output_patch_len = output_patch_len
        self.num_layers = num_layers
        self.model_dims = model_dims
        self.per_core_batch_size = per_core_batch_size
        self.device = "cuda" if backend == "gpu" else backend
        self.verbose = verbose
        self.model = None

    def _build_model(self):
        from transformers import TimesFmConfig, TimesFmModelForPrediction

        config = TimesFmConfig(
            patch_length=self.input_patch_len,
            context_length=self.context_len,
            horizon_length=self.output_patch_len,
            num_hidden_layers=self.num_layers,
            hidden_size=self.model_dims,
            intermediate_size=self.model_dims,
            head_dim=self.model_dims // 16,
            num_attention_heads=16,
            use_positional_embedding=True,
        )
        return TimesFmModelForPrediction(config)

    def load_from_checkpoint(self, repo_id):
        """Load the weights of a TimesFM 1.0 PyTorch checkpoint from Hugging Face.

        Parameters
        ----------
        repo_id : str
            Hugging Face repository with the checkpoint ``torch_model.ckpt``.
            Repositories in ``_PYTORCH_REPO_IDS`` are loaded from their PyTorch
            version.
        """
        import torch
        from huggingface_hub import hf_hub_download

        repo_id = _PYTORCH_REPO_IDS.get(repo_id, repo_id)
        if self.verbose:
            print(f"Restoring checkpoint from {repo_id}.")
        path = hf_hub_download(repo_id=repo_id, filename="torch_model.ckpt")
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        """Load weights in the format of TimesFM 1.0 PyTorch checkpoints.

        Parameters
        ----------
        state_dict : dict of str to torch.Tensor
            Weights, named as in ``torch_model.ckpt`` of
            ``google/timesfm-1.0-200m-pytorch``.

        Raises
        ------
        ValueError
            If weights of the model architecture, as given by ``input_patch_len``,
            ``output_patch_len``, ``num_layers`` and ``model_dims``, are missing,
            or weights not in the model architecture are given.
        RuntimeError
            If the shapes of the weights do not match the model architecture.
        """
        model = self._build_model()
        result = model.load_state_dict(
            _convert_timesfm_state_dict(state_dict), strict=False
        )
        buffers = {name for name, _ in model.named_buffers()}
        if result.unexpected_keys or not buffers.issuperset(result.missing_keys):
            raise ValueError(
                "The TimesFM checkpoint does not match the model architecture given "
                "by input_patch_len, output_patch_len, num_layers and model_dims. "
                f"Missing weights: {result.missing_keys}, "
                f"unexpected weights: {result.unexpected_keys}."
            )
        self.model = model.to(self.device).eval()

    def init_from_random_weights(self):
        """Initialize the model with random weights, from a fixed seed."""
        import torch

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = self._build_model()
        self.model = model.to(self.device).eval()

    def forecast(self, inputs):
        """Forecast ``horizon_len`` steps ahead of each series.

        Parameters
        ----------
        inputs : sequence of 1D array-like
            The series to forecast, of possibly different lengths.

        Returns
        -------
        mean_outputs : np.ndarray of shape (len(inputs), horizon_len)
            Point forecasts.
        full_outputs : np.ndarray of shape (len(inputs), horizon_len, 10)
            Point forecasts, followed by the forecasts of the quantiles
            0.1, 0.2, ..., 0.9.
        """
        import torch

        inputs = [np.asarray(ts)[-self.context_len :] for ts in inputs]
        n_steps = -(-self.horizon_len // self.output_patch_len)

        mean_outputs = []
        full_outputs = []
        for start in range(0, len(inputs), self.per_core_batch_size):
            past_values = [
                torch.tensor(ts, dtype=torch.float32, device=self.device)
                for ts in inputs[start : start + self.per_core_batch_size]
            ]
            means = []
            fulls = []
            with torch.no_grad():
                for _ in range(n_steps):
                    output = self.model(
                        past_values=past_values, forecast_context_len=self.context_len
                    )
                    means.append(output.mean_predictions)
                    fulls.append(output.full_predictions)
                    past_values = [
                        torch.cat([ts, mean])
                        for ts, mean in zip(past_values, output.mean_predictions)
                    ]
            mean_outputs.append(torch.cat(means, dim=1)[:, : self.horizon_len].cpu())
            full_outputs.append(torch.cat(fulls, dim=1)[:, : self.horizon_len].cpu())

        mean_outputs = torch.cat(mean_outputs).numpy()
        full_outputs = torch.cat(full_outputs).numpy()
        if min(np.min(ts) for ts in inputs) >= 0:
            mean_outputs = np.maximum(mean_outputs, 0.0)
            full_outputs = np.maximum(full_outputs, 0.0)
        return mean_outputs, full_outputs


def _convert_timesfm_state_dict(state_dict):
    """Convert weights of TimesFM 1.0 PyTorch checkpoints to ``transformers`` names.

    Parameters
    ----------
    state_dict : dict of str to torch.Tensor
        Weights, named as in ``torch_model.ckpt`` of
        ``google/timesfm-1.0-200m-pytorch``.

    Returns
    -------
    converted : dict of str to torch.Tensor
        The same weights, named as in ``transformers.TimesFmModelForPrediction``,
        with each fused ``qkv_proj`` projection split into ``q_proj``, ``k_proj``
        and ``v_proj``.
    """
    converted = {}
    for key, value in state_dict.items():
        name = key.replace(".hidden_layer.0.", ".input_layer.")
        name = name.replace("stacked_transformer.layers.", "decoder.layers.")
        if name.startswith(("input_ff_layer.", "freq_emb.")):
            name = f"decoder.{name}"
        if ".qkv_proj." in name:
            for proj, part in zip(["q_proj", "k_proj", "v_proj"], value.chunk(3)):
                converted[name.replace("qkv_proj", proj)] = part
        else:
            converted[name] = value
    return converted
