# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TabICL-TS forecaster."""

__author__ = ["keshavnanda"]
__all__ = ["TabICLForecaster"]

from sktime.forecasting.base._delegate import _DelegatedForecaster


class TabICLForecaster(_DelegatedForecaster):
    """TabICL foundation model forecaster via reduction.

    This forecaster interfaces ``tabicl.TabICLRegressor`` and delegates forecasting
    to ``make_reduction``. TabICL is a tabular in-context learning foundation model
    with a scikit-learn compatible regressor API.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of TabICL ensemble members.
    norm_methods : str, list of str, or None, default=None
        Normalization methods used by TabICL.
    feat_shuffle_method : {"none", "shift", "random", "latin"}, default="latin"
        Feature permutation strategy for TabICL ensemble diversity.
    outlier_threshold : float, default=4.0
        Z-score threshold for TabICL outlier clipping.
    batch_size : int or None, default=8
        Batch size for TabICL inference.
    kv_cache : bool or {"kv", "repr"}, default=False
        Whether TabICL caches training data computations for prediction.
    model_path : str, path-like, or None, default=None
        Path to a TabICL checkpoint.
    allow_auto_download : bool, default=True
        Whether TabICL may download the checkpoint if it is not found locally.
    checkpoint_version : str, default="tabicl-regressor-v2-20260212.ckpt"
        TabICL checkpoint version.
    device : str, torch.device, or None, default=None
        Device used by TabICL, e.g., "cpu", "cuda", or "mps".
    use_amp : bool or "auto", default="auto"
        Whether TabICL uses automatic mixed precision.
    use_fa3 : bool or "auto", default="auto"
        Whether TabICL uses Flash Attention 3 where available.
    offload_mode : {"auto", "gpu", "cpu", "disk"} or bool, default="auto"
        TabICL offloading strategy for memory-heavy intermediate tensors.
    disk_offload_dir : str or None, default=None
        Directory for TabICL disk offloading.
    random_state : int or None, default=42
        Random seed passed to TabICL.
    n_jobs : int or None, default=None
        Number of PyTorch CPU threads used by TabICL.
    verbose : bool, default=False
        Whether TabICL prints inference details.
    inference_config : dict or object, default=None
        Advanced TabICL inference configuration.
    strategy : {"recursive", "direct", "multioutput", "dirrec"}, default="recursive"
        Reduction strategy passed to ``make_reduction``.
    window_length : int, default=10
        Window length used by the reducer.
    pooling : {"local", "global"}, default="local"
        Pooling mode passed to ``make_reduction``.
    windows_identical : bool, default=True
        Whether direct reduction models use identical windows across horizons.
    ignore_deps : bool, default=False
        If True, dependency checks are skipped.

    Attributes
    ----------
    forecaster_ : sktime forecaster
        Fitted reduction forecaster wrapping ``tabicl.TabICLRegressor``.

    References
    ----------
    .. [1] https://github.com/soda-inria/tabicl
    .. [2] Qu, Jingang and others (2026). TabICLv2: A better, faster, scalable,
       and open tabular foundation model. arXiv:2602.11139.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tabicl import TabICLForecaster
    >>> y = load_airline()
    >>> forecaster = TabICLForecaster(n_estimators=1)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    TabICLForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _delegate_name = "forecaster_"

    _tags = {
        "authors": ["keshavnanda"],
        "maintainers": ["keshavnanda"],
        "python_dependencies": "tabicl",
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": ["pd.Series", "pd.DataFrame", "None"],
        "capability:exogenous": True,
        "capability:insample": False,
        "capability:missing_values": False,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "tests:vm": True,
        "tests:skip_by_name": [
            "test_persistence_via_pickle",
            "test_save_estimators_to_file",
        ],
    }

    def __init__(
        self,
        n_estimators=8,
        norm_methods=None,
        feat_shuffle_method="latin",
        outlier_threshold=4.0,
        batch_size=8,
        kv_cache=False,
        model_path=None,
        allow_auto_download=True,
        checkpoint_version="tabicl-regressor-v2-20260212.ckpt",
        device=None,
        use_amp="auto",
        use_fa3="auto",
        offload_mode="auto",
        disk_offload_dir=None,
        random_state=42,
        n_jobs=None,
        verbose=False,
        inference_config=None,
        strategy="recursive",
        window_length=10,
        pooling="local",
        windows_identical=True,
        ignore_deps=False,
    ):
        self.n_estimators = n_estimators
        self.norm_methods = norm_methods
        self.feat_shuffle_method = feat_shuffle_method
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.allow_auto_download = allow_auto_download
        self.checkpoint_version = checkpoint_version
        self.device = device
        self.use_amp = use_amp
        self.use_fa3 = use_fa3
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.inference_config = inference_config
        self.strategy = strategy
        self.window_length = window_length
        self.pooling = pooling
        self.windows_identical = windows_identical
        self.ignore_deps = ignore_deps

        if ignore_deps:
            self.set_tags(python_dependencies=[])

        super().__init__()

    def _get_tabicl_regressor_class(self):
        """Return TabICL regressor class."""
        from tabicl import TabICLRegressor

        return TabICLRegressor

    def _get_tabicl_params(self):
        """Return parameters passed to ``tabicl.TabICLRegressor``."""
        return {
            "n_estimators": self.n_estimators,
            "norm_methods": self.norm_methods,
            "feat_shuffle_method": self.feat_shuffle_method,
            "outlier_threshold": self.outlier_threshold,
            "batch_size": self.batch_size,
            "kv_cache": self.kv_cache,
            "model_path": self.model_path,
            "allow_auto_download": self.allow_auto_download,
            "checkpoint_version": self.checkpoint_version,
            "device": self.device,
            "use_amp": self.use_amp,
            "use_fa3": self.use_fa3,
            "offload_mode": self.offload_mode,
            "disk_offload_dir": self.disk_offload_dir,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "inference_config": self.inference_config,
        }

    def _make_delegate(self):
        """Construct reduction forecaster wrapping TabICL."""
        from sktime.forecasting.compose import make_reduction

        TabICLRegressor = self._get_tabicl_regressor_class()
        regressor = TabICLRegressor(**self._get_tabicl_params())

        return make_reduction(
            estimator=regressor,
            strategy=self.strategy,
            window_length=self.window_length,
            scitype="tabular-regressor",
            pooling=self.pooling,
            windows_identical=self.windows_identical,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        self.forecaster_ = self._make_delegate()
        self.forecaster_.fit(y=y, X=X, fh=fh)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {
            "n_estimators": 1,
            "batch_size": 1,
            "allow_auto_download": False,
            "ignore_deps": True,
        }
