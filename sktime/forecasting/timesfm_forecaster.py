# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TimesFM (Time Series Foundation Model)."""

__author__ = ["rajatsen91", "geetu040"]
# rajatsen91 for google-research/timesfm

import os

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.utils.singleton import _multiton


class TimesFMForecaster(_BaseGlobalForecaster):
    """
    Implementation of TimesFM (Time Series Foundation Model) for Zero-Shot Forecasting.

    TimesFM (Time Series Foundation Model) is a pretrained time-series foundation model
    developed by Google Research for time-series forecasting. This method has been
    proposed in [2]_ and official code is given at [1]_.

    TimesFM can be used either as a locally maintained package in sktime or directly
    from the source package, allowing users to leverage either their own
    environment or the latest updates from the source package `timesfm`.

    The class offers two flags for handling dependencies and source package behavior:

    - ``use_source_package``: Determines the source of the package code:
      False for the vendor fork in ``sktime`` with its default dependencies.
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

    repo_id : str, optional (default="google/timesfm-1.0-200m")
        The identifier for the model repository.
        The default model is the 200M parameter version.
    input_patch_len : int, optional (default=32)
        The fixed length of input patches that the model processes.
        This parameter is fixed to 1280 for the 200M model and should not be changed.
    output_patch_len : int, optional (default=128)
        The fixed length of output patches that the model generates.
        This parameter is fixed to 1280 for the 200M model and should not be changed.
    num_layers : int, optional (default=20)
        The number of layers in the model architecture.
        This parameter is fixed to 1280 for the 200M model and should not be changed.
    model_dims : int, optional (default=1280)
        The dimensionality of the model.
        This parameter is fixed to 1280 for the 200M model and should not be changed.
    per_core_batch_size : int, optional (default=32)
        The batch size to be used per core during model inference.
    backend : str, optional (default="cpu")
        The computational backend to be used,
        which can be one of "cpu", "gpu", or "tpu".
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
        If True, the model will be loaded directly from the source package ``timesfm``.
        This also enforces a version bound for ``timesfm`` to be <1.2.0.
        This setting is useful if the latest updates from the source package are needed,
        bypassing the local version of the package.
    ignore_deps : bool, default=False
        If True, dependency checks will be ignored, and the user is expected to handle
        the installation of required packages manually. If False, the class will enforce
        the default dependencies required for the vendor library or the pypi
        package, as described above, via ``use_source_package``.

    References
    ----------
    .. [1] https://github.com/google-research/timesfm
    .. [2] Das, A., Kong, W., Sen, R., & Zhou, Y. (2024).
    A decoder-only foundation model for time-series forecasting. CoRR.

    Examples
    --------
    >>> from sktime.forecasting.timesfm_forecaster import TimesFMForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> forecaster = TimesFMForecaster(
    ...     context_len=32,
    ...     horizon_len=8,
    ... ) # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP

    >>> from sktime.forecasting.timesfm_forecaster import TimesFMForecaster
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
    >>> # train and predict
    >>> forecaster.fit(y, fh=[1, 2, 3]) # doctest: +SKIP
    >>> y_pred = forecaster.predict() # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["rajatsen91", "geetu040"],
        # rajatsen91 for google-research/timesfm
        "maintainers": ["geetu040"],
        # when relaxing deps, check whether the extra test in
        # test_timesfm.py are still needed
        "python_version": ">=3.10,<3.11",
        "env_marker": "sys_platform=='linux'",
        "python_dependencies": [
            "tensorflow",
            "einshape",
            "jax",
            "praxis",
            "huggingface-hub",
            "paxml",
            "utilsforecast",
        ],
        # estimator type
        # --------------
        "y_inner_mtype": [
            "pd.Series",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
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

        if not self.ignore_deps:
            if self.use_source_package:
                # Use timesfm with a version bound if use_source_package is True
                # todo 0.37.0: Regularly check whether timesfm version can be updated
                # if changed, also needs to be changed in docstring
                self.set_tags(python_dependencies=["timesfm<1.2.0"])
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
                    "capability:global_forecasting": False,
                }
            )

        # Set environment variables for JAX backend based on CPU, GPU, or TPU
        os.environ["JAX_PLATFORM_NAME"] = backend
        os.environ["JAX_PLATFORMS"] = backend

        super().__init__()

    def _fit(self, y, X, fh):
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

        self.tfm = _CachedTimesFM(
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
            "backend": self.backend,
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

    def _predict(self, fh, X, y=None):
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        if max(fh._values.values) > self._horizon_len:
            raise ValueError(
                f"Error in {self.__class__.__name__}, the forecast horizon exceeds the"
                f" specified horizon_len of {self._horizon_len}. Change the horizon_len"
                " when initializing the model or try another forecasting horizon."
            )

        _y = y if self._global_forecasting else self._y

        # multi-index conversion goes here
        if isinstance(_y.index, pd.MultiIndex):
            hist = _frame2numpy(_y).squeeze(2)
        else:
            hist = np.expand_dims(_y.values, axis=0)

        # hist.shape: (batch_size, n_timestamps)

        pred, _ = self.tfm.forecast(hist)

        # converting pred datatype

        batch_size, n_timestamps = pred.shape

        if isinstance(_y.index, pd.MultiIndex):
            ins = np.array(list(np.unique(_y.index.droplevel(-1)).repeat(n_timestamps)))
            ins = [ins[..., i] for i in range(ins.shape[-1])] if ins.ndim > 1 else [ins]

            idx = (
                ForecastingHorizon(range(1, n_timestamps + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values.tolist()
                * pred.shape[0]
            )
            index = pd.MultiIndex.from_arrays(
                ins + [idx],
                names=_y.index.names,
            )
            pred = pd.DataFrame(
                # batch_size * num_timestamps
                pred.ravel(),
                index=index,
                columns=_y.columns,
            )
        else:
            index = (
                ForecastingHorizon(range(1, n_timestamps + 1))
                .to_absolute(self._cutoff)
                ._values
            )
            pred = pd.Series(
                # batch_size * num_timestamps
                pred.ravel(),
                index=index,
                name=_y.name,
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
            },
            {
                "context_len": 64,
                "horizon_len": 32,
                "freq": 0,
                "verbose": False,
            },
        ]
        params_no_broadcasting = [
            dict(p, **{"broadcasting": True}) for p in test_params
        ]
        test_params.extend(params_no_broadcasting)
        return test_params


def _same_index(data):
    data = data.groupby(level=list(range(len(data.index.levels) - 1))).apply(
        lambda x: x.index.get_level_values(-1)
    )
    assert data.map(lambda x: x.equals(data.iloc[0])).all(), (
        "All series must has the same index"
    )
    return data.iloc[0], len(data.iloc[0])


def _frame2numpy(data):
    idx, length = _same_index(data)
    arr = np.array(data.values, dtype=np.float32).reshape(
        (-1, length, len(data.columns))
    )
    return arr


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
            else:
                from sktime.libs.timesfm import TimesFm

            self.tfm = TimesFm(**self.timesfm_kwargs)
            self.tfm.load_from_checkpoint(repo_id=self.repo_id)

        return self.tfm
