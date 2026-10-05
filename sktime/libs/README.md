The `libs` folder contains libraries in `sktime`, namely:

* libraries distributed with `sktime`. These are maintained libraries meant for public and direct use. They can be used without `sktime`, and are also used in dedicated `sktime` estimators.
* private vendor forks. These are complete or partial vendor forks of other libraries, intended for use through `sktime` but not directly.


# libraries distributed with `sktime`

This folder contains libraries directly distributed with, and maintained by, `sktime`.

* `fracdiff` - a package implementing fractional differentiation of time series,
  a la "Advances in Financial Machine Learning" by M. Prado.
  Unofficial fork of abandoned package from July 2024,
  see [issue 6700](https://github.com/sktime/sktime/issues/6700).

* `tbats` - a package implementing BATS and TBATS.
  Unofficial fork of abandoned package from September 2026 onwards,
  see [sktime issue 11097](https://github.com/sktime/sktime/issues/11097).

* `vmdpy` - a package implementing Variational Mode Decomposition.
  Official fork, `vmdpy` is maintained in `sktime` since August 2023.


# private vendor forks in `sktime`

* `chronos` - a package implementing Chronos and Chronos-Bolt.
  Unofficial fork of the `amazon-science/chronos-forecasting` package from https://github.com/amazon-science/chronos-forecasting. Licensed under Apache 2.0.

* `falcon_tst` - unofficial fork of the `ant-intl/Falcon-TST_Large`
  model code from https://huggingface.co/ant-intl/Falcon-TST_Large.
  Licensed under Apache 2.0.

* `granite_ttm` - a package implementing TinyTimeMixer.
  Unofficial fork of package which is not available on pypi.

* `kronos` - a package implementing Kronos.
  Unofficial fork of the `shiyu-coder/Kronos` model code from
  https://github.com/shiyu-coder/Kronos. Licensed under MIT.

* `mira` - partial fork of MIRA, from [microsoft/MIRA](https://github.com/microsoft/MIRA).
  Unofficial fork of partial code specific to the forecaster. An official package on
  PyPI is not available. Licensed under MIT.

* `lag_llama` - partial fork of Lag-Llama, from [time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama).
  Unofficial fork of partial code specific to the forecaster. An official package on pypi is not available.
  Licensed under Apache 2.0.

* `momentfm` - a package implementing the `momentfm` library, unofficial fork
  maintained since April 2025.

* `sundial` - partial fork of Sundial, adapted from [thuml/sundial-base-128m](https://huggingface.co/thuml/sundial-base-128m).
  Unofficial fork of partial code specific to the forecaster. An official package on pypi is not available.
  Licensed under Apache 2.0.

* `time_llm` - partial fork of the `time_llm` package, from [KimMeen/time-LLM](https://github.com/KimMeen/Time-LLM). Unofficial fork of partial code
  specific to the forecaster. An official package on pypi is not available.

* `timer` - a package implementing Timer.
  Unofficial fork of the `thuml/timer-base-84m` model code from
  https://huggingface.co/thuml/timer-base-84m.
  Licensed under Apache 2.0.

* `timer_s1` - a package implementing Timer-S1.
  Unofficial fork of the `bytedance-research/Timer-S1` model code from
  https://huggingface.co/bytedance-research/Timer-S1. Licensed under Apache 2.0.

* `timemoe` - partial fork of `time-moe` package, from [Time-MoE/Time-MoE](https://github.com/Time-MoE/Time-MoE). Unofficial fork of partial code specific to the forecaster. An official package on pypi is not available.

* `timesfm` - partial fork of TimesFM, adapted from [google-research/timesfm](https://github.com/google-research/timesfm). This is an unofficial fork created to address the lack of recent updates of `timesfm` package on [pypi](https://pypi.org/project/timesfm/) and the instability caused by significant interface changes in recent versions without prior deprecation warnings. The fork has minimal dependencies and focuses on the core features required for compatibility with the `sktime` forecaster.

* `uni2ts` - a package implementing the MOIRAIForecaster. Unofficial fork of
 the package with minimal dependencies and code specific to the forecaster.
 Official package available at [pypi](https://pypi.org/project/uni2ts/).

* `windfm` - partial fork of WindFM, from
  [shiyu-coder/WindFM](https://github.com/shiyu-coder/WindFM).
  Unofficial fork of partial code specific to the forecaster. An official
  package on pypi is not available.

* `xlstm_time` - fork of the [repository of the same name](https://github.com/muslehal/xLSTMTime), implementing the xLSTM forecaster, by `mushlehal`.


# Snippets from other libraries

The `libs` folder contains also some private snippets from other libraries,
in folders starting with underscore. These should not be accessed by users of `sktime` directly.

* `_aws_fortuna_enbpi` - Parts of the `EnbPI` class from aws-fortuna.
  The installation of the original package is not working due to dependency
  mismatches.

* `_keras_self_attention` - fork of some layers from the [abandoned package `keras-self-attention`](https://github.com/CyberZHG/keras-self-attention), archived in March 2024.

* `_torch_self_attention` - PyTorch implementation of a sequential
  self-attention layer used by the TapNet model.
