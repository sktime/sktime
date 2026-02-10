The `libs` folder contains libraries in `sktime`, namely:

* libraries distributed with `sktime`. These are maintained libraries meant for public and direct use. They can be used without `sktime`, and are also used in dedicated `sktime` estimators.
* private vendor forks. These are complete or partial vendor forks of other libraries, intended for use through `sktime` but not directly.


# libraries distributed with `sktime`

This folder contains libraries directly distributed with, and maintained by, `sktime`.

* `fracdiff` - a package implementing fractional differentiation of time series,
  a la "Advances in Financial Machine Learning" by M. Prado.
  Unofficial fork of abandoned package from July 2024,
  see [issue 6700](https://github.com/sktime/sktime/issues/6700).

* `pykalman` - a package implementing the Kálmán Filter and variants.
  Unofficial fork of abandoned package from June 2024 onwards,
  see [pykalman issue 109](https://github.com/pykalman/pykalman/issues/109).

* `vmdpy` - a package implementing Variational Mode Decomposition.
  Official fork, `vmdpy` is maintained in `sktime` since August 2023.


# private vendor forks in `sktime`

* `granite_ttm` - a package implementing TinyTimeMixer.
  Unofficial fork of package which is not available on pypi.

* `momentfm` - a package implementing the `momentfm` library, unofficial fork
  maintained since April 2025.

* `time_llm` - partial fork of the `time_llm` package, from [KimMeen/time-LLM](https://github.com/KimMeen/Time-LLM). Unofficial fork of partial code specific to the forecaster. An official package on pypi is not available.

* `timemoe` - partial fork of `time-moe` package, from [Time-MoE/Time-MoE](https://github.com/Time-MoE/Time-MoE). Unofficial fork of partial code specific to the forecaster. An official package on pypi is not available.

* `timesfm` - partial fork of TimesFM, adapted from [google-research/timesfm](https://github.com/google-research/timesfm). This is an unofficial fork created to address the lack of recent updates of `timesfm` package on [pypi](https://pypi.org/project/timesfm/) and the instability caused by significant interface changes in recent versions without prior deprecation warnings. The fork has minimal dependencies and focuses on the core features required for compatibility with the `sktime` forecaster.

* `uni2ts` - a package implementing the MOIRAIForecaster. Unofficial fork of
 the package with minimal dependencies and code specific to the forecaster.
 Official package available at [pypi](https://pypi.org/project/uni2ts/).


# Snippets from other libraries

The `libs` folder contains also some private snippets from other libraries,
in folders starting with underscore. These should not be accessed by users of `sktime` directly.

* `_aws_fortuna-enbpi` - Parts of the `EnbPI` class from aws-fortuna.
  The installation of the original package is not working due to dependency
  mismatches.

* `_keras_self_attention` - fork of some layers from the [abandoned package `keras-self-attention`](https://github.com/CyberZHG/keras-self-attention), archived in March 2024.
