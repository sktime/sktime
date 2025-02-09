# libraries distributed with `sktime`

This folder contains libraries directly distributed with, and maintained by, `sktime`.

* `fracdiff` - a package implementing fractional differentiation of time series,
  a la "Advances in Financial Machine Learning" by M. Prado.
  Unofficial fork of abandoned package from July 2024,
  see [issue 6700](https://github.com/sktime/sktime/issues/6700).

* `granite_ttm` - a package implementing TinyTimeMixer.
  Unofficial fork of package which is not available on pypi.

* `pykalman` - a package implementing the Kálmán Filter and variants.
  Unofficial fork of abandoned package from June 2024 onwards,
  see [pykalman issue 109](https://github.com/pykalman/pykalman/issues/109).

* `timesfm` - a package implementing TimesFM, adapted from [google-research/timesfm](https://github.com/google-research/timesfm). This is an unofficial fork created to address the lack of recent updates of `timesfm` package on [pypi](https://pypi.org/project/timesfm/) and the instability caused by significant interface changes in recent versions without prior deprecation warnings. The fork has minimal dependencies and focuses on the core features required for compatibility with the `sktime` forecaster.

* `uni2ts` - a package implementing the MOIRAIForecaster. Unofficial fork of
 the package with minimal dependencies and code specific to the forecaster.
 Official package available at [pypi](https://pypi.org/project/uni2ts/).

* `vmdpy` - a package implementing Variational Mode Decomposition.
  Official fork, `vmdpy` is maintained in `sktime` since August 2023.

# Snippets from other libraries:

This folder contains also some private snippets from other libraries,
in folders starting with underscore. These should not be accessed by users of `sktime` directly.

* `_aws_fortuna-enbpi` - Parts of the `EnbPI` class from aws-fortuna.
  The installation of the original package is not working due to dependency
  mismatches.
