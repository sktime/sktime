# Vendored From

This directory vendors the upstream `tsbootstrap` library source.

- Upstream repository: https://github.com/astrogilda/tsbootstrap
- Upstream release: `v0.1.0`
- Upstream commit: `1ee1be3e52ab74bca27674a022f7a303af1266d1`
- Vendored date: 2026-04-26
- Upstream license: MIT (see `LICENSE` in this directory)

## Local adjustments

1. `__init__.py` registers a `sys.modules` alias (`tsbootstrap`) so upstream
   absolute imports resolve in the vendored namespace `sktime.libs.tsbootstrap`.
2. `__init__.py` falls back to `__version__ = "0+vendored"` when the standalone
   `tsbootstrap` distribution is not installed.
