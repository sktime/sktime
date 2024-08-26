# Fracdiff: Super-fast Fractional Differentiation

[![python versions](https://img.shields.io/pypi/pyversions/fracdiff.svg)](https://pypi.org/project/fracdiff)
[![version](https://img.shields.io/pypi/v/fracdiff.svg)](https://pypi.org/project/fracdiff)
[![CI](https://github.com/fracdiff/fracdiff/actions/workflows/ci.yml/badge.svg)](https://github.com/fracdiff/fracdiff/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fracdiff/fracdiff/branch/main/graph/badge.svg)](https://codecov.io/gh/fracdiff/fracdiff)
[![dl](https://img.shields.io/pypi/dm/fracdiff)](https://pypi.org/project/fracdiff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Documentation](https://fracdiff.github.io/fracdiff/)

***Fracdiff*** performs fractional differentiation of time-series,
a la "Advances in Financial Machine Learning" by M. Prado.
Fractional differentiation processes time-series to a stationary one while preserving memory in the original time-series.
Fracdiff features super-fast computation and scikit-learn compatible API.

![spx](./examples/fig/spx.png)

## What is fractional differentiation?

See [M. L. Prado, "Advances in Financial Machine Learning"][prado].

## Installation

```sh
pip install fracdiff
```

## Features

### Functionalities

- [`fdiff`][doc-fdiff]: A function that extends [`numpy.diff`](https://numpy.org/doc/stable/reference/generated/numpy.diff.html) to fractional differentiation.
- [`sklearn.Fracdiff`][doc-sklearn.Fracdiff]: A scikit-learn [transformer](https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html) to compute fractional differentiation.
- [`sklearn.FracdiffStat`][doc-sklearn.FracdiffStat]: `Fracdiff` plus automatic choice of differentiation order that makes time-series stationary.
- [`torch.fdiff`][doc-torch.fdiff]: A functional that extends [`torch.diff`](https://pytorch.org/docs/stable/generated/torch.diff.html) to fractional differentiation.
- [`torch.Fracdiff`][doc-torch.Fracdiff]: A module that computes fractional differentiation.

[doc-fdiff]: https://fracdiff.github.io/fracdiff/generated/fracdiff.fdiff.html
[doc-sklearn.Fracdiff]: https://fracdiff.github.io/fracdiff/generated/fracdiff.sklearn.Fracdiff.html
[doc-sklearn.FracdiffStat]: https://fracdiff.github.io/fracdiff/generated/fracdiff.sklearn.FracdiffStat.html
[doc-torch.fdiff]: https://fracdiff.github.io/fracdiff/generated/fracdiff.torch.fdiff.html
[doc-torch.Fracdiff]: https://fracdiff.github.io/fracdiff/generated/fracdiff.torch.Fracdiff.html

### Speed

Fracdiff is blazingly fast.

The following graphs show that *Fracdiff* computes fractional differentiation much faster than the "official" implementation.

It is especially noteworthy that execution time does not increase significantly as the number of time-steps (`n_samples`) increases, thanks to NumPy engine.

![time](https://user-images.githubusercontent.com/24503967/128821902-d38c2f46-989c-44e7-bd71-899f95553696.png)

The following tables of execution times (in unit of ms) show that *Fracdiff* can be ~10000 times faster than the "official" implementation.

|   n_samples | fracdiff        | official            |
|------------:|:----------------|:--------------------|
|         100 | 0.675 +-0.086   | 20.008 +-1.472      |
|        1000 | 5.081 +-0.426   | 135.613 +-3.415     |
|       10000 | 50.644 +-0.574  | 1310.033 +-17.708   |
|      100000 | 519.969 +-8.166 | 13113.457 +-105.274 |

|   n_features | fracdiff       | official             |
|-------------:|:---------------|:---------------------|
|            1 | 5.081 +-0.426  | 135.613 +-3.415      |
|           10 | 6.146 +-0.247  | 1350.161 +-15.195    |
|          100 | 6.903 +-0.654  | 13675.023 +-193.960  |
|         1000 | 13.783 +-0.700 | 136610.030 +-540.572 |

(Run on Ubuntu 20.04, Intel(R) Xeon(R) CPU E5-2673 v3 @ 2.40GHz. See [fracdiff/benchmark](https://github.com/fracdiff/benchmark/releases/tag/1115171075) for details.)

## How to use

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fracdiff/fracdiff/blob/main/examples/example_howto.ipynb)

### Fractional differentiation

A function [`fdiff`](https://fracdiff.github.io/fracdiff/#fdiff) calculates fractional differentiation.
This is an extension of `numpy.diff` to a fractional order.

```python
import numpy as np
from fracdiff import fdiff

a = np.array([1, 2, 4, 7, 0])
fdiff(a, 0.5)
# array([ 1.       ,  1.5      ,  2.875    ,  4.6875   , -4.1640625])
np.array_equal(fdiff(a, n=1), np.diff(a, n=1))
# True

a = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
fdiff(a, 0.5, axis=0)
# array([[ 1. ,  3. ,  6. , 10. ],
#        [-0.5,  3.5,  3. ,  3. ]])
fdiff(a, 0.5, axis=-1)
# array([[1.    , 2.5   , 4.375 , 6.5625],
#        [0.    , 5.    , 3.5   , 4.375 ]])
```

### Scikit-learn API

#### Preprocessing by fractional differentiation

A transformer class [`Fracdiff`](https://fracdiff.github.io/fracdiff/#id1) performs fractional differentiation by its method `transform`.

```python
from fracdiff.sklearn import Fracdiff

X = ...  # 2d time-series with shape (n_samples, n_features)

f = Fracdiff(0.5)
X = f.fit_transform(X)
```

For example, 0.5th differentiation of S&P 500 historical price looks like this:

![spx](./examples/fig/spx.png)

[`Fracdiff`](https://fracdiff.github.io/fracdiff/#id1) is compatible with scikit-learn API.
One can imcorporate it into a pipeline.

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X, y = ...  # Dataset

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('fracdiff', Fracdiff(0.5)),
    ('regressor', LinearRegression()),
])
pipeline.fit(X, y)
```

#### Fractional differentiation while preserving memory

A transformer class [`FracdiffStat`](https://fracdiff.github.io/fracdiff/#fracdiffstat) finds the minumum order of fractional differentiation that makes time-series stationary.
Differentiated time-series with this order is obtained by subsequently applying `transform` method.
This series is interpreted as a stationary time-series keeping the maximum memory of the original time-series.

```python
from fracdiff.sklearn import FracdiffStat

X = ...  # 2d time-series with shape (n_samples, n_features)

f = FracdiffStat()
X = f.fit_transform(X)
f.d_
# array([0.71875 , 0.609375, 0.515625])
```

The result for Nikkei 225 index historical price looks like this:

![nky](./examples/fig/nky.png)


### PyTorch API

One can fracdiff a PyTorch tensor. One can enjoy strong GPU acceleration.

```py
from fracdiff.torch import fdiff

input = torch.tensor(...)
output = fdiff(input, 0.5)
```

```py
from fracdiff.torch import Fracdiff

module = Fracdiff(0.5)
module
# Fracdiff(0.5, dim=-1, window=10, mode='same')

input = torch.tensor(...)
output = module(input)
```

### More Examples

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fracdiff/fracdiff/blob/main/examples/example_prado.ipynb)

More examples are provided [here](examples/example_prado.ipynb).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fracdiff/fracdiff/blob/main/examples/example_exercise.ipynb)

Example solutions of exercises in Section 5 of "Advances in Financial Machine Learning" are provided [here](examples/example_exercise.ipynb).

## Contributing

Any contributions are more than welcome.

The maintainer (simaki) is not making further enhancements and appreciates pull requests to make them.
See [Issue](https://github.com/fracdiff/fracdiff/issues) for proposed features.
Please take a look at [CONTRIBUTING.md](.github/CONTRIBUTING.md) before creating a pull request.

## References

- [Marcos Lopez de Prado, "Advances in Financial Machine Learning", Wiley, (2018).][prado]

[prado]: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
