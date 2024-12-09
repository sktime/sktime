# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]

__all__ = ["BaseDistribution"]

from warnings import warn

import numpy as np
import pandas as pd

from sktime.base import BaseObject
from sktime.utils.dependencies import _check_estimator_deps
from sktime.utils.pandas import df_map
from sktime.utils.warnings import warn as _warn


class BaseDistribution(BaseObject):
    """Base probability distribution."""

    # default tag values - these typically make the "safest" assumption
    _tags = {
        # packaging info
        # --------------
        "authors": "sktime developers",  # author(s) of the object
        "maintainers": "sktime developers",  # current maintainer(s) of the object
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # str or list of str, package soft dependencies
        # estimator type
        # --------------
        "object_type": "distribution",  # type of object, e.g., 'distribution'
        "reserved_params": ["index", "columns"],
        "capabilities:approx": ["energy", "mean", "var", "pdfnorm"],
        "approx_mean_spl": 1000,  # sample size used in MC estimates of mean
        "approx_var_spl": 1000,  # sample size used in MC estimates of var
        "approx_energy_spl": 1000,  # sample size used in MC estimates of energy
        "approx_spl": 1000,  # sample size used in other MC estimates
        "bisect_iter": 1000,  # max iters for bisection method in ppf
    }

    def __init__(self, index=None, columns=None):
        self.index = index
        self.columns = columns

        super().__init__()
        _check_estimator_deps(self)

        _warn(
            "The proba module in sktime is deprecated and will "
            "be fully replaced by skpro in sktime version 0.38.0. "
            "Until 0.38.0, imports from proba will continue working, "
            "defaulting to sktime.proba if skpro is not present, "
            "otherwise redirecting imports to skpro objects. "
            "To silence this message, ensure skpro is installed in the environment."
            "If using the sktime.proba module direclty, "
            "in addition, replace any imports from "
            "sktime.proba with imports from skpro.distributions.",
            obj=self,
            stacklevel=2,
        )

    @property
    def loc(self):
        """Location indexer.

        Use `my_distribution.loc[index]` for `pandas`-like row/column subsetting
        of `BaseDistribution` descendants.

        `index` can be any `pandas` `loc` compatible index subsetter.

        `my_distribution.loc[index]` or `my_distribution.loc[row_index, col_index]`
        subset `my_distribution` to rows defined by `row_index`, cols by `col_index`,
        to exactly the same/cols rows as `pandas` `loc` would subset
        rows in `my_distribution.index` and columns in `my_distribution.columns`.
        """
        return _Indexer(ref=self, method="_loc")

    @property
    def iloc(self):
        """Integer location indexer.

        Use `my_distribution.iloc[index]` for `pandas`-like row/column subsetting
        of `BaseDistribution` descendants.

        `index` can be any `pandas` `iloc` compatible index subsetter.

        `my_distribution.iloc[index]` or `my_distribution.iloc[row_index, col_index]`
        subset `my_distribution` to rows defined by `row_index`, cols by `col_index`,
        to exactly the same/cols rows as `pandas` `iloc` would subset
        rows in `my_distribution.index` and columns in `my_distribution.columns`.
        """
        return _Indexer(ref=self, method="_iloc")

    @property
    def shape(self):
        """Shape of self, a pair (2-tuple)."""
        return (len(self.index), len(self.columns))

    def _loc(self, rowidx=None, colidx=None):
        if rowidx is not None:
            row_iloc = self.index.get_indexer_for(rowidx)
        else:
            row_iloc = None
        if colidx is not None:
            col_iloc = self.columns.get_indexer_for(colidx)
        else:
            col_iloc = None
        return self._iloc(rowidx=row_iloc, colidx=col_iloc)

    def _subset_params(self, rowidx, colidx):
        params = self._get_dist_params()

        subset_param_dict = {}
        for param, val in params.items():
            if val is not None:
                arr = np.array(val)
            else:
                arr = None
            # if len(arr.shape) == 0:
            # do nothing with arr
            if len(arr.shape) >= 1 and rowidx is not None:
                arr = arr[rowidx]
            if len(arr.shape) >= 2 and colidx is not None:
                arr = arr[:, colidx]
            if np.issubdtype(arr.dtype, np.integer):
                arr = arr.astype("float")
            subset_param_dict[param] = arr
        return subset_param_dict

    def _iloc(self, rowidx=None, colidx=None):
        # distr_type = type(self.distr)
        subset_params = self._subset_params(rowidx=rowidx, colidx=colidx)
        # distr_subset = distr_type(**subset_params)

        def subset_not_none(idx, subs):
            if subs is not None:
                return idx.take(subs)
            else:
                return idx

        index_subset = subset_not_none(self.index, rowidx)
        columns_subset = subset_not_none(self.columns, colidx)

        sk_distr_type = type(self)
        return sk_distr_type(
            index=index_subset,
            columns=columns_subset,
            **subset_params,
        )

    def _get_dist_params(self):
        params = self.get_params(deep=False)
        paramnames = params.keys()
        reserved_names = ["index", "columns"]
        paramnames = [x for x in paramnames if x not in reserved_names]

        return {k: params[k] for k in paramnames}

    def to_str(self):
        """Return string representation of self."""
        params = self._get_dist_params()

        prt = f"{self.__class__.__name__}("
        for paramname, val in params.items():
            prt += f"{paramname}={val}, "
        prt = prt[:-2] + ")"

        return prt

    def _method_error_msg(self, method="this method", severity="warn", fill_in=None):
        msg = (
            f"{type(self)} does not have an implementation of the '{method}' method, "
            "via numerically exact implementation or fill-in approximation."
        )
        if fill_in is None:
            fill_in = "by an approximation via other methods"
        msg_approx = (
            f"{type(self)} does not have a numerically exact implementation of "
            f"the '{method}' method, it is "
            f"filled in {fill_in}."
        )
        if severity == "warn":
            return msg_approx
        else:
            return msg

    def _get_bc_params(self, *args, dtype=None):
        """Fully broadcast tuple of parameters given param shapes and index, columns.

        Parameters
        ----------
        args : float, int, array of floats, or array of ints (1D or 2D)
            Distribution parameters that are to be made broadcastable. If no positional
            arguments are provided, all parameters of `self` are used except for `index`
            and `columns`.
        dtype : str, optional
            broadcasted arrays are cast to all have datatype `dtype`. If None, then no
            datatype casting is done.

        Returns
        -------
        Tuple of float or integer arrays
            Each element of the tuple represents a different broadcastable distribution
            parameter.
        """
        number_of_params = len(args)
        if number_of_params == 0:
            # Handle case where no positional arguments are provided
            params = self.get_params()
            params.pop("index")
            params.pop("columns")
            args = tuple(params.values())
            number_of_params = len(args)

        if hasattr(self, "index") and self.index is not None:
            args += (self.index.to_numpy().reshape(-1, 1),)
        if hasattr(self, "columns") and self.columns is not None:
            args += (self.columns.to_numpy(),)
        bc = np.broadcast_arrays(*args)
        if dtype is not None:
            bc = [array.astype(dtype) for array in bc]
        return bc[:number_of_params]

    def pdf(self, x):
        r"""Probability density function.

        Let :math:`X` be a random variables with the distribution of `self`,
        taking values in `(N, n)` `DataFrame`-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input `x` representing :math:`x`,
        is a `DataFrame` with same columns and indices as `self`,
        and entries :math:`p_{X_{ij}}(x_{ij})`.

        Parameters
        ----------
        x : `pandas.DataFrame` or 2D np.ndarray
            representing :math:`x`, as above

        Returns
        -------
        `DataFrame` with same columns and index as `self`
            containing :math:`p_{X_{ij}}(x_{ij})`, as above
        """
        if self._has_implementation_of("log_pdf"):
            approx_method = (
                "by exponentiating the output returned by the log_pdf method, "
                "this may be numerically unstable"
            )
            warn(self._method_error_msg("pdf", fill_in=approx_method))
            return df_map(self.log_pdf(x=x))(np.exp)

        raise NotImplementedError(self._method_error_msg("pdf", "error"))

    def log_pdf(self, x):
        r"""Logarithmic probability density function.

        Numerically more stable than calling pdf and then taking logartihms.

        Let :math:`X` be a random variables with the distribution of `self`,
        taking values in `(N, n)` `DataFrame`-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input `x` representing :math:`x`,
        is a `DataFrame` with same columns and indices as `self`,
        and entries :math:`\log p_{X_{ij}}(x_{ij})`.

        If `self` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : `pandas.DataFrame` or 2D np.ndarray
            representing :math:`x`, as above

        Returns
        -------
        `DataFrame` with same columns and index as `self`
            containing :math:`\log p_{X_{ij}}(x_{ij})`, as above
        """
        if self._has_implementation_of("pdf"):
            approx_method = (
                "by taking the logarithm of the output returned by the pdf method, "
                "this may be numerically unstable"
            )
            warn(self._method_error_msg("log_pdf", fill_in=approx_method))

            return df_map(self.pdf(x=x))(np.log)

        raise NotImplementedError(self._method_error_msg("log_pdf", "error"))

    def cdf(self, x):
        """Cumulative distribution function."""
        N = self.get_tag("approx_spl")
        approx_method = (
            "by approximating the expected value by the indicator function on "
            f"{N} samples"
        )
        warn(self._method_error_msg("mean", fill_in=approx_method))

        splx = pd.concat([x] * N, keys=range(N))
        spl = self.sample(N)
        ind = splx <= spl

        return ind.groupby(level=1, sort=False).mean()

    def ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        if self._has_implementation_of("cdf"):
            from scipy.optimize import bisect

            max_iter = self.get_tag("bisect_iter")
            approx_method = (
                "by using the bisection method (scipy.optimize.bisect) on "
                f"the cdf, at {max_iter} maximum iterations"
            )
            warn(self._method_error_msg("cdf", fill_in=approx_method))

            result = pd.DataFrame(index=p.index, columns=p.columns, dtype="float")
            for ix in p.index:
                for col in p.columns:
                    d_ix = self.loc[[ix], [col]]
                    p_ix = p.loc[ix, col]

                    def opt_fun(x):
                        """Optimization function, to find x s.t. cdf(x) = p_ix."""
                        x = pd.DataFrame(x, index=[ix], columns=[col])  # noqa: B023
                        return d_ix.cdf(x).values[0][0] - p_ix  # noqa: B023

                    left_bd = -1e6
                    right_bd = 1e6
                    while opt_fun(left_bd) > 0:
                        left_bd *= 10
                    while opt_fun(right_bd) < 0:
                        right_bd *= 10
                    result.loc[ix, col] = bisect(
                        opt_fun, left_bd, right_bd, maxiter=max_iter
                    )
            return result

        raise NotImplementedError(self._method_error_msg("ppf", "error"))

    def energy(self, x=None):
        r"""Energy of self, w.r.t. self or a constant frame x.

        Let :math:`X, Y` be i.i.d. random variables with the distribution of `self`.

        If `x` is `None`, returns :math:`\mathbb{E}[|X-Y|]` (for each row),
        "self-energy" (of the row marginal distribution).
        If `x` is passed, returns :math:`\mathbb{E}[|X-x|]` (for each row),
        "energy wrt x" (of the row marginal distribution).

        Parameters
        ----------
        x : None or pd.DataFrame, optional, default=None
            if pd.DataFrame, must have same rows and columns as `self`

        Returns
        -------
        pd.DataFrame with same rows as `self`, single column `"energy"`
        each row contains one float, self-energy/energy as described above.
        """
        # we want to approximate E[abs(X-Y)]
        # if x = None, X,Y are i.i.d. copies of self
        # if x is not None, X=x (constant), Y=self

        approx_spl_size = self.get_tag("approx_energy_spl")
        approx_method = (
            "by approximating the energy expectation by the arithmetic mean of "
            f"{approx_spl_size} samples"
        )
        warn(self._method_error_msg("energy", fill_in=approx_method))

        # splx, sply = i.i.d. samples of X - Y of size N = approx_spl_size
        N = approx_spl_size
        if x is None:
            splx = self.sample(N)
            sply = self.sample(N)
        else:
            splx = pd.concat([x] * N, keys=range(N))
            sply = self.sample(N)

        # approx E[abs(X-Y)] via mean of samples of abs(X-Y) obtained from splx, sply
        spl = splx - sply
        energy = spl.apply(np.linalg.norm, axis=1, ord=1)
        energy = energy.groupby(level=1, sort=False).mean()
        energy = pd.DataFrame(energy, index=self.index, columns=["energy"])
        return energy

    def mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        approx_spl_size = self.get_tag("approx_mean_spl")
        approx_method = (
            "by approximating the expected value by the arithmetic mean of "
            f"{approx_spl_size} samples"
        )
        warn(self._method_error_msg("mean", fill_in=approx_method))

        spl = self.sample(approx_spl_size)
        return spl.groupby(level=1, sort=False).mean()

    def var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        approx_spl_size = self.get_tag("approx_var_spl")
        approx_method = (
            "by approximating the variance by the arithmetic mean of "
            f"{approx_spl_size} samples of squared differences"
        )
        warn(self._method_error_msg("var", fill_in=approx_method))

        spl1 = self.sample(approx_spl_size)
        spl2 = self.sample(approx_spl_size)
        spl = (spl1 - spl2) ** 2
        return spl.groupby(level=1, sort=False).mean()

    def pdfnorm(self, a=2):
        r"""a-norm of pdf, defaults to 2-norm.

        computes a-norm of the entry marginal pdf, i.e.,
        :math:`\mathbb{E}[p_X(X)^{a-1}] = \int p(x)^a dx`,
        where :math:`X` is a random variable distributed according to the entry marginal
        of `self`, and :math:`p_X` is its pdf

        Parameters
        ----------
        a: int or float, optional, default=2

        Returns
        -------
        pd.DataFrame with same rows and columns as `self`
        each entry is :math:`\mathbb{E}[p_X(X)^{a-1}] = \int p(x)^a dx`, see above
        """
        # special case: if a == 1, this is just the integral of the pdf, which is 1
        if a == 1:
            return pd.DataFrame(1.0, index=self.index, columns=self.columns)

        approx_spl_size = self.get_tag("approx_spl")
        approx_method = (
            f"by approximating the {a}-norm of the pdf by the arithmetic mean of "
            f"{approx_spl_size} samples"
        )
        warn(self._method_error_msg("pdfnorm", fill_in=approx_method))

        # uses formula int p(x)^a dx = E[p(X)^{a-1}], and MC approximates the RHS
        spl = [self.pdf(self.sample()) ** (a - 1) for _ in range(approx_spl_size)]
        return pd.concat(spl, axis=0).groupby(level=1, sort=False).mean()

    def _coerce_to_self_index_df(self, x):
        x = np.array(x)
        x = x.reshape(1, -1)
        df_shape = self.shape
        x = np.broadcast_to(x, df_shape)
        df = pd.DataFrame(x, index=self.index, columns=self.columns)
        return df

    def quantile(self, alpha):
        """Return entry-wise quantiles, in Proba/pred_quantiles mtype format.

        This method broadcasts as follows:
        for a scalar `alpha`, computes the `alpha`-quantile entry-wise,
        and returns as a `pd.DataFrame` with same index, and columns as in return.
        If `alpha` is iterable, multiple quantiles will be calculated,
        and the result will be concatenated column-wise (axis=1).

        The `ppf` method also computes quantiles, but broadcasts differently, in
        `numpy` style closer to `tensorflow`.
        In contrast, this `quantile` method broadcasts
        as ``sktime`` forecaster `predict_quantiles`, i.e., columns first.

        Parameters
        ----------
        alpha : float or list of float of unique values
            A probability or list of, at which quantiles are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from `self.columns`,
            second level being the values of `alpha` passed to the function.
            Row index is `self.index`.
            Entries in the i-th row, (j, p)-the column is
            the p-th quantile of the marginal of `self` at index (i, j).
        """
        if not isinstance(alpha, list):
            alpha = [alpha]

        qdfs = []
        for p in alpha:
            p = self._coerce_to_self_index_df(p)
            qdf = self.ppf(p)
            qdfs += [qdf]

        qres = pd.concat(qdfs, axis=1, keys=alpha)
        qres = qres.reorder_levels([1, 0], axis=1)

        cols = pd.MultiIndex.from_product([self.columns, alpha])
        quantiles = qres.loc[:, cols]
        return quantiles

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None

        Returns
        -------
        if `n_samples` is `None`:
        returns a sample that contains a single sample from `self`,
        in `pd.DataFrame` mtype format convention, with `index` and `columns` as `self`
        if n_samples is `int`:
        returns a `pd.DataFrame` that contains `n_samples` i.i.d. samples from `self`,
        in `pd-multiindex` mtype format convention, with same `columns` as `self`,
        and `MultiIndex` that is product of `RangeIndex(n_samples)` and `self.index`
        """

        def gen_unif():
            np_unif = np.random.uniform(size=self.shape)
            return pd.DataFrame(np_unif, index=self.index, columns=self.columns)

        # if ppf is implemented, we use inverse transform sampling
        if self._has_implementation_of("ppf"):
            if n_samples is None:
                return self.ppf(gen_unif())
            else:
                pd_smpl = [self.ppf(gen_unif()) for _ in range(n_samples)]
                df_spl = pd.concat(pd_smpl, keys=range(n_samples))
                return df_spl

        raise NotImplementedError(self._method_error_msg("sample", "error"))


class _Indexer:
    """Indexer for BaseDistribution, for pandas-like index in loc and iloc property."""

    def __init__(self, ref, method="_loc"):
        self.ref = ref
        self.method = method

    def __getitem__(self, key):
        """Getitem dunder, for use in my_distr.loc[index] an my_distr.iloc[index]."""

        def is_noneslice(obj):
            res = isinstance(obj, slice)
            res = res and obj.start is None and obj.stop is None and obj.step is None
            return res

        ref = self.ref
        indexer = getattr(ref, self.method)

        if isinstance(key, tuple):
            if not len(key) == 2:
                raise ValueError(
                    "there should be one or two keys when calling .loc, "
                    "e.g., mydist[key], or mydist[key1, key2]"
                )
            rows = key[0]
            cols = key[1]
            if is_noneslice(rows) and is_noneslice(cols):
                return ref
            elif is_noneslice(cols):
                return indexer(rowidx=rows, colidx=None)
            elif is_noneslice(rows):
                return indexer(rowidx=None, colidx=cols)
            else:
                return indexer(rowidx=rows, colidx=cols)
        else:
            return indexer(rowidx=key, colidx=None)


class _BaseTFDistribution(BaseDistribution):
    """Adapter for tensorflow-probability distributions."""

    _tags = {
        "python_dependencies": "tensorflow_probability",
        "capabilities:approx": ["energy"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
    }

    def __init__(self, index=None, columns=None, distr=None):
        self.distr = distr

        super().__init__(index=index, columns=columns)

    def __str__(self):
        return self.to_str()

    def pdf(self, x):
        r"""Probability density function.

        Let :math:`X` be a random variables with the distribution of `self`,
        taking values in `(N, n)` `DataFrame`-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input `x` representing :math:`x`,
        is a `DataFrame` with same columns and indices as `self`,
        and entries :math:`p_{X_{ij}}(x_{ij})`.

        If `self` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : `pandas.DataFrame` or 2D np.ndarray
            representing :math:`x`, as above

        Returns
        -------
        `DataFrame` with same columns and index as `self`
            containing :math:`p_{X_{ij}}(x_{ij})`, as above
        """
        if isinstance(x, pd.DataFrame):
            dist_at_x = self.loc[x.index, x.columns]
            tensor = dist_at_x.distr.prob(x.values)
            return pd.DataFrame(tensor, index=x.index, columns=x.columns)
        else:
            dist_at_x = self
            return dist_at_x.distr.prob(x)

    def log_pdf(self, x):
        r"""Logarithmic probability density function.

        Numerically more stable than calling pdf and then taking logartihms.

        Let :math:`X` be a random variables with the distribution of `self`,
        taking values in `(N, n)` `DataFrame`-s
        Let :math:`x\in \mathbb{R}^{N\times n}`.
        By :math:`p_{X_{ij}}`, denote the marginal pdf of :math:`X` at the
        :math:`(i,j)`-th entry.

        The output of this method, for input `x` representing :math:`x`,
        is a `DataFrame` with same columns and indices as `self`,
        and entries :math:`\log p_{X_{ij}}(x_{ij})`.

        If `self` has a mixed or discrete distribution, this returns
        the weighted continuous part of `self`'s distribution instead of the pdf,
        i.e., the marginal pdf integrate to the weight of the continuous part.

        Parameters
        ----------
        x : `pandas.DataFrame` or 2D np.ndarray
            representing :math:`x`, as above

        Returns
        -------
        `DataFrame` with same columns and index as `self`
            containing :math:`\log p_{X_{ij}}(x_{ij})`, as above
        """
        if isinstance(x, pd.DataFrame):
            dist_at_x = self.loc[x.index, x.columns]
            tensor = dist_at_x.distr.log_prob(x.values)
            return pd.DataFrame(tensor, index=x.index, columns=x.columns)
        else:
            dist_at_x = self
            return dist_at_x.distr.log_prob(x)

    def cdf(self, x):
        """Cumulative distribution function."""
        if isinstance(x, pd.DataFrame):
            dist_at_x = self.loc[x.index, x.columns]
            tensor = dist_at_x.distr.cdf(x.values)
            return pd.DataFrame(tensor, index=x.index, columns=x.columns)
        else:
            dist_at_x = self
            return dist_at_x.distr.cdf(x)

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None

        Returns
        -------
        if `n_samples` is `None`:
        returns a sample that contains a single sample from `self`,
        in `pd.DataFrame` mtype format convention, with `index` and `columns` as `self`
        if n_samples is `int`:
        returns a `pd.DataFrame` that contains `n_samples` i.i.d. samples from `self`,
        in `pd-multiindex` mtype format convention, with same `columns` as `self`,
        and `MultiIndex` that is product of `RangeIndex(n_samples)` and `self.index`
        """
        if n_samples is None:
            np_spl = self.distr.sample().numpy()
            return pd.DataFrame(np_spl, index=self.index, columns=self.columns)
        else:
            np_spl = self.distr.sample(n_samples).numpy()
            np_spl = np_spl.reshape(-1, np_spl.shape[-1])
            mi = _prod_multiindex(range(n_samples), self.index)
            df_spl = pd.DataFrame(np_spl, index=mi, columns=self.columns)
            return df_spl


def _prod_multiindex(ix1, ix2):
    rows = []

    def add_rows(rows, ix):
        if isinstance(ix, pd.MultiIndex):
            ix = ix.to_frame()
            rows += [ix[col] for col in ix.columns]
        else:
            rows += [ix]
        return rows

    rows = add_rows(rows, ix1)
    rows = add_rows(rows, ix2)
    res = pd.MultiIndex.from_product(rows)
    res.names = [None] * len(res.names)
    return res
