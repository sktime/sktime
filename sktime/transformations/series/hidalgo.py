# -*- coding: utf-8 -*-

"""Hidalgo (Heterogeneous Intrinsic Dimensionality Algorithm) Segmentation."""

__author__ = ["KatieBuc"]
__all__ = ["Hidalgo"]


from functools import reduce
from typing import Union

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_random_state

from sktime.transformations.base import BaseTransformer


class Hidalgo(BaseTransformer):
    """Heteregeneous Intrinsic Dimensionality Algorithm (Hidalgo) model.

    Hidalgo is a robust approach in discriminating regions with
    different local intrinsic dimensionality (topological feature
    measuring complexity). Hidalgo offers unsupervised segmentation
    of high-dimensional data.

    Parameters
    ----------
    metric : str, or callable, optional, default="euclidean"
        directly passed to sklearn KNearestNeighbors,
        must be str or callable that can be passed to KNearestNeighbors
        distance used in the nearest neighbors part of the algorithm
    K : int, optional, default=2
        number of manifolds used in algorithm
    zeta : float, optional, defualt=0.8
        "local homogeneity level" used in the algorithm, see equation (4)
    q : int, optional, default=3
        number of points for local Z interaction, "local homogeneity range"
        see equation (4)
    n_iter : int, optional, default=1000
        number of Gibbs sampling iterations
    n_replicas : int, optional, default=1
        number of random starts to run Gibbs sampling
    burn_in : float, optional, default=0.9
        percentage of Gibbs sampling iterations discarded, "burn-in fraction"
    fixed_Z : bool, optional, default=False
        estimate parameters with fixed z (joint posterior approximation via Gibbs)
        z = (z_1, ..., z_K) is a latent variable introduced, where z_i = k
        indicates point i belongs to manifold K
    use_Potts : bool, optional, default=True
        if using local interaction between z, see equation (4)
    estimate_zeta : bool, optional, default=False
        update zeta in the sampling
    sampling_rate: int, optional, default=10
        rate at which to save samples for each n_iter
    a : np.ArrayLike, optional, default=None
        prior parameters of d, the dimensionality of manifold k
    b : np.ArrayLike, optional, default=None
        prior parameters of d, the dimensionality of manifold k
    c : np.ArrayLike, optional, default=None
        prior parameters of p, the probability that point belongs to manifold k
    f : np.ArrayLike, optional, default=None
        parameters of zeta
    seed : int, optional, default = 1
        generate random numbers with seed

    References
    ----------
    Allegra, Michele, et al. "Data segmentation based on the local
    intrinsic dimension." Scientific reports 10.1 (2020): 1-12.
    https://www.nature.com/articles/s41598-020-72222-0


    Examples
    --------
    >>> from sktime.transformations.series.hidalgo import Hidalgo
    >>> import numpy as np
    >>> np.random.seed(123)
    >>> X = np.random.rand(10,3)
    >>> X[:6, 1:] += 10
    >>> X[6:, 1:] = 0
    >>> model = Hidalgo(K=2, burn_in=0.8, n_iter=100, seed=10)
    >>> fitted_model = model.fit(X)
    >>> Z = fitted_model.transform(X)
    >>> Z.tolist()
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "transform-returns-same-time-index": True,
        "univariate-only": False,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        metric="euclidean",
        K=1,
        zeta=0.8,
        q=3,
        n_iter=1000,
        n_replicas=1,
        burn_in=0.9,
        fixed_Z=False,
        use_Potts=True,
        estimate_zeta=False,
        sampling_rate=10,
        a=None,
        b=None,
        c=None,
        f=None,
        seed=1,
    ):

        self.metric = metric
        self.K = K
        self.zeta = zeta
        self.q = q
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.n_replicas = n_replicas
        self.fixed_Z = fixed_Z
        self.use_Potts = use_Potts
        self.estimate_zeta = estimate_zeta
        self.sampling_rate = sampling_rate
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.seed = seed

        super(Hidalgo, self).__init__()

    def _get_neighbourhood_params(self, X):
        """
        Neighbourhood information from input data X.

        Parameters
        ----------
        X : 2D np.ndarray of shape (N, dim), where dim > 1
            data to fit the algorithm to

        Returns
        -------
        N : int
            number of rows of X
        mu : 1D np.ndarray of length N
            paramerer in Pereto distribtion estimated by r2/r1
        Iin : 1D np.ndarray of length N * q
            encodes the q neighbour index values for point index i in 0:N-1
            e.g. popint i=0 has neighbours 2, 4, 7 and point i=1
            has neighbours 3, 9, 4
            Iin = np.array([2, 4, 7, 3, 9, 4,...])
        Iout : 1D np.ndarray of length N * q
            array of indices for which i is neighbour for i in 0:N-1
            e.g. point i=0 is also neighbour of points 2, 4 and
            point i=1 is a neighbour of point 3 only
            Iout = np.array([2, 4, 3,...])
        Iout_count : 1D np.ndarray of length N
            count of how many neighbours point i has for i in 0:N-1
            e.g. Iout_count = np.array([2, 1, 1,...])
        Iout_track : 1D np.ndarray of length N
            cumulative sum of Iout_count at i-1 for i in 1:N-1
            e.g. Iout_track = np.array([0, 2, 3, 4,...])
        """
        q = self.q
        metric = self.metric

        N, _ = np.shape(X)

        nbrs = NearestNeighbors(
            n_neighbors=q + 1, algorithm="ball_tree", metric=metric
        ).fit(X)
        distances, Iin = nbrs.kneighbors(X)
        mu = np.divide(distances[:, 2], distances[:, 1])

        nbrmat = np.zeros((N, N))
        for n in range(q):
            nbrmat[Iin[:, 0], Iin[:, n + 1]] = 1

        Iout_count = np.sum(nbrmat, axis=0).astype(int)
        Iout = np.where(nbrmat.T)[1].astype(int)
        Iout_track = np.cumsum(Iout_count)
        Iout_track = np.append(0, Iout_track[:-1]).astype(int)
        Iin = Iin[:, 1:]
        Iin = np.reshape(Iin, (N * q,)).astype(int)

        return N, mu, Iin, Iout, Iout_count, Iout_track

    def _update_zeta_prior(self, Z, N, Iin):
        """Update prior parameters for zeta."""
        q = self.q
        f = self.f

        if f is None:
            f = np.ones(2)

        N_in = sum([Z[Iin[q * i + j]] == Z[i] for j in range(q) for i in range(N)])

        f1 = np.empty(shape=2)
        f1[0] = f[0] + N_in
        f1[1] = f[1] + N * q - N_in

        return N_in, f1

    def _initialise_params(self, N, mu, Iin, _rng):
        """
        Initialise parameters used in algorithm.

        Outputs
        ----------
        V : 1D np.ndarray of length K
            sum(log(mu_i)) for k in 0:K-1, when mu_i belongs to manifold k
        NN : 1D np.ndarray of length K
            count for k in 0:K-1, when data at index i belongs to manifold k
        a1 : 1D np.ndarray of length K
            prior parameters of d
        b1 : 1D np.ndarray of length K
            prior parameters of d
        c1 : 1D np.ndarray of length K
            prior parameters of p
        Z : 1D np.ndarray of length N
            segmentation based on manifold k
        f1 : 1D np.ndarray of length 2
            parameters of zeta
        N_in : int
            parameters of zeta
        """
        K = self.K
        a = self.a
        b = self.b
        c = self.c
        fixed_Z = self.fixed_Z

        if a is None:
            a = np.ones(K)
        if b is None:
            b = np.ones(K)
        if c is None:
            c = np.ones(K)

        if not fixed_Z:
            random_z = _rng.randint(0, K, N)
            Z = np.array(random_z, dtype=int)
        else:
            Z = np.zeros(N, dtype=int)

        V = [sum(np.log(mu[[Z[i] == k for i in range(N)]])) for k in range(K)]
        NN = [sum([Z[i] == k for i in range(N)]) for k in range(K)]

        a1 = a + NN
        b1 = b + V
        c1 = c + NN

        N_in, f1 = self._update_zeta_prior(Z, N, Iin)

        return (V, NN, a1, b1, c1, Z, f1, N_in)

    def _gibbs_sampling(
        self,
        N,
        mu,
        Iin,
        Iout,
        Iout_count,
        Iout_track,
        V,
        NN,
        a1,
        b1,
        c1,
        Z,
        f1,
        N_in,
        _rng,
    ):
        """
        Gibbs sampling method to find joint posterior distribution of target variables.

        Notes
        -----
        Target parameters are d, p, Z
        zeta must be computed for the probability distribution of the q-Neighbourhood
        matrix

        Parameters
        ----------
        V : 1D np.ndarray of length K
            sum(log(mu_i)) for k in 0:K-1, when mu_i belongs to manifold k
        NN : 1D np.ndarray of length K
            count for k in 0:K-1, when data at index i belongs to manifold k
        a1 : 1D np.ndarray of length K
            prior parameters of d
        b1 : 1D np.ndarray of length K
            prior parameters of d
        c1 : 1D np.ndarray of length K
            prior parameters of p
        Z : 1D np.ndarray of length N
            segmentation based on manifold k
        f1 : 1D np.ndarray of length 2
            parameters of zeta
        N_in : int
            parameters of zeta

        Returns
        -------
        sampling : 2D np.ndarray of shape (n_iter, Npar), where Npar = N + 2 * K + 2 + 1
            posterior samples of d, p, Z and likelihood samples, respectively.

        """
        zeta = self.zeta
        q = self.q
        K = self.K
        n_iter = self.n_iter
        fixed_Z = self.fixed_Z
        use_Potts = self.use_Potts
        estimate_zeta = self.estimate_zeta

        sampling = np.empty(shape=0)
        pp = (K - 1) / K
        p = np.ones(shape=K) / K

        def sample_d(K, a1, b1, _rng):
            """Sample d, the dimension of manifold k."""
            d = np.empty(shape=K)
            for k in range(K):
                stop = False

                while stop is False:
                    r1 = _rng.random() * 200  # random sample for d[k]
                    r2 = _rng.random()  # random number for accepting

                    rmax = (a1[k] - 1) / b1[k]

                    if a1[k] - 1 > 0:
                        assert rmax > 0
                        frac = np.exp(
                            -b1[k] * (r1 - rmax)
                            - (a1[k] - 1) * (np.log(rmax) - np.log(r1))
                        )
                    else:
                        frac = np.exp(-b1[k] * r1)

                    if frac > r2:
                        stop = True
                        d[k] = r1

            return d

        def sample_p(K, p, pp, c1, _rng):
            """Sample p, the prior probability that a point belongs to manifold k."""
            for k in range(K - 1):
                stop = False

                while stop is False:
                    r1 = _rng.random()  # random sample for p[k]
                    r2 = _rng.random()  # random number for accepting

                    rmax = (c1[k] - 1) / (c1[k] - 1 + c1[K - 1] - 1)
                    frac = ((r1 / rmax) ** (c1[k] - 1)) * (
                        ((1 - r1) / (1 - rmax)) ** (c1[K - 1] - 1)
                    )

                    if frac > r2:
                        stop = True
                        r1 = r1 * (1.0 - pp + p[k])
                        p[K - 1] += p[k] - r1
                        pp -= p[k] - r1
                        p[k] = r1

            return (p, pp)

        def sample_zeta(N, K, zeta, use_Potts, estimate_zeta, q, NN, f1, it, _rng):
            """Sample zeta, sample probability neighbour of point from same manifold."""
            stop = False
            maxval = -100000

            if use_Potts and estimate_zeta:
                for zeta_candidates in range(10):
                    zeta1 = 0.5 + 0.05 * zeta_candidates
                    ZZ = [partition_function(N, NN[k], zeta1, q) for k in range(K)]
                    h = [NN[k] * np.log(ZZ[k]) for k in range(K)]
                    val = (
                        (f1[0] - 1) * np.log(zeta1)
                        + (f1[1] - 1) * np.log(1 - zeta1)
                        - h
                    )

                    if val > maxval:
                        maxval = val

                while stop is False:
                    r1 = _rng.random()  # random sample for zeta
                    r2 = _rng.random()  # random number for accepting

                    ZZ = [partition_function(N, NN[k], r1, q) for k in range(K)]
                    h = [NN[k] * np.log(ZZ[k]) for k in range(K)]
                    val = (f1[0] - 1) * np.log(r1) + (f1[1] - 1) * np.log(1 - r1) - h
                    frac = np.exp(val - maxval)

                    if frac > r2:
                        stop = True
                        if it > 0:
                            zeta = r1

            return zeta

        def sample_Z(
            N,
            mu,
            Iin,
            Iout,
            Iout_count,
            Iout_track,
            Z,
            NN,
            a1,
            c1,
            V,
            b1,
            zeta,
            fixed_Z,
            q,
            _rng,
        ):
            """Sample z, latent variable indicating point i belongs to manifold k."""
            if (abs(zeta - 1) < 1e-5) or fixed_Z:
                return Z, NN, a1, c1, V, b1

            for i in range(N):
                stop = False
                gg = np.empty(shape=K)
                gmax = 0

                for k1 in range(K):
                    g = 0
                    if use_Potts:

                        n_in = sum([Z[Iin[q * i + j]] == k1 for j in range(q)])

                        m_in = sum(
                            [
                                Iout[Iout_track[i] + j] > -1
                                and Z[Iout[Iout_track[i] + j]] == k1
                                for j in range(Iout_count[i])
                            ]
                        )

                        g = (n_in + m_in) * np.log(zeta / (1 - zeta)) - np.log(
                            partition_function(N, NN[k1], zeta, q)
                        )
                        var = partition_function(
                            N, NN[k1] - 1, zeta, q
                        ) / partition_function(N, NN[k1], zeta, q)
                        assert var > 0
                        g = g + np.log(var) * (NN[k1] - 1)

                    if g > gmax:
                        gmax = g
                    gg[k1] = g

                gg = [np.exp(gg[k1] - gmax) for k1 in range(K)]

                prob = p * d * mu[i] ** (-(d + 1)) * gg
                prob /= prob.sum()

                while stop is False:
                    r1 = int(np.floor(_rng.random() * K))  # random sample for Z
                    r2 = _rng.random()  # random number for accepting

                    if prob[r1] > r2:
                        stop = True
                        # minus values
                        NN[Z[i]] -= 1
                        a1[Z[i]] -= 1
                        c1[Z[i]] -= 1
                        V[Z[i]] -= np.log(mu[i])
                        b1[Z[i]] -= np.log(mu[i])
                        # change Z, add values
                        Z[i] = r1
                        NN[Z[i]] += 1
                        a1[Z[i]] += 1
                        c1[Z[i]] += 1
                        V[Z[i]] += np.log(mu[i])
                        b1[Z[i]] += np.log(mu[i])

            return Z, NN, a1, c1, V, b1

        def sample_likelihood(N, mu, p, d, Z, N_in, zeta, NN):
            """Sample likelihood values of mu, and local inhomogeneity penalisation."""
            lik0 = 0
            for i in range(N):
                lik0 = (
                    lik0
                    + np.log(p[Z[i]])
                    + np.log(d[Z[i]])
                    - (d[Z[i]] + 1) * np.log(mu[i])
                )

            lik1 = lik0 + np.log(zeta / (1 - zeta)) * N_in

            for k1 in range(K):
                lik1 = lik1 - (NN[k1] * np.log(partition_function(N, NN[k1], zeta, q)))

            return lik0, lik1

        for it in range(n_iter):

            d = sample_d(K, a1, b1, _rng)
            sampling = np.append(sampling, d)

            (p, pp) = sample_p(K, p, pp, c1, _rng)
            sampling = np.append(sampling, p[: K - 1])
            sampling = np.append(sampling, (1 - pp))

            zeta = sample_zeta(
                N, K, zeta, use_Potts, estimate_zeta, q, NN, f1, it, _rng
            )
            sampling = np.append(sampling, zeta)

            Z, NN, a1, c1, V, b1 = sample_Z(
                N,
                mu,
                Iin,
                Iout,
                Iout_count,
                Iout_track,
                Z,
                NN,
                a1,
                c1,
                V,
                b1,
                zeta,
                fixed_Z,
                q,
                _rng,
            )
            sampling = np.append(sampling, Z)

            N_in, f1 = self._update_zeta_prior(Z, N, Iin)

            lik = sample_likelihood(N, mu, p, d, Z, N_in, zeta, NN)
            sampling = np.append(sampling, lik)

        return sampling

    def _fit(self, X, y=None):
        """Run the Hidalgo algorithm.

        Find parameter esimates as distributions in sampling.
        Iterate through n_replicas random starts and get posterior
        samples with best max likelihood.

        Notes
        -----
        Writes to self
        _d : 1D np.ndarray of length K
            posterior mean of d, from posterior sample in gibbs_sampling
        _derr : 1D np.ndarray of length K
            posterior std of d, from posterior sample in gibbs_sampling
        _p : 1D np.ndarray of length K
            posterior mean of p, from posterior sample in gibbs_sampling
        _perr : 1D np.ndarray of length K
            posterior std of p, from posterior sample in gibbs_sampling
        _lik : float
            mean of likelihood, from sample in gibbs_sampling
        _likerr : float
            std of likelihood, from sample in gibbs_sampling
        _Pi : 2D np.ndarray of shape (K, N)
            probability of posterior of z_i = k, point i can be safely
            assigned to manifold k if Pi > 0.8
        _Z : 1D np.ndarray of length N
            base-zero integer values corresponsing to segment (manifold k)

        Parameters
        ----------
        X : 2D np.ndarray of shape (N, dim), where dim > 1
            data to fit the algorithm to

        Returns
        -------
        self
        """
        K = self.K
        n_replicas = self.n_replicas
        n_iter = self.n_iter
        sampling_rate = self.sampling_rate
        burn_in = self.burn_in
        seed = self.seed

        _rng = check_random_state(seed)

        N, mu, Iin, Iout, Iout_count, Iout_track = self._get_neighbourhood_params(X)
        V, NN, a1, b1, c1, Z, f1, N_in = self._initialise_params(N, mu, Iin, _rng)

        Npar = N + 2 * K + 2 + 1
        bestsampling = np.zeros(shape=0)
        maxlik = -1e10

        for _ in range(n_replicas):

            sampling = self._gibbs_sampling(
                N,
                mu,
                Iin,
                Iout,
                Iout_count,
                Iout_track,
                V,
                NN,
                a1,
                b1,
                c1,
                Z,
                f1,
                N_in,
                _rng,
            )
            sampling = np.reshape(sampling, (n_iter, Npar))

            idx = [
                it
                for it in range(n_iter)
                if it % sampling_rate == 0 and it >= n_iter * burn_in
            ]
            sampling = sampling[
                idx,
            ]

            lik = np.mean(sampling[:, -1], axis=0)

            if lik > maxlik:
                bestsampling = sampling
                maxlik = lik

        self._d = np.mean(bestsampling[:, :K], axis=0)
        self._derr = np.std(bestsampling[:, :K], axis=0)
        self._p = np.mean(bestsampling[:, K : 2 * K], axis=0)
        self._perr = np.std(bestsampling[:, K : 2 * K], axis=0)
        self._lik = np.mean(bestsampling[:, -1], axis=0)
        self._likerr = np.std(bestsampling[:, -1], axis=0)

        Pi = np.zeros((K, N))

        for k in range(K):
            Pi[k, :] = np.sum(bestsampling[:, (2 * K) + 1 : 2 * K + N + 1] == k, axis=0)

        Pi = Pi / len(idx)
        self._Pi = Pi

        Z = np.argmax(Pi, axis=0)
        pZ = np.max(Pi, axis=0)
        Z[np.where(pZ < 0.8)] = -1
        self._Z = Z

        return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series of mtype y_inner_mtype, default=None
            Not used in this unsupervised implementation

        Returns
        -------
        transformed version of X
        """
        return self._Z

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            Reserved values for classifiers:
                "results_comparison" - used for identity testing in some classifiers
                    should contain parameter settings comparable to "TSC bakeoff"

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {
            "metric": "euclidean",
            "K": 1,
            "zeta": 0.8,
            "q": 3,
            "n_iter": 10,
            "n_replicas": 1,
            "burn_in": 0.5,
            "fixed_Z": False,
            "use_Potts": True,
            "estimate_zeta": False,
            "sampling_rate": 2,
            "a": None,
            "b": None,
            "c": None,
            "f": None,
            "seed": 1,
        }


def binom(N: Union[int, float], q: Union[int, float]):
    """Calculate the binomial coefficient.

    Parameters
    ----------
    N : int, float
        number of fixed elements from qhich q is chosen
    q : int, float
        number of subset q elements chosen from N
    """
    if q == 0:
        return 1.0
    if N < 0:
        return 0.0
    return reduce(lambda x, y: x * y, [(N - q1) / (q1 + 1) for q1 in range(q)])


def partition_function(N, N1, zeta, q):
    """Partition function for Z.

    Parameters
    ----------
    N : int, float
        number of rows of input data X
    N1 : int, float
        parameter value from NN[k] for k=0:K-1
    zeta : float
        parameter value zeta
    q : int, float
        parameter value q
    """
    return sum(
        [
            binom(N1 - 1, q1)
            * binom(N - N1, q - q1)
            * zeta ** (q1)
            * (1 - zeta) ** (q - q1)
            for q1 in range(q + 1)
        ]
    )
