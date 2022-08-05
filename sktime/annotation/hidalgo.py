# -*- coding: utf-8 -*-

"""
HidAlgo (Heterogeneous Intrinsic Dimensionality Algorithm) Segmentation.

Notes
-----
As described in
@article{allegra2020data,
  title={Data segmentation based on the local intrinsic dimension},
  author={Allegra, Michele and Facco, Elena and Denti, Francesco and Laio,
        Alessandro and Mira, Antonietta},
  journal={Scientific reports},
  volume={10},
  number={1},
  pages={1--12},
  year={2020},
  publisher={Nature Publishing Group}
}
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_random_state


def get_deterministic_z():
    return [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]


def get_deterministic_number():
    global random_list
    file_path = Path(__file__).parent.joinpath("tests", "random_numbers.csv")
    random_list = pd.read_csv(file_path, header=None).values.tolist()[0]


def next_deterministic_number():
    global random_list
    return random_list.pop(0)


def binom(N, q):
    """
    Replicate function in c++ implementation.

    print(binom(-1,0)) -> 1
    print(binom(-1,1)) -> -1
    print(binom(-1,2)) -> 1
    print(binom(-1,3)) -> -1
    whereas in scipy.special, returns nan
    """
    ss = 1.0
    if q == 0:
        return 1.0
    for q1 in range(q):
        ss = ss * (N - q1) / (q1 + 1)
    return ss


def Zpart(N, N1, zeta, q):
    """Partition function for Z."""
    s = 0
    for q1 in range(q + 1):
        s += (
            binom(N1 - 1, q1)
            * binom(N - N1, q - q1)
            * zeta ** (q1)
            * (1 - zeta) ** (q - q1)
        )

    return s


class Hidalgo:
    """Class to fit parameters of the HidAlgo intrinsic dimension model.

    explain, reference

    Parameters
    ----------
    metric : str, or callable, optional, default="euclidean"
        directly passed to sklearn KNearestNeighbors,
        must be str or callable that can be passed to KNearestNeighbors
        distance used in the nearest neighbors part of the algorithm
    K : int, optional, default=2
        number of manifolds used in algorithm
    zeta : float, optional, defualt=0.8
        "local homogeneity level" used in the algorithm, see equation ?
    q : int, optional, default=3
        number of points for local Z interaction, "local homogeneity range"
        see equation ?
    Niter : int, optional, default=10000
        number of Gibbs sampling iterations
    Nreplicas : int, optional, default=1
        number of random starts to run Gibbs sampling
    burn_in : float, optional, default=0.9
        percentage of Gibbs sampling iterations discarded, "burn-in fraction"
    fixed_Z : bool, optional, default=0 ** False
        estimate parameters with fixed z (joint posterior approximation via Gibbs)
        z = (z_1, ..., z_K) is a latent variable introduced, where z_i = k
        indicates point i belongs to manifold K
    use_Potts : bool, optional, default=1 ** True
        if using local interaction between z, see equation (4)
    estimate_zeta : bool, optional, default=0 ** False
        update zeta in the sampling
    sampling_rate: int, optional, default=10 ** currently 2
        rate at which to save samples for each Niter
    a : np.ArrayLike, optional, default=1.0
        prior parameters of d
    b : np.ArrayLike, optional, default=1.0
        prior parameters of d
    c : np.ArrayLike, optional, default=1.0
        prior parameters of p
    f : np.ArrayLike, optional, default=1.0
        parameters of zeta
    seed : int, None, optional, default = 1
        if None read in pre-generated random numbers from file
        otherwise, generate random numbers with seed
    """

    def __init__(
        self,
        metric="euclidean",
        K=1,
        zeta=0.8,
        q=3,
        Niter=10000,
        Nreplicas=1,
        burn_in=0.5,
        fixed_Z=0,
        use_Potts=1,
        estimate_zeta=0,
        sampling_rate=2,
        a=None,
        b=None,
        c=None,
        f=None,
        seed=1,
    ):

        if a is None:
            a = np.ones(K)
        if b is None:
            b = np.ones(K)
        if c is None:
            c = np.ones(K)
        if f is None:
            f = np.ones(K)

        self.metric = metric
        self.K = K
        self.zeta = zeta
        self.q = q
        self.Niter = Niter
        self.burn_in = burn_in
        self.Nreplicas = Nreplicas
        self.fixed_Z = fixed_Z
        self.use_Potts = use_Potts
        self.estimate_zeta = estimate_zeta
        self.sampling_rate = sampling_rate
        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.seed = seed

    def _get_neighbourhood_params(self, X):
        """
        Neighbourhood information from input data X.

        Parameters
        ----------
        X :
            input data

        Outputs
        ----------
        mu :
            paramerer in Pereto distribtion estimated by r2/r1
        Iin :
            ? array of neighbours of point i
        Iout :
            ? array of points for which i is neighbour
        Iout_count :
            ?
        Iout_track :
            ?
        """
        N, _ = np.shape(X)
        self.N = N

        nbrs = NearestNeighbors(
            n_neighbors=self.q + 1, algorithm="ball_tree", metric=self.metric
        ).fit(X)
        distances, Iin = nbrs.kneighbors(X)
        mu = np.divide(distances[:, 2], distances[:, 1])

        nbrmat = np.zeros((N, N))
        for n in range(self.q):
            nbrmat[Iin[:, 0], Iin[:, n + 1]] = 1

        Iout_count = np.sum(nbrmat, axis=0).astype(int)
        Iout = np.where(nbrmat.T)[1].astype(int)
        Iout_track = np.cumsum(Iout_count)
        Iout_track = np.append(0, Iout_track[:-1]).astype(int)
        Iin = Iin[:, 1:]
        Iin = np.reshape(Iin, (N * self.q,)).astype(int)

        self.mu = mu
        self.Iin = Iin
        self.Iout = Iout
        self.Iout_count = Iout_count
        self.Iout_track = Iout_track

    def update_zeta_prior(self, Z):
        f1 = np.empty(shape=2)
        N_in = 0
        for i in range(self.N):
            k = Z[i]

            for j in range(self.q):
                index = self.Iin[self.q * i + j]
                if Z[index] == k:
                    N_in += 1

        f1[0] = self.f[0] + N_in
        f1[1] = self.f[1] + self.N * self.q - N_in

        return N_in, f1

    def get_random_z(self):
        return self._rng.randint(0, self.K, self.N)

    def _initialise_params(self):
        """
        Decription.

        Outputs
        ----------
        V :
        NN :
        d :
        p :
        a1 :
        b1 :
        c1 :
        Z :
        f1 :
        N_in :
        pp :
        """
        # params to initialise
        V = np.zeros(shape=self.K)
        NN = np.zeros(shape=self.K)
        a1 = np.empty(shape=self.K)
        b1 = np.empty(shape=self.K)
        c1 = np.empty(shape=self.K)
        Z = np.empty(shape=self.N, dtype=int)

        if bool(self.fixed_Z) is False:

            if self.seed is None:
                random_z = get_deterministic_z()
            else:
                random_z = self.get_random_z()

            Z = np.array(random_z, dtype=int)
        else:
            Z = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            V[Z[i]] = V[Z[i]] + np.log(self.mu[i])
            NN[Z[i]] += 1

        a1 = self.a + NN
        b1 = self.b + V
        c1 = self.c + NN

        N_in, f1 = self.update_zeta_prior(Z)

        return (V, NN, a1, b1, c1, Z, f1, N_in)

    def gibbs_sampling(
        self,
        V,
        NN,
        a1,
        b1,
        c1,
        Z,
        f1,
        N_in,
    ):
        """
        Gibbs sampling method to find joint posterior distribution of target variables.

        Notes
        -----
        Target parameters are d, p, Z
        zeta must be computed for the probability distribution of the q-Neighbourhood
        matrix
        """
        get_deterministic_number()

        zeta = self.zeta
        q = self.q
        K = self.K
        Niter = self.Niter
        fixed_Z = self.fixed_Z
        use_Potts = self.use_Potts
        estimate_zeta = self.estimate_zeta

        sampling = np.empty(shape=0)
        pp = (K - 1) / K
        p = np.ones(shape=K) / K

        def sample_d(K, a1, b1):

            d = np.empty(shape=K)
            for k in range(K):
                stop = False

                while stop is False:
                    if self.seed is None:
                        r1 = next_deterministic_number()
                        r2 = next_deterministic_number()
                    else:
                        r1 = self._rng.random() * 200  # random sample for d[k]
                        r2 = self._rng.random()  # random number for accepting

                    rmax = (a1[k] - 1) / b1[k]

                    if a1[k] - 1 > 0:
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

        def sample_p(K, p, pp, c1):

            for k in range(K - 1):
                stop = False

                while stop is False:
                    if self.seed is None:
                        r1 = next_deterministic_number()
                        r2 = next_deterministic_number()
                    else:
                        r1 = self._rng.random()  # random sample for p[k]
                        r2 = self._rng.random()  # random number for accepting

                    rmax = (c1[k] - 1) / (c1[k] - 1 + c1[K - 1] - 1)
                    frac = ((r1 / rmax) ** (c1[k] - 1)) * (
                        ((1 - r1) / (1 - rmax)) ** (c1[K - 1] - 1)
                    )

                    if frac > r2:
                        stop = True
                        r1 = r1 * (1.0 - pp + p[k])
                        p[K - 1] += p[k] - r1  # why???
                        pp -= p[k] - r1
                        p[k] = r1

            return (p, pp)

        def sample_zeta(K, zeta, use_Potts, estimate_zeta, q, NN, f1, it):
            stop = False
            maxval = -100000

            if bool(use_Potts) and bool(estimate_zeta):
                for zeta_candidates in range(10):
                    zeta1 = 0.5 + 0.05 * zeta_candidates
                    ZZ = np.empty((K, 0))
                    for k in range(K):
                        ZZ = np.append(ZZ, Zpart(self.N, NN[k], zeta1, q))
                    h = 0
                    for k in range(K):
                        h = h + NN[k] * np.log(ZZ[k])

                    val = (
                        (f1[0] - 1) * np.log(zeta1)
                        + (f1[1] - 1) * np.log(1 - zeta1)
                        - h
                    )

                    if val > maxval:
                        maxval = val  # found max val for below frac

                while stop is False:
                    if self.seed is None:
                        r1 = next_deterministic_number()
                        r2 = next_deterministic_number()
                    else:
                        r1 = self._rng.random()  # random sample for zeta
                        r2 = self._rng.random()  # random number for accepting

                    ZZ = np.empty((K, 0))
                    for k in range(K):
                        ZZ = np.append(ZZ, Zpart(self.N, NN[k], r1, q))
                    h = 0
                    for k in range(K):
                        h = h + NN[k] * np.log(ZZ[k])

                    val = (f1[0] - 1) * np.log(r1) + (f1[1] - 1) * np.log(1 - r1) - h
                    frac = np.exp(val - maxval)

                    if frac > r2:
                        stop = True
                        if it > 0:
                            zeta = r1

            return zeta

        def sampling_Z(Z, NN, a1, c1, V, b1, zeta, fixed_Z, q):
            if (abs(zeta - 1) < 1e-5) or fixed_Z:
                return Z, NN, a1, c1, V, b1

            for i in range(self.N):
                stop = False
                prob = np.empty(shape=K)
                gg = np.empty(shape=K)
                norm = 0
                gmax = 0

                for k1 in range(K):
                    g = 0
                    if use_Potts:
                        n_in = 0
                        for j in range(q):
                            index = int(self.Iin[q * i + j])
                            if Z[index] == k1:
                                n_in = n_in + 1.0
                        m_in = 0
                        for j in range(int(self.Iout_count[i])):
                            index = int(self.Iout[self.Iout_track[i] + j])
                            if index > -1 and Z[index] == k1:
                                m_in = m_in + 1.0

                        g = (n_in + m_in) * np.log(zeta / (1 - zeta)) - np.log(
                            Zpart(self.N, NN[k1], zeta, q)
                        )
                        g = g + np.log(
                            Zpart(self.N, NN[k1] - 1, zeta, q)
                            / Zpart(self.N, NN[k1], zeta, q)
                        ) * (NN[k1] - 1)

                    if g > gmax:
                        gmax = g
                    gg[k1] = g

                for k1 in range(K):
                    gg[k1] = np.exp(gg[k1] - gmax)

                for k1 in range(K):
                    prob[k1] = p[k1] * d[k1] * self.mu[i] ** (-(d[k1] + 1)) * gg[k1]
                    norm += prob[k1]

                for k1 in range(K):
                    prob[k1] = prob[k1] / norm

                while stop is False:
                    if self.seed is None:
                        r1 = int(next_deterministic_number())
                        r2 = next_deterministic_number()
                    else:
                        r1 = int(
                            np.floor(self._rng.random() * K)
                        )  # random sample for Z
                        r2 = self._rng.random()  # random number for accepting

                    if prob[r1] > r2:
                        stop = True
                        # minus values
                        NN[Z[i]] -= 1
                        a1[Z[i]] -= 1
                        c1[Z[i]] -= 1
                        V[Z[i]] -= np.log(self.mu[i])
                        b1[Z[i]] -= np.log(self.mu[i])
                        # change, add values
                        Z[i] = r1
                        NN[Z[i]] += 1
                        a1[Z[i]] += 1
                        c1[Z[i]] += 1
                        V[Z[i]] += np.log(self.mu[i])
                        b1[Z[i]] += np.log(self.mu[i])

            return Z, NN, a1, c1, V, b1

        def sample_likelihood(p, d, Z, N_in, zeta, NN):
            lik0 = 0
            for i in range(self.N):
                lik0 = (
                    lik0
                    + np.log(p[Z[i]])
                    + np.log(d[Z[i]])
                    - (d[Z[i]] + 1) * np.log(self.mu[i])
                )

            lik1 = lik0 + np.log(zeta / (1 - zeta)) * N_in

            for k1 in range(K):
                lik1 = lik1 - (NN[k1] * np.log(Zpart(self.N, NN[k1], zeta, q)))

            return lik0, lik1

        for it in range(Niter):

            d = sample_d(K, a1, b1)
            sampling = np.append(sampling, d)

            (p, pp) = sample_p(K, p, pp, c1)
            sampling = np.append(sampling, p[: K - 1])
            sampling = np.append(sampling, (1 - pp))

            zeta = sample_zeta(K, zeta, use_Potts, estimate_zeta, q, NN, f1, it)
            sampling = np.append(sampling, zeta)

            Z, NN, a1, c1, V, b1 = sampling_Z(Z, NN, a1, c1, V, b1, zeta, fixed_Z, q)
            sampling = np.append(sampling, Z)

            N_in, f1 = self.update_zeta_prior(Z)

            lik = sample_likelihood(p, d, Z, N_in, zeta, NN)
            sampling = np.append(sampling, lik)

        return sampling

    def _fit(self, X, y=None):
        self._rng = check_random_state(self.seed)
        return self

    def predict(self, X):
        """
        Run the Hidalgo algorithm and writes results to self.

        Find parameter esimates as distributions in sampling.
        Iterate through Nreplicas random starts and get posterior
        samples with best max likelihood.

        Write to self:
        self.d_ : 1D np.ndarray of length K
            posterior mean of d, from posterior sample in _fit
        self.derr_ : 1D np.ndarray of length K
            posterior std of d, from posterior sample in _fit
        self.p_ : 1D np.ndarray of length K
            posterior mean of p, from posterior sample in _fit
        self.perr_ : 1D np.ndarray of length K
            posterior std of p, from posterior sample in _fit
        self.lik_ : float
            mean of likelihood, from sample in _fit
        self.likerr_ : float
            std of likelihood, from sample in _fit
        Pi : 2D np.ndarray of shape (K, N)
            probability of posterior of z_i = k, point i can be safely
            assigned to manifold k if Pi > 0.8
        Z : 1D np.ndarray of length N
            ****

        Parameters
        ----------
        X : 2D np.ndarray of shape (N, dim)
            data to fit the algorithm to

        Returns
        -------
        None
        """

        self._get_neighbourhood_params(X)
        V, NN, a1, b1, c1, Z, f1, N_in = self._initialise_params()

        Npar = self.N + 2 * self.K + 2 + 1

        bestsampling = np.zeros(shape=0)

        maxlik = -1e10

        # this can be run in parallel...FIXME: ISSUE
        for _ in range(self.Nreplicas):

            sampling = self.gibbs_sampling(
                V,
                NN,
                a1,
                b1,
                c1,
                Z,
                f1,
                N_in,
            )
            sampling = np.reshape(sampling, (self.Niter, Npar))

            idx = [
                it
                for it in range(self.Niter)
                if it % self.sampling_rate == 0 and it >= self.Niter * self.burn_in
            ]
            sampling = sampling[
                idx,
            ]

            lik = np.mean(sampling[:, -1], axis=0)

            if lik > maxlik:
                bestsampling = sampling
                maxlik = lik

        K = self.K
        self.sampling = bestsampling
        self.d_ = np.mean(bestsampling[:, :K], axis=0)
        self.derr_ = np.std(bestsampling[:, :K], axis=0)
        self.p_ = np.mean(bestsampling[:, K : 2 * K], axis=0)
        self.perr_ = np.std(bestsampling[:, K : 2 * K], axis=0)
        self.lik_ = np.mean(bestsampling[:, -1], axis=0)
        self.likerr_ = np.std(bestsampling[:, -1], axis=0)

        Pi = np.zeros((K, self.N))

        for k in range(K):
            Pi[k, :] = np.sum(
                bestsampling[:, (2 * K) + 1 : 2 * K + self.N + 1] == k, axis=0
            )

        # self.sampling_z = bestsampling[:, (2 * K) + 1 : 2 * K + self.N + 1]

        self.Pi = Pi / np.shape(bestsampling)[0]
        Z = np.argmax(Pi, axis=0)
        pZ = np.max(Pi, axis=0)
        Z = Z + 1  # we can make base-zero
        Z[np.where(pZ < 0.8)] = 0

        return Z
