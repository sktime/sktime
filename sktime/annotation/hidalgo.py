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

random_z = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0]


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


class hidalgo:
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

    """

    def __init__(
        self,
        metric="euclidean",
        K=1,
        zeta=0.8,
        q=3,
        Niter=10,
        Nreplicas=10,
        burn_in=0.5,
        fixed_Z=0,
        use_Potts=1,
        estimate_zeta=0,
        sampling_rate=2,
        a=None,
        b=None,
        c=None,
        f=None,
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
        indicesIn :
            ? array of neighbours of point i
        indicesOut :
            ? array of points for which i is neighbour
        nbrcount :
            ?
        indicesTrack :
            ?
        """
        N, _ = np.shape(X)
        q = self.q

        nbrs = NearestNeighbors(
            n_neighbors=q + 1, algorithm="ball_tree", metric=self.metric
        ).fit(X)
        distances, indicesIn = nbrs.kneighbors(X)
        mu = np.divide(distances[:, 2], distances[:, 1])

        nbrmat = np.zeros((N, N))
        for n in range(self.q):
            nbrmat[indicesIn[:, 0], indicesIn[:, n + 1]] = 1

        nbrcount = np.sum(nbrmat, axis=0).astype(int)
        indicesOut = np.where(nbrmat.T)[1].astype(int)
        indicesTrack = np.cumsum(nbrcount)
        indicesTrack = np.append(0, indicesTrack[:-1]).astype(int)
        indicesIn = indicesIn[:, 1:]
        indicesIn = np.reshape(indicesIn, (N * self.q,)).astype(int)

        return (mu, indicesIn, indicesOut, nbrcount, indicesTrack)

    def update_zeta_prior(self, Iin, N, Z):
        f1 = np.empty(shape=2)
        N_in = 0
        for i in range(N):
            k = Z[i]

            for j in range(self.q):
                index = Iin[self.q * i + j]
                if Z[index] == k:
                    N_in += 1

        f1[0] = self.f[0] + N_in
        f1[1] = self.f[1] + N * self.q - N_in

        return N_in, f1

    def _initialise_params(self, N, MU, Iin):
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
        d = np.ones(shape=self.K)
        p = np.ones(shape=self.K) / self.K
        a1 = np.empty(shape=self.K)
        b1 = np.empty(shape=self.K)
        c1 = np.empty(shape=self.K)
        Z = np.empty(shape=N, dtype=int)
        pp = (self.K - 1) / self.K

        if bool(self.fixed_Z) is False:
            # z = int(np.floor(random.random()*K))  #FIXME
            Z = np.array(random_z, dtype=int)
        else:
            Z = np.zeros(N, dtype=int)

        for i in range(N):
            V[Z[i]] = V[Z[i]] + np.log(MU[i])
            NN[Z[i]] += 1

        a1 = self.a + NN
        b1 = self.b + V
        c1 = self.c + NN

        N_in, f1 = self.update_zeta_prior(Iin, N, Z)

        return (V, NN, d, p, a1, b1, c1, Z, f1, N_in, pp)

    def gibbs_sampling(
        self,
        N,
        MU,
        Iin,
        Iout,
        Iout_count,
        Iout_track,
        V,
        NN,
        d,
        p,
        a1,
        b1,
        c1,
        Z,
        f1,
        N_in,
        pp,
        r,
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

        def sample_d(K, d, a1, b1):

            for k in range(K):
                stop = False

                while stop is False:

                    # r1 = random.random()*200 # random sample for d[k]
                    # r2 = random.random() # random number for accepting
                    r1 = next_deterministic_number()
                    r2 = next_deterministic_number()

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

                    # r1 = random.random() # random sample for p[k]
                    # r2 = random.random() # random number for accepting

                    r1 = next_deterministic_number()
                    r2 = next_deterministic_number()

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

        def sample_zeta(K, N, zeta, use_Potts, estimate_zeta, q, NN, f1, it):
            stop = False
            maxval = -100000

            if bool(use_Potts) and bool(estimate_zeta):
                for zeta_candidates in range(10):
                    zeta1 = 0.5 + 0.05 * zeta_candidates
                    ZZ = np.empty((K, 0))
                    for k in range(K):
                        ZZ = np.append(ZZ, Zpart(N, NN[k], zeta1, q))
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
                    # r1 = random.random() # random sample for zeta
                    # r2 = random.random() # random number for accepting

                    r1 = next_deterministic_number()
                    r2 = next_deterministic_number()

                    ZZ = np.empty((K, 0))
                    for k in range(K):
                        ZZ = np.append(ZZ, Zpart(N, NN[k], r1, q))
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

        def sampling_Z(Z, NN, a1, c1, V, b1, zeta, N, fixed_Z, q):
            if (abs(zeta - 1) < 1e-5) or fixed_Z:
                return Z, NN, a1, c1, V, b1

            for i in range(N):
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
                            index = int(Iin[q * i + j])
                            if Z[index] == k1:
                                n_in = n_in + 1.0
                        m_in = 0
                        for j in range(int(Iout_count[i])):
                            index = int(Iout[Iout_track[i] + j])
                            if index > -1 and Z[index] == k1:
                                m_in = m_in + 1.0

                        g = (n_in + m_in) * np.log(zeta / (1 - zeta)) - np.log(
                            Zpart(N, NN[k1], zeta, q)
                        )
                        g = g + np.log(
                            Zpart(N, NN[k1] - 1, zeta, q) / Zpart(N, NN[k1], zeta, q)
                        ) * (NN[k1] - 1)

                    if g > gmax:
                        gmax = g
                    gg[k1] = g

                for k1 in range(K):
                    gg[k1] = np.exp(gg[k1] - gmax)

                for k1 in range(K):
                    prob[k1] = p[k1] * d[k1] * MU[i] ** (-(d[k1] + 1)) * gg[k1]
                    norm += prob[k1]

                for k1 in range(K):
                    prob[k1] = prob[k1] / norm

                while stop is False:

                    # r1 = int(np.floor(random.random()*K))
                    # r2 = random.random()

                    r1 = int(next_deterministic_number())
                    r2 = next_deterministic_number()

                    if prob[r1] > r2:
                        stop = True
                        # minus values
                        NN[Z[i]] -= 1
                        a1[Z[i]] -= 1
                        c1[Z[i]] -= 1
                        V[Z[i]] -= np.log(MU[i])
                        b1[Z[i]] -= np.log(MU[i])
                        # change, add values
                        Z[i] = r1
                        NN[Z[i]] += 1
                        a1[Z[i]] += 1
                        c1[Z[i]] += 1
                        V[Z[i]] += np.log(MU[i])
                        b1[Z[i]] += np.log(MU[i])

            return Z, NN, a1, c1, V, b1

        def sample_likelihood(N, p, d, Z, MU, N_in, zeta, NN):
            lik0 = 0
            for i in range(N):
                lik0 = (
                    lik0
                    + np.log(p[Z[i]])
                    + np.log(d[Z[i]])
                    - (d[Z[i]] + 1) * np.log(MU[i])
                )

            lik1 = lik0 + np.log(zeta / (1 - zeta)) * N_in

            for k1 in range(K):
                lik1 = lik1 - (NN[k1] * np.log(Zpart(N, NN[k1], zeta, q)))

            return lik0, lik1

        for it in range(Niter):

            d = sample_d(K, d, a1, b1)
            sampling = np.append(sampling, d)

            (p, pp) = sample_p(K, p, pp, c1)
            sampling = np.append(sampling, p[: K - 1])
            sampling = np.append(sampling, (1 - pp))

            zeta = sample_zeta(K, N, zeta, use_Potts, estimate_zeta, q, NN, f1, it)
            sampling = np.append(sampling, zeta)

            Z, NN, a1, c1, V, b1 = sampling_Z(Z, NN, a1, c1, V, b1, zeta, N, fixed_Z, q)
            sampling = np.append(sampling, Z)

            N_in, f1 = self.update_zeta_prior(Iin, N, Z)

            lik = sample_likelihood(N, p, d, Z, MU, N_in, zeta, NN)
            sampling = np.append(sampling, lik)

        return sampling

    def _fit(self, X):
        """
        Find parameter esimates as distributions in sampling.

        Iterate through Nreplicas random starts and get posterior
        samples with best max likelihood.
        """
        MU, Iin, Iout, Iout_count, Iout_track = self._get_neighbourhood_params(X)

        N = np.shape(X)[0]
        V, NN, d, p, a1, b1, c1, Z, f1, N_in, pp = self._initialise_params(N, MU, Iin)

        Npar = N + 2 * self.K + 2 + 1

        bestsampling = np.zeros(shape=0)

        maxlik = -1e10
        # this can be run in parallel...
        for r in range(self.Nreplicas):
            sampling = self.gibbs_sampling(
                N,
                MU,
                Iin,
                Iout,
                Iout_count,
                Iout_track,
                V,
                NN,
                d,
                p,
                a1,
                b1,
                c1,
                Z,
                f1,
                N_in,
                pp,
                r,
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

        return bestsampling

    def fit(self, X):
        """Run the Hidalgo algorithm and writes results to self.

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
        N = np.shape(X)[0]

        sampling = self._fit(X)
        K = self.K

        self.d_ = np.mean(sampling[:, :K], axis=0)
        self.derr_ = np.std(sampling[:, :K], axis=0)
        self.p_ = np.mean(sampling[:, K : 2 * K], axis=0)
        self.perr_ = np.std(sampling[:, K : 2 * K], axis=0)
        self.lik_ = np.mean(sampling[:, -1], axis=0)
        self.likerr_ = np.std(sampling[:, -1], axis=0)

        Pi = np.zeros((K, N))

        for k in range(K):
            Pi[k, :] = np.sum(sampling[:, (2 * K) + 1 : 2 * K + N + 1] == k, axis=0)

        self.Pi = Pi / np.shape(sampling)[0]
        Z = np.argmax(Pi, axis=0)
        pZ = np.max(Pi, axis=0)
        Z = Z + 1
        Z[np.where(pZ < 0.8)] = 0
        self.Z = Z
