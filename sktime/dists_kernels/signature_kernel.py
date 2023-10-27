"""Signature kernels from Kiraly et al, 2016."""

__author__ = ["fkiraly"]

import collections
from functools import partial

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.base import BaseEstimator, TransformerMixin

from sktime.dists_kernels.base import BasePairwiseTransformerPanel

# cumsum varia
# ------------


def _coerce_to_list_or_tuple(i):
    """Coerce integers to list of integers."""
    if not isinstance(i, (list, tuple)):
        return [i]
    else:
        return i


def cumsum_rev_first(array):
    """Reverse cumsum over 0th axis."""
    out = np.zeros_like(array)
    out[:-1] = np.cumsum(array[:0:-1], axis=0)[::-1]
    return out


def cumsum_rev(array):
    """Reverse cumsum over both axes 0 and 1."""
    out = np.zeros_like(array)
    out[:-1, :-1] = np.cumsum(np.cumsum(array[:0:-1, :0:-1], 0), 1)[::-1, ::-1]
    return out


def cumsum_mult(array, dims):
    """Cumsum over all axes in dims."""
    dims = _coerce_to_list_or_tuple(dims)
    for dimind in dims:
        array = np.cumsum(array, axis=dimind)
    return array


def roll_mult(array, shift, dims):
    """Roll over all axes in dims."""
    dims = _coerce_to_list_or_tuple(dims)
    for dimind in dims:
        array = np.roll(array, shift, axis=dimind)
    return array


def makeinds(indlist):
    """Make mesh from sequences."""
    return np.ix_(*indlist)


def cumsum_shift_mult(array, dims):
    """Apply cumsum and shift to all axes in dims."""
    array = cumsum_mult(array, dims)
    array = roll_mult(array, 1, dims)

    arrayshape = array.shape
    indarr = []
    for ind in range(len(arrayshape)):
        indarr = indarr + [range(arrayshape[ind])]

    for dimind in dims:
        slicearr = indarr[:]
        slicearr[dimind] = [0]
        array[makeinds(slicearr)] = 0

    return array


# low rank reduction utilities
# ----------------------------


def rankreduce(array, rankbound):
    """Project 2D array on top rankbound singular values."""
    arraysvd = svds(array.astype("f"), k=rankbound)
    return np.dot(arraysvd[0], np.diag(arraysvd[1]))


def rankreduce_batch(arrays, rankbound):
    """Apply rankreduce to axis 0 stack of arrays."""
    resultarrays = np.zeros([arrays.shape[0], arrays.shape[1], rankbound])
    for i in range(arrays.shape[0]):
        resultarrays[i, :, :] = rankreduce(arrays[i, :, :], rankbound)
    return resultarrays


# kernels and distances
# ---------------------


def sqdist(X, Y):
    """Row-wise squared distance between 2D array X and 2D array Y."""
    M = np.shape(X)[0]
    N = np.shape(Y)[0]
    X_squared = np.tile((X * X).sum(-1), [N, 1]).T
    Y_squared = np.tile((Y * Y).sum(-1), [M, 1])
    X_times_Y = np.inner(X, Y)
    return X_squared + Y_squared - 2 * X_times_Y


def k_polynom(x, y, scale, deg):
    """Polynomial kernel of degree deg, with scale coeff."""
    return (1 + scale * np.inner(x, y)) ** deg


def k_gauss(x, y, scale):
    """Gaussian kernel with scale coeff."""
    return np.exp(-(scale**2) * sqdist(x, y) / 2)


def k_euclid(x, y, scale):
    """Euclidean kernel with scale coeff."""
    return scale * np.inner(x, y)


def k_laplace(x, y, scale):
    """Laplace kernel with scale coeff."""
    return np.exp(-scale * np.sqrt(np.inner(x - y, x - y)))


def k_tanh(x, y, off, scale):
    """Tanh kernel with scale and offset."""
    return np.tanh(off + scale * np.inner(x, y))


def mirror(K):
    """Mirrors an upper triangular kernel matrix, helper for Sqize_kernel."""
    return K - np.diag(np.diag(K)) + np.transpose(K)


# sequential kernel implementation
# --------------------------------


def sqize_kernel(K, L, theta=1.0, normalize=False):
    """Compute the sequential kernel from a pairwise kernel matrix.

    aka compute "sequentialization of the kernel K"

    Parameters
    ----------
    K : 2D np.ndarray
        the kernel matrix of increments, i.e.,
        K[i,j] is the kernel between the i-th increment of path 1 (rows),
        and the j-th increment of path 2 (columns)
    L : an integer >= 1, representing the level of truncation
    theta : float, optional, default=1.0
        a positive scaling factor for the levels, i-th level is scaled by theta^i
    normalize : bool, optional, default = False
        whether the output kernel matrix is normalized
        if True, sums and cumsums are divided by prod(K.shape)

    Returns
    -------
    a real number, the sequential kernel between path 1 (rows of K) and path 2 (cols)
    """
    # L-1 runs through loop;
    # returns R_ij=(1+\sum_i2>i,j2>j A_i2,j2(1+\sum A_iLjL)...)
    if normalize:
        normfac = np.prod(K.shape)
        Id = np.ones(K.shape)
        R = np.ones(K.shape)
        for _ in range(L - 1):
            R = (Id + theta * cumsum_rev(K * R) / normfac) / (1 + theta)
        return (1 + theta * np.sum(K * R) / normfac) / (1 + theta)
    else:
        Id = np.ones(K.shape)
        R = np.ones(K.shape)
        for _ in range(L - 1):
            R = Id + cumsum_rev(K * R)  # A*R is componentwise
        # outermost bracket: since i1>=1 and not i1>1 we do it outside of loop
        return 1 + np.sum(K * R)


def sqize_kernel_ho(K, L, D=1, theta=1.0, normalize=False):
    """Compute the higher-order sequential kernel from a pairwise kernel matrix.

    aka compute "sequentialization of the kernel K", higher-order approximation

    Parameters
    ----------
    K : 2D np.ndarray
        the kernel matrix of increments, i.e.,
        K[i,j] is the kernel between the i-th increment of path 1 (rows),
        and the j-th increment of path 2 (columns)
    L : an integer >= 1, representing the level of truncation
    D : int, optional, default = 1
        an integer >= 1, representing the order of approximation
    theta : float, optional, default=1.0
        a positive scaling factor for the levels, i-th level is scaled by theta^i
    normalize : bool, optional, default = False
        whether the output kernel matrix is normalized
        if True, sums and cumsums are divided by prod(K.shape)

    Returns
    -------
    a real number, the sequential kernel between path 1 (rows of K) and path 2 (cols)
    """
    A = np.zeros(np.concatenate(([L, D, D], K.shape)))
    Id = np.ones(K.shape)

    for ell in range(1, L):
        Dprime = min(D, ell)
        Acs = cumsum_shift_mult(np.sum(A[ell - 1, :, :, :, :], (0, 1)), (0, 1))
        A[ell, 0, 0, :, :] = K * (Id + Acs)
        for d1 in range(1, Dprime):
            Acs1 = cumsum_shift_mult(np.sum(A[ell - 1, d1 - 1, :, :, :], 0), 1)
            Acs2 = cumsum_shift_mult(np.sum(A[ell - 1, :, d1 - 1, :, :], 0), 0)
            A[ell, d1, 0, :, :] = A[ell, d1, 0, :, :] + (1 / d1) * K * Acs1
            A[ell, :, d1, :, :] = A[ell, 0, d1, :, :] + (1 / d1) * K * Acs2

            for d2 in range(1, Dprime):
                Acs12 = cumsum_shift_mult(
                    np.sum(A[ell - 1, d1 - 1, d2 - 1, :, :], 0), 0
                )
                A[ell, d1, d2, :, :] = (
                    A[ell, d1, d2, :, :] + (1 / (d1 * d2)) * K * Acs12
                )
    return 1 + np.sum(A[L - 1, :, :, :, :])


# low-rank decomposition
# ----------------------

# LRdec object
#  models matrix A = U x V.T
#  U and V should be *arrays*, not *matrices*
LRdec = collections.namedtuple("LRdec", ["U", "V"])


def get_low_rank_matrix(K):
    """Produce the matrix from the LRdec object.

    Parameters
    ----------
    K : LRdec type object

    Returns
    -------
    2D np.ndarray the matrix K.U x K.V.T modelled by the LRdec object
    """
    return np.inner(K.U, K.V)


def add_low_rank(K, R):
    """Efficient addition of two low-rank matrices.

    efficient computation of sum of low-rank representations
    using this and then get_low_rank_matrix is more efficient than an
    explicit computation if the rank of the final matrix is not full

    Parameters
    ----------
    K, R : LRdec type objects to add

    Returns
    -------
    LRdec type object for sum of K and R
    """
    return LRdec(np.concatenate((K.U, R.U), axis=1), np.concatenate((K.V, R.V), axis=1))


def add_low_rank_one(U, P):
    """Efficient addition of low-rank matrices (symmetric)."""
    return np.concatenate((U, P), axis=1)


def mult_low_rank(K, theta):
    """Efficient multiplication of two low-rank matrices.

    efficient multiplication of sum of low-rank representations
    using this and then get_low_rank_matrix is more efficient than an
    explicit computation if the rank of the final matrix is not full

    Parameters
    ----------
    K, R : LRdec type objects to add

    Returns
    -------
    LRdec type object for product of K and R
    """
    return LRdec(theta * K.U, theta * K.V)


def hadamard_low_rank(K, R):
    """Efficient Hadamard product of two low-rank matrices.

    efficient multiplication of sum of low-rank representations
    using this and then get_low_rank_matrix is more efficient than an
    explicit computation if the rank of the final matrix is not full

    Parameters
    ----------
    K, R : LRdec type objects to add

    Returns
    -------
    LRdec type object for Hadamard product of K and R
    """
    rankK = K.U.shape[1]
    rankR = R.U.shape[1]
    U = np.tile(K.U, rankR) * np.repeat(R.U, rankK, 1)
    V = np.tile(K.V, rankR) * np.repeat(R.V, rankK, 1)
    return LRdec(U, V)


# multiplies U with every component (1st index) of P
# def HadamardLowRankBatch(U, P):
#    rankU = U.shape[1]
#    N = P.shape[0]
#    rankP = P.shape[2]
#    return (np.repeat(np.repeat(np.array(U,ndmin = 3), rankP, 2),N,0)
#           *np.repeat(P,rankU,2))


def hadamard_low_rank_batch(U, P):
    """Hadamard multiply U and P component-wise (1st)."""
    rankU = U.shape[2]
    rankP = P.shape[2]
    return np.tile(U, rankP) * np.repeat(P, rankU, 2)


def hadamard_low_rank_subsample(U, P, rho):
    """Hadamard multiply U and P component-wise (1st), with Nyström type subsampling."""
    rankU = U.shape[2]
    rankP = P.shape[2]
    permut = np.sort(np.random.permutation(range(rankU * rankP))[range(rho)])
    return (np.tile(U, rankP) * np.repeat(P, rankU, 2))[:, :, permut]


def cumsum_rev_low_rank(K):
    """Compute cumsum for LRdec type collections.

    equivalent of cumsum_rev for LRdec type objects

    Parameters
    ----------
    K : LRdec type object to compute cumsum of

    Return
    ------
    LRdec type object for cumsum_rev of K
    """
    return LRdec(cumsum_rev_first(K.U), cumsum_rev_first(K.V))


def sum_low_rank(K):
    """Compute cumsum for LRdec type collections.

    equivalent of sum for LRdec type objects

    Parameters
    ----------
    K : LRdec type object to compute sum of

    Return
    ------
    LRdec type object for sum of K
    """
    return np.inner(sum(K.U), sum(K.V))


# sequential kernel - low-rank version
# ------------------------------------


def sqize_kernel_low_rank(K, L, theta=1.0, normalize=False, rankbound=float("inf")):
    """Compute the sequential kernel from kernel matrix, with low-rank approximation.

    Parameters
    ----------
    K : 2D np.ndarray
        the kernel matrix of increments, i.e.,
        K[i,j] is the kernel between the i-th increment of path 1 (rows),
        and the j-th increment of path 2 (columns)
    L : an integer >= 1, representing the level of truncation
    theta : float, optional, default=1.0
        a positive scaling factor for the levels, i-th level is scaled by theta^i
    normalize : bool, optional, default = False
        whether the output kernel matrix is normalized
        if True, sums and cumsums are divided by prod(K.shape)
    rankbound : int, optional, default = infinity
        a hard threshold for the rank of the level matrices

    Returns
    -------
    a real number, the sequential kernel between path 1 (rows of K) and path 2 (cols)
    """
    # L-1 runs through loop;
    # returns R_ij=(1+\sum_i2>i,j2>j A_i2,j2(1+\sum A_iLjL)...)
    if normalize:
        K = get_low_rank_matrix(K)
        normfac = np.prod(K.shape)
        Id = np.ones(K.shape)
        R = np.ones(K.shape)
        for _ in range(L - 1):
            R = (Id + theta * cumsum_rev(K * R) / normfac) / (1 + theta)
        return (1 + theta * np.sum(K * R) / normfac) / (1 + theta)
    else:
        Id = LRdec(np.ones([K.U.shape[0], 1]), np.ones([K.V.shape[0], 1]))
        # Id = np.ones(K.shape)
        R = Id
        for _ in range(L - 1):
            # todo: execute only if rank is lower than rankbound
            #       reduce to rank
            R = add_low_rank(
                Id, mult_low_rank(cumsum_rev_low_rank(hadamard_low_rank(K, R)), theta)
            )
            # R = Id + cumsum_rev(K * R)
        return 1 + theta * sum_low_rank(hadamard_low_rank(K, R))
        # return 1 + np.sum(K * R)
        # outermost bracket: since i1>=1 and not i1>1 we do it outside of loop


def sqize_kernel_low_rank_fast(
    K,
    L,
    theta=1.0,
    normalize=False,
    rankbound=float("inf"),
):
    """Compute the sequential kernel from kernel matrix, with low-rank approximation.

    Vectorized across series, and using (faster) low-rank approximation across series.

    Parameters
    ----------
    K : 3D np.ndarray
        containing joint low-rank factors (symmetric singular factor)
            1st index counts sequences
            2nd index counts time
            3rd index counts features
        so K[m,:,:] is the mth factor,
        and K[m,:,:] x K[m,:,:]^t is the kernel matrix of the mth factor
    L : an integer >= 1, representing the level of truncation
    theta : float, optional, default=1.0
        a positive scaling factor for the levels, i-th level is scaled by theta^i
    normalize : bool, optional, default = False
        whether the output kernel matrix is normalized
        if True, sums and cumsums are divided by prod(K.shape)
    rankbound : int, optional, default = infinity
        a hard threshold for the rank of the level matrices

    Returns
    -------
    np.ndarray of shape (m, r), where r = min(rankbound, K.shape[2])
        a matrix R such that R*R^t is the sequential kernel matrix
        R*R^t[i,j] is sequential kernel (low-rank) between i-th and j-th sequence in K
    """
    if normalize:
        Ksize = K.shape[0]
        B = np.ones([Ksize, 1, 1])
        R = np.ones([Ksize, 1])

        for _ in range(L):
            P = np.sqrt(theta) * hadamard_low_rank_batch(K, B) / Ksize
            B = cumsum_shift_mult(P, [1])

            if rankbound < B.shape[2]:
                # B = rankreduce_batch(B, rankbound)
                permut = np.sort(
                    np.random.permutation(range(B.shape[2]))[range(rankbound)]
                )
                B = B[:, :, permut]

            R = np.concatenate((R, np.sum(B, axis=1)), axis=1) / (np.sqrt(1 + theta))

        return R

    else:
        Ksize = K.shape[0]
        B = np.ones([Ksize, 1, 1])
        R = np.ones([Ksize, 1])

        for _ in range(L):
            # todo: execute only if rank is lower than rankbound
            #       reduce to rank
            P = np.sqrt(theta) * hadamard_low_rank_batch(K, B)
            B = cumsum_shift_mult(P, [1])

            if rankbound < B.shape[2]:
                # B = rankreduce_batch(B, rankbound)
                permut = np.sort(
                    np.random.permutation(range(B.shape[2]))[range(rankbound)]
                )
                B = B[:, :, permut]

            R = np.concatenate((R, np.sum(B, axis=1)), axis=1)

        return R


# sequential kernel - wraps all versions
# --------------------------------------


def seq_kernel(
    X,
    kernelfun=None,
    L=2,
    D=1,
    theta=1.0,
    normalize=False,
    lowrank=False,
    rankbound=float("inf"),
):
    """Compute the sequential kernel between sequence/time series.

    Provides interface for vanilla sequential kernel, low-rank, and higher-order.

    Parameters
    ----------
    X : 3D np.ndarray of shape (N, d, _)
        collection of sequences/time series
        1st index = instance index
        2nd index = variable/feature index
        3rd index = time index
    kernelfun : function (2D np.ndarray x 2D np.ndarray) -> 2D np.ndarray
        pairwise kernel function, matrix sizes (n, d) x (m, d) -> (n x m)
        optional, default = Euclidean (linear) kernel with scale parameter 1
    L : int, optional, default = 2
        an integer >= 1, representing the level of truncation
    D : int, optional, default = 1
        an integer >= 1, representing the order of approximation
        can be set only if lowrank = False, otherwise ignored (always = 1)
    theta : float, optional, default=1.0
        a positive scaling factor for the levels, i-th level is scaled by theta^i
    normalize : bool, optional, default = False
        whether the output kernel matrix is normalized
        if True, sums and cumsums are divided by prod(K.shape)
    lowrank : bool, optional, default = False
        whether to use low rank approximation in computing the kernel
    rankbound : int, optional, default = infinity
        a hard threshold for the rank of the level matrices
        used only if lowrank = True

    Returns
    -------
    np.ndarray of shape (N, N), sequential kernel matrix
        [i,j]-th entry is sequential kernel between X[i] and X[j]
    """
    N = np.shape(X)[0]
    KSeq = np.zeros((N, N))

    if kernelfun is None:
        kernelfun = partial(k_euclid, scale=1)

    if not lowrank:
        if D == 1:
            for row1ind in range(N):
                for row2ind in range(row1ind + 1):
                    KSeq[row1ind, row2ind] = sqize_kernel(
                        K=kernelfun(X[row1ind].T, X[row2ind].T),
                        L=L,
                        theta=theta,
                        normalize=normalize,
                    )
        else:
            for row1ind in range(N):
                for row2ind in range(row1ind + 1):
                    KSeq[row1ind, row2ind] = sqize_kernel_ho(
                        K=kernelfun(X[row1ind].T, X[row2ind].T),
                        L=L,
                        D=D,
                        theta=theta,
                        normalize=normalize,
                    )
    else:
        R = sqize_kernel_low_rank_fast(
            K=X.transpose([0, 2, 1]),
            L=L,
            theta=theta,
            normalize=normalize,
            rankbound=rankbound,
        )
        KSeq = np.inner(R, R)
        # todo: kernelfun gives back a LRdec object
        #  for now, linear low-rank approximation is done
        # KSeq[row1ind,row2ind] = qqize_kernel_low_rank(
        #   kernelfun(X[row1ind].T,X[row2ind].T),L,theta,normalize = True)
    return mirror(KSeq)


def seq_kernel_XY(
    X,
    Y=None,
    kernelfun=None,
    L=2,
    D=1,
    theta=1.0,
    normalize=False,
    lowrank=False,
    rankbound=float("inf"),
):
    """Compute the sequential kernel between two different collections of sequence.

    Provides interface for vanilla sequential kernel, low-rank, and higher-order.

    Parameters
    ----------
    X : 3D np.ndarray of shape (N, d, _)
        collection of sequences/time series
        1st index = instance index
        2nd index = variable/feature index
        3rd index = time index
    Y : 3D np.ndarray of shape (M, d, _)
        collection of sequences/time series
        1st index = instance index
        2nd index = variable/feature index
        3rd index = time index
    kernelfun : function (2D np.ndarray x 2D np.ndarray) -> 2D np.ndarray
        pairwise kernel function, matrix sizes (n, d) x (m, d) -> (n x m)
        optional, default = Euclidean (linear) kernel with scale parameter 1
    L : int, optional, default = 2
        an integer >= 1, representing the level of truncation
    D : int, optional, default = 1
        an integer >= 1, representing the order of approximation
        can be set only if lowrank = False, otherwise ignored (always = 1)
    theta : float, optional, default=1.0
        a positive scaling factor for the levels, i-th level is scaled by theta^i
    normalize : bool, optional, default = False
        whether the output kernel matrix is normalized
        if True, sums and cumsums are divided by prod(K.shape)
    lowrank : bool, optional, default = False
        whether to use low rank approximation in computing the kernel
    rankbound : int, optional, default = infinity
        a hard threshold for the rank of the level matrices
        used only if lowrank = True

    Returns
    -------
    np.ndarray of shape (N, M), sequential kernel matrix
        [i,j]-th entry is sequential kernel between X[i] and Y[j]
    """
    # if no Y is passed, call seq_kernel
    if Y is None:
        return seq_kernel(
            X=X,
            kernelfun=kernelfun,
            L=L,
            D=D,
            theta=theta,
            normalize=normalize,
            lowrank=lowrank,
            rankbound=rankbound,
        )

    N = np.shape(X)[0]
    M = np.shape(Y)[0]

    KSeq = np.zeros((N, M))

    if kernelfun is None:
        kernelfun = partial(k_euclid, scale=1)

    kwargs = {"L": L, "theta": theta, "normalize": normalize}

    if not lowrank:
        if D == 1:
            for row1ind in range(N):
                for row2ind in range(M):
                    KSeq[row1ind, row2ind] = sqize_kernel(
                        K=kernelfun(X[row1ind].T, Y[row2ind].T), **kwargs
                    )
        else:
            for row1ind in range(N):
                for row2ind in range(M):
                    KSeq[row1ind, row2ind] = sqize_kernel_ho(
                        K=kernelfun(X[row1ind].T, Y[row2ind].T), D=D, **kwargs
                    )
    else:
        U = sqize_kernel_low_rank_fast(
            K=X.transpose([0, 2, 1]), rankbound=rankbound, **kwargs
        )
        V = sqize_kernel_low_rank_fast(
            K=Y.transpose([0, 2, 1]), rankbound=rankbound, **kwargs
        )
        KSeq = np.inner(U, V)
        # KSeq = np.inner(sqize_kernel_low_rank_fast(X, L, theta, normalize),
        # wqize_kernel_low_rank_fast(Y, L, theta, normalize))

    return KSeq


def data_tabulator(X):
    """Tabulates sequence 3D np.ndarray into sklearn compatible 2D np.ndarray format."""
    Xshape = np.shape(X)
    return np.reshape(X, (Xshape[0], np.prod(Xshape[1:])))


def time_series_reshaper(Xflat, numfeatures, subsample=1, differences=True):
    """Convert 2D np.ndarray into a time series 3D np.ndarray.

    Useful as part of sklearn pipeline for internal conversion to time series 3D format.

    optionally, subsamples or differences time series

    Parameters
    ----------
    Xflat : 2D np.ndarray (instances, flattened time series)
    numfeatures : number of features/variables in the time series
    subsample : int, optional, default = 1
        time index step size to sub-sample
    differences : bool, optional, default = True
        whether to take first temporal differences (True) or not (False)

    Returns
    -------
    Xflat, as a 3D array
        regular subsampling is applied if subsample > 1
        differencing is applied (after subsampling) if differences=True
    """
    flatXshape = np.shape(Xflat)
    Xshape = (flatXshape[0], numfeatures, flatXshape[1] / numfeatures)
    X = np.reshape(Xflat, Xshape)[:, :, ::subsample]

    if differences:
        return np.diff(X)
    else:
        return X


# scikit-learn estimator (original 2016 paper)
# --------------------------------------------
# for historical reasons - old scikit-learn version of the estimator
class SeqKernelizer(BaseEstimator, TransformerMixin):
    """Compute the sequential kernel matrix row features on collection of series.

    Original implementation of signature kernel in [1]_ and [2]_.
    The algorithm used therein is sklearn GridSearchCV wrapping
    the sklearn pipeline of SeqKernelizer and SVC.

    Identical behaviour can be obtained in sktime as
    a grid search wrapped DistanceFeatures(SignatureKernel()) * SVC() pipeline

    Included for historical purposes only, as reference to original paper code,
    and for reproduction of the original experiments in the JMLR publication.
    Users and developers should use/modify SequentialKernel instead.

    This sklearn estimator requires passing of integer "numfeatures" as parameter,
    and will interpret rows of X as time series with `numfeatures` features/vars,
    and X.shape[1]/numfeatures time stamps, reshaped in (vars, time stamps) order.

    In transform, will transform a series to the row of the kernel matrix
    between that series and all the series seen in fit,, via seq_kernel_XY.

    Parameters
    ----------
    level : int, optional, default = 2
        an integer >= 1, representing the level of truncation of the sequential kernel
    degree : int, optional, default = 1
        an integer >= 1, representing the order of approximation of sequential kernel
        can be set only if lowrank = False, otherwise ignored (always = 1)
    theta : float, optional, default=1.0
        a positive scaling factor for the levels, i-th level is scaled by theta^i
    kernel : str, one of "linear", "Gauss", "Laplace", "poly"
        code for inner kernel in the sequential kernel, with kernel parameters
        "linear" - Euclidean kernel with scale parameter
        "Gauss" - Gaussian kernel with scale parameter
        "Laplace" - Laplace kernel with scale parameter
        "poly" - polynomial kernel with degree deg and scale parameter
    scale : float, optional, default = 1.0
        a positive scaling factor for the inner kernel
    degree : int, optional, default = 1, used only for polynomial kernel (kernel="poly")
        degree of the polynomial kernel (if used)
    numfeatures : int, optional, default = 2
        number of features/variables in the time series
    subsample : int, optional, default = 1
        time index step size to sub-sample
    differences : bool, optional, default = True
        whether to take first temporal differences (True) or not (False)
    normalize : bool, optional, default = False
        whether the output kernel matrix is normalized
        if True, sums and cumsums are divided by prod(K.shape)
    lowrank : bool, optional, default = False
        whether to use low rank approximation in computing the kernel
    rankbound : int, optional, default = infinity
        a hard threshold for the rank of the level matrices
        used only if lowrank = True

    References
    ----------
    .. [1] F. Kiraly, H. Oberhauser. 2016. "Kernels for sequentially ordered data.",
        arXiv: 1601.08169.
    .. [2] F. Kiraly, H. Oberhauser. 2019. "Kernels for sequentially ordered data.",
        Journal of Machine Learning Research.
    """

    def __init__(
        self,
        level=2,
        degree=1,
        theta=1,
        kernel="linear",
        scale=1,
        deg=2,
        numfeatures=2,
        subsample=100,
        differences=True,
        normalize=False,
        lowrank=False,
        rankbound=float("inf"),
    ):
        self.level = level
        self.degree = degree
        self.theta = theta
        self.kernel = kernel
        self.scale = scale
        self.deg = deg
        self.numfeatures = numfeatures
        self.subsample = subsample
        self.differences = differences
        self.normalize = normalize
        self.lowrank = lowrank
        self.rankbound = rankbound

        self._reshape_kwargs = {
            "numfeatures": numfeatures,
            "subsample": subsample,
            "differences": differences,
        }
        self._kern_kwargs = {
            "level": level,
            "degree": degree,
            "theta": theta,
            "normalize": normalize,
            "lowrank": lowrank,
            "rankbound": rankbound,
        }

    def fit(self, X, y=None):
        """Fit = reshape the series X."""
        self._X = time_series_reshaper(X, **self._reshape_kwargs)
        return self

    def transform(self, X):
        """Transform the data to kernel matrix rows."""
        X = time_series_reshaper(X, **self._reshape_kwargs)

        def kernselect(kername):
            switcher = {
                "linear": partial(k_euclid, scale=self.scale),
                "Gauss": partial(k_gauss, scale=self.scale),
                "Laplace": partial(k_laplace, scale=self.scale),
                "poly": partial(k_polynom, scale=self.scale, deg=self.deg),
            }
            return switcher.get(kername, "nothing")

        KSeq = seq_kernel_XY(X, self._X, kernselect(self.kernel), **self._kern_kwargs)

        return KSeq


# sktime interface - pairwise transformer
# ---------------------------------------


class SignatureKernel(BasePairwiseTransformerPanel):
    """Time series signature kernel, including high-order and low-rank variants.

    Implements the signature kernel of Kiraly et al, see [1]_ and [2]_,
    including higher-order and low-rank approximation variants described therein.

    Parameters
    ----------
    kernel : sktime pairwise (tabular) transformer, callable, or None
        inner (tabular) kernel used in the signature sequence kernel
        if callable: function (2D np.ndarray x 2D np.ndarray) -> 2D np.ndarray
        pairwise kernel function, matrix sizes (n, d) x (m, d) -> (n x m)
        optional, default = None = Euclidean (linear) kernel with scale parameter 1
    level : int, optional, default = 2
        an integer >= 1, representing the level of truncation of the sequential kernel
    degree : int, optional, default = 1
        an integer >= 1, representing the order of approximation of sequential kernel
        can be set only if lowrank = False, otherwise ignored (always = 1)
    theta : float, optional, default=1.0
        a positive scaling factor for the levels, i-th level is scaled by theta^i
    normalize : bool, optional, default = False
        whether the output kernel matrix is normalized
        if True, sums and cumsums are divided by prod(K.shape)
    lowrank : bool, optional, default = False
        whether to use low rank approximation in computing the kernel
    rankbound : int, optional, default = infinity
        a hard threshold for the rank of the level matrices
        used only if lowrank = True

    References
    ----------
    .. [1] F. Kiraly, H. Oberhauser. 2016. "Kernels for sequentially ordered data.",
        arXiv: 1601.08169.
    .. [2] F. Kiraly, H. Oberhauser. 2019. "Kernels for sequentially ordered data.",
        Journal of Machine Learning Research.
    """

    _tags = {"X_inner_mtype": "numpy3D", "pwtrafo_type": "kernel"}

    def __init__(
        self,
        kernel=None,
        level=2,
        degree=1,
        theta=1,
        normalize=False,
        lowrank=False,
        rankbound=float("inf"),
    ):
        self.kernel = kernel
        self.level = level
        self.degree = degree
        self.theta = theta
        self.normalize = normalize
        self.lowrank = lowrank
        self.rankbound = rankbound

        if kernel is None:
            self._kernel = partial(k_euclid, scale=1)
        else:
            self._kernel = kernel

        self._kern_kwargs = {
            "L": level,
            "D": degree,
            "theta": theta,
            "normalize": normalize,
            "lowrank": lowrank,
            "rankbound": rankbound,
        }

        super().__init__()

    def _transform(self, X, X2=None):
        """Compute distance/kernel matrix.

        private _transform containing core logic, called from public transform

        Behaviour: returns pairwise distance/kernel matrix
            between samples in X and X2
                if X2 is not passed, is equal to X
                if X/X2 is a pd.DataFrame and contains non-numeric columns,
                    these are removed before computation

        Parameters
        ----------
        X: 3D np.array of shape [num_instances, num_vars, num_time_points]
        X2: 3D np.array of shape [num_instances, num_vars, num_time_points], optional
            default X2 = X

        Returns
        -------
        distmat: np.array of shape [n, m]
            (i,j)-th entry contains distance/kernel between X[i] and X2[j]
        """
        kwargs = self._kern_kwargs
        kernel = self._kernel

        if X2 is None:
            return seq_kernel(X, kernelfun=kernel, **kwargs)
        else:
            return seq_kernel_XY(X, X2, kernelfun=kernel, **kwargs)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for distance/kernel transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        param1 = {}

        # test higher level and normalization
        param2 = {"level": 3, "normalize": True}

        # test higher-order function
        param3 = {"degree": 2}

        # test low-rank approximation
        param4 = {"lowrank": True}

        paramlist = [param1, param2, param3, param4]

        return paramlist
