# -*- coding: utf-8 -*-
"""Signature kernels from Kiraly et al, 2015."""

__author__ = ["fkiraly"]

import collections
import numpy as np
from scipy.sparse.linalg import svds

from sktime.dists_kernels._base import BasePairwiseTransformerPanel


# cumsum varia
# ------------
def _coerce_to_list(i):
    """Coerce integers to list of integers."""
    if not isinstance(i, list):
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
    out[:-1, :-1] = np.cumsum(np.cumsum(array[:0:-1, :0:-1], 0),1)[::-1, ::-1]
    return out


def cumsum_mult(array, dims):
    """Cumsum over all axes in dims."""
    dims = _coerce_to_list(dims)
    for dimind in dims:
        array = np.cumsum(array, axis=dimind)
    return array


def roll_mult(array, shift, dims):
    """Roll over all axes in dims."""
    dims = _coerce_to_list(dims)
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
    arraysvd = svds(array.astype('f'), k=rankbound)
    return np.dot(arraysvd[0],np.diag(arraysvd[1]))


def rankreduce_batch(arrays, rankbound):
    """Apply rankreduce to axis 0 stack of arrays."""
    resultarrays = np.zeros([arrays.shape[0], arrays.shape[1], rankbound])
    for i in range(arrays.shape[0]):
        resultarrays[i, :, :] = rankreduce(arrays[i, :, :], rankbound)
    return resultarrays


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
    return np.exp(-(scale ** 2) * sqdist(x, y)/2)


def k_euclid(x, y, scale):
    """Euclidea kernel with scale coeff.""" 
    return scale * np.inner(x, y)


def k_laplace(x, y, scale):
    """Laplace kernel with scale coeff."""
    return np.exp(-scale * np.sqrt(np.inner(x-y, x-y)))


def k_tanh(x, y, off, scale):
    """Tanh kernel with scale and offset."""
    return np.tanh(off + scale * np.inner(x, y))


def mirror(K):
    """Mirrors an upper triangular kernel matrix, helper for Sqize_kernel."""
    return K - np.diag(np.diag(K)) + np.transpose(K)


def sqize_kernel(K, L, theta=1.0, normalize=False):
    """Compute the sequential kernel from a sequential kernel matrix.

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
        for _ in range(L-1):
            R = (I + theta*cumsum_rev(K * R)/normfac)/(1+theta)
        return (1 + theta*np.sum(K * R)/normfac)/(1+theta)
    else:
        Id = np.ones(K.shape)
        R = np.ones(K.shape)
        for _ in range(L - 1):
            R = Id + cumsum_rev(K * R)  # A*R is componentwise
        # outermost bracket: since i1>=1 and not i1>1 we do it outside of loop
        return 1 + np.sum(K*R)


def sqize_kernel_ho(K, L, D=1, theta=1.0, normalize=False):
    """Compute the higher-order sequential kernel from a sequential kernel matrix.

    aka compute "sequentialization of the kernel K", higher-order approximation

    Parameters
    ----------
    K : 2D np.ndarray
        the kernel matrix of increments, i.e.,
        K[i,j] is the kernel between the i-th increment of path 1 (rows),
        and the j-th increment of path 2 (columns)
    L : an integer >= 1, representing the level of truncation
    D : int, optional, default = 1
        an integer \geq 1, representing the order of approximation
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
        Acs = cumsum_shift_mult(np.sum(A[ell-1, :, :, :, :], (0, 1)), (0, 1))
        A[ell, 0, 0, :, :] = K * (Id + Acs)
        for d1 in range(1, Dprime):
            Acs1 = cumsum_shift_mult(np.sum(A[ell-1, d1-1, :, :, :], 0), 1)
            Acs2 = cumsum_shift_mult(np.sum(A[ell-1, :, d1-1, :, :], 0), 0)
            A[ell, d1, 0, :, :] = A[ell, d1, 0, :, :] + (1/d1) * K * Acs1
            A[ell, :, d1, :, :] = A[ell, 0, d1, :, :] + (1/d1) * K * Acs2

            for d2 in range(1, Dprime):
                Acs12 = cumsum_shift_mult(np.sum(A[ell-1, d1-1, d2-1, :, :], 0), 0)
                A[ell, d1, d2, :, :] = A[ell, d1, d2, :, :] + (1/(d1*d2)) * K * Acs12
                
    return 1 + np.sum(A[L-1, :, :, :, :])



# low-rank decomposition
#  models matrix A = U x V.T
#  U and V should be *arrays*, not *matrices*
LRdec = collections.namedtuple('LRdec', ['U','V'])


# FUNCTION GetLowRankMatrix
#  produce the matrix from the LRdec object
#
# Inputs:
#  K            a LRdec type object
#
# Output:
#  the matrix K.U x K.V.T modelled by the LRdec object
def GetLowRankMatrix(K):
    return np.inner(K.U, K.V)


# FUNCTION AddLowRank
#  efficient computation of sum of low-rank representations
#   using this and then GetLowRankMatrix is more efficient than an
#   explicit computation if the rank of the final matrix is not full
#
# Inputs:
#  K, R           LRdec type objects to add
#
# Output:
#  LRdec type object for sum of K and R

def AddLowRank(K, R):
    return LRdec(np.concatenate((K.U,R.U), axis=1),np.concatenate((K.V,R.V), axis=1))

def AddLowRankOne(U, P):
    return np.concatenate((U,P), axis=1)


def MultLowRank(K, theta):
    return LRdec(theta*K.U, theta*K.V)


# FUNCTION HadamardLowRank
#  efficient computation of Hadamard product of low-rank representations
#   using this and then GetLowRankMatrix is more efficient than an
#   explicit computation if the rank of the final matrix is not full
#
# Inputs:
#  K, R           LRdec type objects to multiply
#
# Output:
#  LRdec type object for Hadamard product of K and R

def HadamardLowRank(K, R):
    rankK = K.U.shape[1]
    rankR = R.U.shape[1]
    U = (np.tile(K.U,rankR)*np.repeat(R.U,rankK,1))
    V = (np.tile(K.V,rankR)*np.repeat(R.V,rankK,1))
    return LRdec(U,V)
    
# multiplies U with every component (1st index) of P
#def HadamardLowRankBatch(U, P):
#    rankU = U.shape[1]
#    N = P.shape[0]
#    rankP = P.shape[2]
#    return (np.repeat(np.repeat(np.array(U,ndmin = 3), rankP, 2),N,0)*np.repeat(P,rankU,2))  

# multiplies U and P component-wise (1st)
def HadamardLowRankBatch(U, P):
    rankU = U.shape[2]
    rankP = P.shape[2]
    return (np.tile(U,rankP)*np.repeat(P,rankU,2))  

# with Nystroem type subsampling
def HadamardLowRankSubS(U, P, rho):
    rankU = U.shape[2]
    rankP = P.shape[2]
    permut = np.sort(np.random.permutation(range(rankU*rankP))[range(rho)])
    return (np.tile(U,rankP)*np.repeat(P,rankU,2))[:,:,permut]
 
    
    
# FUNCTION cumsum_LowRank
# cumsum for LRdec type collections
#  equivalent of cumsum_rev for LRdec type objects
#
# Inputs:
#  K            LRdec type object to cumsum
#
# Output:
#  LRdec type object for cumsum_rev of K

def cumsum_LowRank(K):
    return LRdec(cumsum_rev_first(K.U),cumsum_rev_first(K.V))
    
    
# FUNCTION sum_LowRank
# sum for LRdec type collections
#  equivalent of sum_rev for LRdec type objects
#
# Inputs:
#  K            LRdec type object to sum
#
# Output:
#  LRdec type object for sum of K
def sum_LowRank(K):
    return np.inner(sum(K.U),sum(K.V))
    

# FUNCTION Sqize_kernelLowRank
#  computes the sequential kernel from a sequential kernel matrix
#   faster by using a low-rank approximation
#
# Inputs:
#  K              LRdec type object, models low-rank factors
#                   of the increment kernel matrix K such that K = K.U x K.V.T
#                 where K[i,j] is the kernel between the i-th increment of path 1,
#                  and the j-th increment of path 2
#  L             an integer \geq 1, representing the level of truncation
# optional:
#  theta         a positive scaling factor for the levels, i-th level by theta^i
#  normalize     whether the output kernel matrix is normalized
#  rankbound     a hard threshold for the rank of the level matrices
#    defaults: theta = 1.0, normalize = False, rankbound = infinity
#
# Output:
#  a real number, the sequential kernel between path 1 and path 2
#
def Sqize_kernelLowRank(K, L, theta = 1.0, normalize = False, rankbound = float("inf")):
    #L-1 runs through loop;
    #returns R_ij=(1+\sum_i2>i,j2>j A_i2,j2(1+\sum A_iLjL)...)
    if normalize:
        K = GetLowRankMatrix(K)
        normfac = np.prod(K.shape)
        I = np.ones(K.shape)
        R = np.ones(K.shape)
        for l in range(L-1):
            R = (I + theta*cumsum_rev(K*R)/normfac)/(1+theta)
        return (1 + theta*np.sum(K*R)/normfac)/(1+theta)
    else:
        I = LRdec(np.ones([K.U.shape[0],1]),np.ones([K.V.shape[0],1]))
         # I = np.ones(K.shape)
        R = I
        for l in range(L-1):
            #todo: execute only if rank is lower than rankbound
            #       reduce to rank
            R = AddLowRank(I,MultLowRank(cumsum_LowRank(HadamardLowRank(K,R)),theta))
            #R=I + cumsum_rev(K*R)
        return 1 + theta*sum_LowRank(HadamardLowRank(K,R)) 
#        return 1 + np.sum(K*R)
        #outermost bracket: since i1>=1 and not i1>1 we do it outside of loop


# FUNCTION Sqize_kernel_low_rank_fast
#  computes the sequential kernel from a sequential kernel matrix
#   faster by using a low-rank approximation
#
# Inputs:
#  K              Array of dimension 3, containing joint low-rank factors
#                  1st index counts sequences
#                  2nd index counts time
#                  3rd index counts features
#                   so K[m,:,:] is the mth factor,
#                    and K[m,:,:] x K[m,:,:]^t is the kernel matrix of the mth factor
#  L             an integer \geq 1, representing the level of truncation
# optional:
#  theta         a positive scaling factor for the levels, i-th level by theta^i
#  normalize     whether the output kernel matrix is normalized
#  rankbound     a hard threshold for the rank of the level matrices
#    defaults: theta = 1.0, normalize = False, rankbound = infinity
#
# Output:
#  a matrix R such that R*R^t is the sequential kernel matrix
#
def Sqize_kernel_low_rank_fast(K, L, theta = 1.0, normalize = False, rankbound = float("inf")):

    if normalize:

        Ksize = K.shape[0]
        B = np.ones([Ksize,1,1])
        R = np.ones([Ksize,1])

        for l in range(L):
            
            P = np.sqrt(theta)*HadamardLowRankBatch(K,B)/Ksize
            B = cumsum_shift_mult(P,[1])
            
            if rankbound < B.shape[2]:
                #B = rankreduce_batch(B,rankbound)
                permut = np.sort(np.random.permutation(range(B.shape[2]))[range(rankbound)])
                B = B[:,:,permut]
                
            R = np.concatenate((R,np.sum(B,axis = 1)), axis=1)/(np.sqrt(1+theta))
            
        return R
        
    else:

        Ksize = K.shape[0]
        B = np.ones([Ksize,1,1])
        R = np.ones([Ksize,1])

        for l in range(L):
            #todo: execute only if rank is lower than rankbound
            #       reduce to rank
            P = np.sqrt(theta)*HadamardLowRankBatch(K,B)
            B = cumsum_shift_mult(P,[1])

            if rankbound < B.shape[2]:
                #B = rankreduce_batch(B,rankbound)
                permut = np.sort(np.random.permutation(range(B.shape[2]))[range(rankbound)])
                B = B[:,:,permut]
                
            R = np.concatenate((R,np.sum(B,axis = 1)), axis=1)

        return R


# In[]
# FUNCTION SeqKernel
#  computes the sequential kernel matrix for a dataset of time series
def seq_kernel(
    X,
    kernelfun,
    L=2,
    D=1,
    theta=1.0,
    normalize=False,
    lowrank=False,
    rankbound=float("inf"),
):
    
    N = np.shape(X)[0]   
    KSeq = np.zeros((N,N))

    if not(lowrank):
        if D == 1:
            for row1ind in range(N):
                for row2ind in range(row1ind+1):
                    KSeq[row1ind,row2ind] = sqize_kernel(kernelfun(X[row1ind].T,X[row2ind].T),L,theta,normalize)
        else:
            for row1ind in range(N):
                for row2ind in range(row1ind+1):
                    KSeq[row1ind,row2ind] = sqize_kernel_ho(kernelfun(X[row1ind].T,X[row2ind].T),L,D,theta,normalize)
    else:                
        R = Sqize_kernel_low_rank_fast(X.transpose([0,2,1]), L, theta, normalize)
        KSeq = np.inner(R,R)             
        # todo: kernelfun gives back a LRdec object
        #  for now, linear low-rank approximation is done
        # KSeq[row1ind,row2ind] = Sqize_kernelLowRank(kernelfun(X[row1ind].T,X[row2ind].T),L,theta,normalize = True)

    return mirror(KSeq)

    
# FUNCTION SeqKernel
#  computes sequential cross-kernel matrices
def SeqKernelXY(X,Y,kernelfun,L=2,D=1,theta=1.0,normalize = False,lowrank = False,rankbound = float("inf")):

    N = np.shape(X)[0]   
    M = np.shape(Y)[0]   
    
    KSeq = np.zeros((N,M))
    
    if not(lowrank):
        if D == 1:
            for row1ind in range(N):
                for row2ind in range(M):
                    KSeq[row1ind,row2ind] = sqize_kernel(kernelfun(X[row1ind].T,Y[row2ind].T),L,theta,normalize)
        else:
            for row1ind in range(N):
                for row2ind in range(M):
                    KSeq[row1ind,row2ind] = sqize_kernel_ho(kernelfun(X[row1ind].T,Y[row2ind].T),L,D,theta,normalize)
    else:
        
        KSeq = np.inner(sqize_kernel_low_rank_fast(X.transpose([0,2,1]), L, theta, normalize, rankbound),sqize_kernel_low_rank_fast(Y.transpose([0,2,1]), L, theta, normalize, rankbound))             
        #KSeq = np.inner(sqize_kernel_low_rank_fast(X, L, theta, normalize),Sqize_kernel_ow_rank_fast(Y, L, theta, normalize))             
                
    return KSeq
    


# In[]
# FUNCTION DataTabulator(X)
def DataTabulator(X):
    
    Xshape = np.shape(X)
        
    return np.reshape(X,(Xshape[0],np.prod(Xshape[1:])))




# In[]
# FUNCTION TimeSeriesReshaper
#  makes a 3D time series array out of a 2D data array
def TimeSeriesReshaper(Xflat, numfeatures, subsample = 1, differences = True):
    flatXshape = np.shape(Xflat)
    Xshape = (flatXshape[0], numfeatures, flatXshape[1]/numfeatures)        
    X = np.reshape(Xflat,Xshape)[:,:,::subsample]
    
    if differences:
        return np.diff(X)
    else:    
        return X
        

# In[3]

# CLASS SeqKernelizer
#  pipelines pre-processing of a time series datset with support vector classifier
#
# parameters:
#  Level, theta: parameters in of the sequentialization
#   Level = cut-off degree
#   theta = scaling factor
#  kernel, scale, deg: parameter for the primary kernel
#   kernel = name of the kernel used: linear, Gauss, Laplace, poly
#   scale = scaling constant, multiplicative to scalar product
#   deg = degree, for polynomial kernel
#  subsample, numfeatures, differences:
#   pre-processing parameters for time series.
#    numfeatures = number of features per time point, for internal reshaping
#    subsample = time series is subsampled to every subsample-th time point
#    differences = whether first differences are taken or not
#    lowrank = whether low-rank approximations are used or not
#
from sklearn.base import BaseEstimator, TransformerMixin

class SeqKernelizer(BaseEstimator, TransformerMixin):
    def __init__(self, Level = 2, Degree = 1, theta = 1, kernel = 'linear', 
                 scale = 1, deg = 2, X = np.zeros((1,2)), 
                 numfeatures = 2, subsample = 100, differences = True, 
                 normalize = False, lowrank = False, rankbound = float("inf")):
        self.Level = Level
        self.Degree = Degree
        self.theta = theta
        self.subsample = subsample
        self.kernel = kernel
        self.scale = scale
        self.deg = deg
        self.numfeatures = numfeatures
        self.differences = differences
        self.normalize = normalize
        self.lowrank = lowrank
        self.rankbound = rankbound
        self.X = X
        
    def fit(self, X, y=None):
        self.X = TimeSeriesReshaper(X,self.numfeatures,self.subsample,self.differences)
        return self
        
    def transform(self, Y):
        
        Y = TimeSeriesReshaper(Y,self.numfeatures,self.subsample,self.differences)
        
        kPolynom = lambda x,y,scale,deg : (1+scale*np.inner(x,y))**deg
        kGauss = lambda x,y,scale: np.exp(-(scale**2)*sqdist(x,y)/2)
        kEuclid = lambda x,y,scale: scale*np.inner(x,y)
        kLaplace = lambda x,y,scale: np.exp(-scale*np.sqrt(np.inner(x-y,x-y)))
        
        def kernselect(kername):
            switcher = {
                'linear': lambda x,y: kEuclid(x,y,self.scale),
                'Gauss': lambda x,y: kGauss(x,y,self.scale),
                'Laplace': lambda x,y: kLaplace(x,y,self.scale),
                'poly': lambda x,y: kPolynom(x,y,self.scale,self.deg),
                }
            return switcher.get(kername, "nothing")
            
        KSeq = SeqKernelXY(Y,self.X,kernselect(self.kernel),self.Level,self.Degree,self.theta,self.normalize,self.lowrank,self.rankbound)
        
        return KSeq


class SignatureKernel(BasePairwiseTransformerPanel):
    r"""Interface to sktime native edit distances.

    Interface to the following edit distances:
    LCSS - longest common subsequence distance
    ERP - Edit distance for real penalty
    EDR - Edit distance for real sequences
    TWE - Time warp edit distance

    LCSS [1]_ attempts to find the longest common sequence between two time series and
    returns a value that is the percentage that longest common sequence assumes.
    LCSS is computed by matching indexes that are
    similar up until a defined threshold (epsilon).

    The value returned will be between 0.0 and 1.0, where 0.0 means the two time series
    are exactly the same and 1.0 means they are complete opposites.

    EDR [2]_ computes the minimum number of elements (as a percentage) that must be
    removed from x and y so that the sum of the distance between the remaining
    signal elements lies within the tolerance (epsilon).

    The value returned will be between 0 and 1 per time series. The value will
    represent as a percentage of elements that must be removed for the time series to
    be an exact match.

    ERP [3]_ attempts align time series
    by better considering how indexes are carried forward through the cost matrix.
    Usually in the dtw cost matrix, if an alignment can't be found the previous value
    is carried forward. ERP instead proposes the idea of gaps or sequences of points
    that have no matches. These gaps are then punished based on their distance from 'g'.

    TWE [4]_ is a distance measure for discrete time series
    matching with time 'elasticity'. In comparison to other distance measures, (e.g.
    DTW (Dynamic Time Warping) or LCS (Longest Common Subsequence Problem)), TWE is a
    metric. Its computational time complexity is O(n^2), but can be drastically reduced
    in some specific situation by using a corridor to reduce the search space. Its
    memory space complexity can be reduced to O(n).

    Parameters
    ----------
    distance: str, one of ["lcss", "edr", "erp", "twe"], optional, default = "lcss"
        name of the distance that is calculated
    window: float, default = None
        Float that is the radius of the sakoe chiba window (if using Sakoe-Chiba
        lower bounding). Value must be between 0. and 1.
    itakura_max_slope: float, default = None
        Gradient of the slope for itakura parallelogram (if using Itakura
        Parallelogram lower bounding)
    bounding_matrix: 2D np.ndarray, optional, default = None
        if passed, must be of shape (len(X), len(X2)) for X, X2 in `transform`
        Custom bounding matrix to use. If defined then other lower_bounding params
        are ignored. The matrix should be structure so that indexes considered in
        bound should be the value 0. and indexes outside the bounding matrix should
        be infinity.
    epsilon : float, defaults = 1.
        Used in LCSS, EDR, ERP, otherwise ignored
        Matching threshold to determine if two subsequences are considered close
        enough to be considered 'common'.
    g: float, defaults = 0.
        Used in ERP, otherwise ignored.
        The reference value to penalise gaps.
    lmbda: float, optional, default = 1.0
        Used in TWE, otherwise ignored.
        A constant penalty that punishes the editing efforts. Must be >= 1.0.
    nu: float optional, default = 0.001
        Used in TWE, otherwise ignored.
        A non-negative constant which characterizes the stiffness of the elastic
        twe measure. Must be > 0.
    p: int optional, default = 2
        Used in TWE, otherwise ignored.
        Order of the p-norm for local cost.

    References
    ----------
    .. [1] M. Vlachos, D. Gunopoulos, and G. Kollios. 2002. "Discovering
        Similar Multidimensional Trajectories", In Proceedings of the
        18th International Conference on Data Engineering (ICDE '02).
        IEEE Computer Society, USA, 673.
    """

    _tags = {
        "symmetric": True,  # all the distances are symmetric
        "X_inner_mtype": "numpy3D",
    }

    ALLOWED_DISTANCE_STR = ["lcss", "edr", "erp", "twe"]

    def __init__(
        self,

    ):
        self.distance = distance
        self.window = window
        self.itakura_max_slope = itakura_max_slope
        self.bounding_matrix = bounding_matrix
        self.epsilon = epsilon
        self.g = g
        self.lmbda = lmbda
        self.nu = nu
        self.p = p

        super(SignatureKernel, self).__init__()

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
        metric_key = self.distance
        kwargs = self.kwargs

        distmat = pairwise_distance(X, X2, metric=metric_key, **kwargs)

        return distmat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Test parameters for EditDist."""
        param_list = [{"distance": x} for x in cls.ALLOWED_DISTANCE_STR]

        return param_list
