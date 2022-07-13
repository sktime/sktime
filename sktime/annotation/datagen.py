# objective
# make Mirae's code good


# data generator base class in order to allow composition
from typing import List, Tuple, Union
import numpy as np
from sktime.base import BaseObject



class BaseDataGenerator(BaseObject):

    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return NotImplementedError


def a(x):
    return a(x) + 1


class GenBasicGauss(BaseDataGenerator):

    def __init__(
        self,
        centers,
        seg_lengths,
        sds=None,
        sample_funcs=None,
        random_seed=None
    ):
        self.centers = centers
        self.seg_lengths = seg_lengths
        self.sds = sds
        self.sample_funcs = sample_funcs

        assert len(centers) == len(seg_lengths)
        self._sds = sds or [1 for _ in centers]
        assert len(centers) == len(self._sds)

        if not random_seed:
            self.rng = np.random.default_rng()
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(random_seed)
        else:
            self.rng = random_seed

    def sample(self):
        centers = self.centers
        seg_lengths = self.seg_lengths
        sds = self._sds
        rng = self.rng
        sample_funcs = self.sample_funcs

        res = gen_basic_gauss(
            centers=centers, seg_lengths=seg_lengths, sds=sds, rng=rng
        )
        return res


# we would like to have a function that generates synthetic time series
# step wise function - mean shift time series 
# input either means, or pdfs

# what if we could pass data_gen functions - rather than assuming means? 
#   - could make it easier to swap out different mean assumptions
# optionally level of noise should eb adjustable


def gen_basic_gauss(
    centers:list, seg_lengths:list, sds: Union[list, float] = 1., rng=None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate simple gaussian data
    
    Parameters
    ----------
    centers :
    seg_lengths :
    sds : standard deviation

    Returns
    -------
    data : univariate time series as np.array
    labels : integer encoded vector of labels, same length as data
    """
    if not rng:
        rng = np.random.default_rng()
    assert len(centers) == len(seg_lengths)
    
    gauss_data = np.zeros([sum(seg_lengths)])
    labels = np.zeros([sum(seg_lengths)])
    
    if isinstance(sds, float):
        sds = np.repeat(sds, len(centers))
    
    assert(len(sds)==len(centers))

    seg_data = [
        rng.normal(loc=center, scale=sd, size=[length]) for center, length, sd in zip(centers, seg_lengths, sds)
    ]
    gauss_data = np.concatenate(tuple(seg_data))

    num_segs = len(seg_lengths)
    labels = np.repeat(range(num_segs), seg_lengths)
    
    return gauss_data, labels


# import numpy as np

# # this data has no autocorrelation concerns
# # it just creates an (N x 6) data array,
# # where the first half of the data exists in 1D,
# # and the second half of the data exists in 3D
# # need to add noise (and parametarise)

# # generate dataset
# N=10
# np.random.seed(10002)
# X=np.zeros((N,6))

# # half the points from one generating regime
# for j in range(1):
# 	X[:N,j]= np.random.normal(0,3,N/2)

# # the other half from another
# for j in range(3):
# 	X[N:,j]= np.random.normal(2,1,N/2)