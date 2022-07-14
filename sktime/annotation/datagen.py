from typing import List, Tuple, Union
import numpy as np
from sktime.base import BaseObject
from sklearn.utils.validation import check_random_state
import numpy.typing as npt


class BaseDataGenerator(BaseObject):
    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return NotImplementedError


class GenBasicGauss(BaseDataGenerator):
    """Data generator base class in order to allow composition"""

    def __init__(
        self, means, seg_lengths, sds=None, sample_funcs=None, random_seed=None
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

        return gen_basic_gauss(
            centers=centers, seg_lengths=seg_lengths, sds=sds, rng=rng
        )


# we would like to have a function that generates synthetic time series
# step wise function - mean shift time series
# input either means, or pdfs

# what if we could pass data_gen functions - rather than assuming means?
#   - could make it easier to swap out different mean assumptions
# optionally level of noise should eb adjustable


def labels_with_repeats(means, noises):
    """
    Based
    """
    data = [means, noises]
    unique, indices = np.unique(data, axis=1, return_inverse=True)
    labels = np.arange(unique.shape[1])
    return labels[indices]


def mean_shift(
    means: npt.ArrayLike,
    lengths: npt.ArrayLike,
    sds: Union[npt.ArrayLike, float] = 1.0,
    repeated_labels: bool = True,
    random_state: Union[int, np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a series composed of segments, where each segment has length specified
    in ``lengths`` and data sampled from a normal distribution with a mean from ``means``
    and standard deviation from ``sds``.

    Parameters
    ----------
    means : array_like
        Means of the segments to be generated 
    lengths : array_like
        Lengths of the segments to be generated 
    sds : float ir array_like
        Standard deviations fo the segments to be generated
    repeated_labels : bool
        A flag to indicate whether segment labels should be repeated for similar segments.
        If ``True`` same label will be assigned for segments with same mean and noise,
        independently of length. If ``False`` each consecutive segment will have a unique
        label.
    random_state : int or np.random.RandomState
        Either a random seed or RandomState instance

    Returns
    -------
    data : np.array
        univariate time series as np.array
    labels : np.array
        integer encoded array of labels, same length as data
    """

    rng = check_random_state(random_state)
    assert len(means) == len(lengths)

    if isinstance(sds, float):
        sds = np.repeat(sds, len(means))

    assert len(sds) == len(means)

    segments_data = [
        rng.normal(loc=mean, scale=sd, size=[length])
        for mean, length, sd in zip(means, lengths, sds)
    ]
    series = np.concatenate(tuple(segments_data))

    if repeated_labels:
        unique_labels = labels_with_repeats(means, sds)
    else:
        unique_labels = range(len(lengths))
    labels = np.repeat(unique_labels, lengths)

    return series, labels


# this data has no autocorrelation concerns
# it just creates an (N x 6) data array,
# where the first half of the data exists in 1D,
# and the second half of the data exists in 3D
# need to add noise (and parametarise)

# generate dataset
# N=10
# np.random.seed(10002)
# X=np.zeros((N,6))

# # half the points from one generating regime
# for j in range(1):
# 	X[:N,j]= np.random.normal(0,3,N/2)

# # the other half from another
# for j in range(3):
# 	X[N:,j]= np.random.normal(2,1,N/2)
