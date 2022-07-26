"""
Information Gain-based Temporal Segmentation

Based on:

    @article{sadri2017information,
    title={Information gain-based metric for recognizing transitions in human activities},
    author={Sadri, Amin and Ren, Yongli and Salim, Flora D},
    journal={Pervasive and Mobile Computing},
    volume={38},
    pages={92--109},
    year={2017},
    publisher={Elsevier}
    }

- source code: https://github.com/cruiseresearchgroup/IGTS-python

Original docstring
------------------

Authors: Shohreh Deldari and Sam Nolan
GitHub: https://github.com/cruiseresearchgroup/IGTS-python
Matlab Implementation: https://github.com/cruiseresearchgroup/IGTS-matlab
Reference to paper : Sadri, Amin, Yongli Ren, and Flora D. Salim. "Information gain-based metric for recognizing transitions in human activities." Pervasive and Mobile Computing 38 (2017): 92-109.

Information Gain Temporal Segmentation (IGTS) is a method for segmenting
multivariate time series based off reducing the entropy in each segment.

The amount of entropy lost by the segmentations made is called the Information
Gain (IG). The aim is to find the segmentations that have the maximum information
gain for any number of segmentations.

Definitions:
 - A channel is one of the variables in a multivariate time series

Things to note:
 - The splits returned in the time series are taken to be after the kth element, not before

"""

from typing import Tuple
import numpy as np
import numpy.typing as npt


def shannon_entropy(x: npt.ArrayLike) -> npt.ArrayLike:
    """Compute Shannon's entropy.
    
    Parameters
    ----------
        x: array_like
            input array with values
    """
    x = x[(x != 0)]
    p = np.true_divide(x, np.sum(x))
    return -np.sum(p * np.log(p))


def information_gain_increment(X: npt.ArrayLike, prev_change_points, change_points):
    """
    Calculates the information gain incrementally, based off the information
    gain previously recorded.

    This function works by finding the segment for which the change_points splits
    in two. It then subtracts the information gain for the whole segment and
    adds the IG for the split segments. This operation has complexity O(

    Parameters
    ----------

        X: 
            The integral (cumulative sum) of the time series to
            calculate the information gain on.
            Represents as a numpy array of shape (number of series, time)
        prev_change_points:
            The positions on X that have IG_old IG
            Represented as a numpy array of integer positions
        change_points: 
            The new position to add to the X, integer

    Returns
    -------
        ig: float
            Information gain of X over prev_change_points and change_points splits

     Notes:

         old_entropy:
             This operation here is meant to get the sum of all the elements between
             lower and higher, not inclusive of higher. If lower is the very
             beginning of the time series (represented as lower = -1), then the
             cumulative sum at higher is equal to the sum between the start and higher
    """

    n_samples, _ = X.shape

    # The positions of the segment boundaries higher and lower than the change_points
    lower = max([-1] + [x for x in prev_change_points if x <= change_points])
    higher = min([n_samples - 1] + [x for x in prev_change_points if x >= change_points])

    if lower == -1:
        old_entropy = shannon_entropy(X[higher, :])
        new_entropy_left = shannon_entropy(X[change_points, :])
    else:
        old_entropy = shannon_entropy(X[higher, :] - X[lower, :])
        new_entropy_left = shannon_entropy(X[change_points, :] - X[lower, :])

    new_entropy_right = shannon_entropy(X[higher, :] - X[change_points, :])

    # Then we calculate the change in weighted entropy
    weighted_right = (higher - change_points) * new_entropy_right / n_samples
    weighted_old = (higher - lower) * old_entropy / n_samples
    weighted_left = (change_points - lower) * new_entropy_left / n_samples

    return weighted_old - weighted_left - weighted_right


class IGTSTopDown:
    """
    Top down IGTS. This method of IGTS tries to greedily find the next segment
    location that creates the maximum information gain. Once this is found, it
    repeats the process until we have k splits in the time series.

    Parameters
    ----------
        k: int
            The amount of splits to find in the time series. Which makes the
            amount of segments equal to k + 1(int)
        step: int
            The size of the steps to make when searching through the time
            series to find the heighest value. For instance, a step of 5
            will find the max IG out of 0...5...10 etc
    """

    def __init__(self, k: int, step: int):
        self.k = k
        self.step = step

    @property
    def n_segments(self):
        return self.k + 1

    def find_change_points(self, X: npt.ArrayLike) -> Tuple[npt.ArrayLike]:
        """
        Parameters
        ---------
            X: array_like
                Numpy array of dimensions (time, channels).

        Returns
        -------
            (splits, InformationGain, knee)
                splits is a numpy array of integers of size <= k
                The amount of splits (that I will call n) can be smaller than or
                equal to k. It is smaller than k when creating any more segments
                does not increase information gain. This usually (by experience)
                does not occur when the amount of splits is larger than 50% of
                the time series.

                splits represents the positions of splits that are found to be optimal in
                the time series. These splits are after the position they index,
                for instance, if there is a 2 in the array, then {0,1,2} is one
                segment and {3,4,5...} is another. The order of this array is
                important, and is not sorted. The first element is the split that
                was found to have the highest information gain, and the second has
                the second heighest, etc.

                Information Gain is an numpy array of floats of size n. The ith element of the
                arary represents the information gain found for the first i segments.

                knee is a number <= n that is the knee point of the time series.
                Choosing a balance between
                number of segments (usually creating a larger amount of information
                gain) vs the size of the segments (too many segments can make them
                very small). The knee point of the information gain vs segments
                curve is returned to knee.
        """

        n_samples, nseries = X.shape
        # maxTT is the segments found for the maximum IG found so far
        maxTT = np.zeros(self.n_segments, dtype=int)

        # tryTT is the working segments, that we will be trying
        tryTT = np.zeros(self.n_segments, dtype=int)

        # IG_arr is the information gain found for k
        IG_arr = np.zeros(self.n_segments, dtype=float)

        max_ig = 0

        # Segments k times
        for i in range(self.k):

            # Try for a segment in j
            for j in range(0, n_samples, self.step):

                # Add a new segment at point j
                tryTT[i + 1] = n_samples - 1
                tryTT[i] = j

                # Does an incremental IG calculation. The incremental version of
                # this function performs much better for larger k
                IG = IG_arr[i] + information_gain_increment(X, tryTT[:i], tryTT[i])
                if IG > max_ig:
                    # Record
                    maxTT = tryTT.copy()
                    max_ig = IG

            # If we did not make any progress from the information gain we already had
            if max_ig == IG_arr[i]:
                # We didn't get any information gain from this, so we should not continue
                break

            tryTT = maxTT.copy()

            IG_arr[i + 1] = max_ig

        # p_values are the second derivative of the curve at all points. Used to
        # determine the mean
        p_values = np.diff(IG_arr[:-1]) / np.diff(IG_arr[1:])
        # +1 to account for the fact that p_values are calculated starting from 1 index
        knee = np.argmax(p_values) + 1
        return tryTT, IG_arr, knee


class IGTSDynamicProgramming:
    """
    Dynamic Programming IGTS. This method of IGTS tries find the segment boundary
    locations that create the maximum information gain using Dynamic Programming.

    Parameters
    ----------
        k: int
            The number of change points to find in the time series.
        step: int
            The size of the steps to make when searching through the time
            series to find the highest value. For instance, a step of 5
            will find the max IG out of 0...5...10 etc
    """

    def __init__(self, k: int, step: int):
        self.k = k
        self.step = step

    @property
    def n_segments(self):
        return self.k + 1

    def find_change_points(self, X: npt.ArrayLike):
        """
        Parameters
        ----------
            X: a numpy array of dimensions (time, channels).

        Returns
        -------
            (expTT, InformationGain)
                expTT is a numpy array of integers of size = k

                expTT represents the positions of splits that are found to be optimal in
                the time series. These splits are after the position they index,
                for instance, if there is a 2 in the array, then {0,1,2} is one
                segment and {3,4,5...} is another. The order of this array is
                important, and is not sorted. The first element is the split that
                was found to have the highest information gain, and the second has
                the second highest, etc.

                Information Gain is floats represents the information gain of the whole time series
                regards to estimated segment boundaries.

        Warning
        -------
            This will most likely require large available memory since it'll
            try to allocate a full 3D array with the `(n_samples, n_samples, k + 1)` size.
        """

        n_samples, nseries = X.shape

        cost = np.zeros((n_samples, n_samples, self.n_segments), dtype=float)
        distances = np.zeros(nseries, dtype=float)
        pos = np.zeros((n_samples, self.n_segments), dtype=int)
        expTT = np.zeros(self.k, dtype=int)

        for i in range(0, n_samples, self.step):
            for j in range(i + 1, n_samples, self.step):
                distances = X[j, :] - X[i, :]
                cost[i : i + self.step : 1, j : j + self.step : 1, 0] = (
                    (j - i) / n_samples
                ) * shannon_entropy(distances)

        for b in range(1, self.n_segments):
            for i in range(1, n_samples):
                cost[0, i, b] = cost[0, i, b - 1].copy()
                pos[i, b] = 1
                for j in range(self.step, i - 1, self.step):
                    if cost[0, j, b - 1] + cost[j + 1, i, 0] <= cost[0, i, b]:
                        cost[0, i, b] = cost[0, j, b - 1] + cost[j + 1, i, 0]
                        pos[i, b] = j

        maxVAR = cost[0, n_samples - 1, self.k].copy()

        idx = n_samples - 1
        for b in range(self.k, 0, -1):
            expTT[b - 1] = pos[idx, b].copy()
            idx = expTT[b - 1].copy()

        return expTT, maxVAR
