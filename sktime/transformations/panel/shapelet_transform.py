# -*- coding: utf-8 -*-
"""Shapelet transformers.

A transformer from the time domain into the shapelet domain.
"""

__author__ = ["MatthewMiddlehurst", "jasonlines", "dguijo"]
__all__ = ["ShapeletTransform", "RandomShapeletTransform"]

import heapq
import math
import time
import warnings
from itertools import zip_longest
from operator import itemgetter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit
from numba.typed.typedlist import List
from sklearn import preprocessing
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.transformations.base import BaseTransformer
from sktime.utils.numba.general import z_normalise_series
from sktime.utils.validation import check_n_jobs


class Shapelet:
    """A simple class to model a Shapelet with associated information.

    Parameters
    ----------
    series_id: int
        The index of the series within the data (X) that was passed to fit.
    start_pos: int
        The starting position of the shapelet within the original series
    length: int
        The length of the shapelet
    info_gain: flaot
        The calculated information gain of this shapelet
    data: array-like
        The (z-normalised) data of this shapelet.
    """

    def __init__(self, series_id, start_pos, length, info_gain, data):
        self.series_id = series_id
        self.start_pos = start_pos
        self.length = length
        self.info_gain = info_gain
        self.data = data

    def __str__(self):
        """Print."""
        return (
            "Series ID: {0}, start_pos: {1}, length: {2}, info_gain: {3},"
            " ".format(self.series_id, self.start_pos, self.length, self.info_gain)
        )


class ShapeletPQ:
    """Shapelet PQ."""

    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, shapelet):
        """Push."""
        heapq.heappush(self._queue, (shapelet.info_gain, self._index, shapelet))
        self._index += 1

    def pop(self):
        """Pop."""
        return heapq.heappop(self._queue)[-1]

    def peek(self):
        """Peek."""
        return self._queue[0]

    def get_size(self):
        """Get size."""
        return len(self._queue)

    def get_array(self):
        """Get array."""
        return self._queue


class ShapeletTransform(BaseTransformer):
    """Shapelet Transform.

    Original journal publication:
    @article{hills2014classification,
      title={Classification of time series by shapelet transformation},
      author={Hills, Jon and Lines, Jason and Baranauskas, Edgaras and Mapp,
      James and Bagnall, Anthony},
      journal={Data Mining and Knowledge Discovery},
      volume={28},
      number={4},
      pages={851--881},
      year={2014},
      publisher={Springer}
    }

    Parameters
    ----------
    min_shapelet_length                 : int, lower bound on candidate
    shapelet lengths (default = 3)
    max_shapelet_length                 : int, upper bound on candidate
    shapelet lengths (default = inf or series length)
    max_shapelets_to_store_per_class    : int, upper bound on number of
    shapelets to retain from each distinct class (default = 200)
    random_state                        : RandomState, int, or none: to
    control random state objects for deterministic results (default = None)
    verbose                             : int, level of output printed to
    the console (for information only) (default = 0)
    remove_self_similar                 : boolean, remove overlapping
    "self-similar" shapelets from the final transform (default = True)

    Attributes
    ----------
    predefined_ig_rejection_level       : float, minimum information gain
    required to keep a shapelet (default = 0.05)
    self.shapelets                      : list of Shapelet objects,
    the stored shapelets after a dataset has been processed
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "numpy1D",  # and for y?
        "requires_y": True,
        "univariate-only": True,
        "fit_is_empty": False,
    }

    def __init__(
        self,
        min_shapelet_length=3,
        max_shapelet_length=np.inf,
        max_shapelets_to_store_per_class=200,
        random_state=None,
        verbose=0,
        remove_self_similar=True,
    ):

        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.max_shapelets_to_store_per_class = max_shapelets_to_store_per_class
        self.random_state = random_state
        self.verbose = verbose
        self.remove_self_similar = remove_self_similar
        self.predefined_ig_rejection_level = 0.05
        self.shapelets = None
        super(ShapeletTransform, self).__init__()

    def _fit(self, X, y=None):
        """Fit the shapelet transform to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame
            The training input samples.
        y: array-like or list
            The class values for X

        Returns
        -------
        self : ShapeletTransform
            This estimator
        """
        X_lens = np.repeat(X.shape[-1], X.shape[0])
        # note, assumes all dimensions of a case are the same
        # length. A shapelet would not be well defined if indices do not match!
        # may need to pad with nans here for uneq length,
        # look at later

        num_ins = len(y)
        distinct_class_vals = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        candidates_evaluated = 0

        num_series_to_visit = num_ins

        shapelet_heaps_by_class = {i: ShapeletPQ() for i in distinct_class_vals}

        self._random_state = check_random_state(self.random_state)

        # Here we establish the order of cases to sample. We need to sample
        # x cases and y shapelets from each (where x = num_cases_to_sample
        # and y = num_shapelets_to_sample_per_case). We could simply sample
        # x cases without replacement and y shapelets from each case, but
        # the idea is that if we are using a time contract we may extract
        # all y shapelets from each x candidate and still have time remaining.
        # Therefore, if we get a list of the indices of the series and
        # shuffle them appropriately, we can go through the list again and
        # extract
        # another y shapelets from each series (if we have time).

        # We also want to ensure that we visit all classes so we will visit
        # in round-robin order. Therefore, the code below extracts the indices
        # of all series by class, shuffles the indices for each class
        # independently, and then combines them in alternating order. This
        # results in
        # a shuffled list of indices that are in alternating class order (
        # e.g. 1,2,3,1,2,3,1,2,3,1...)

        def _round_robin(*iterables):
            sentinel = object()
            return (
                a
                for x in zip_longest(*iterables, fillvalue=sentinel)
                for a in x
                if a != sentinel
            )

        case_ids_by_class = {i: np.where(y == i)[0] for i in distinct_class_vals}

        num_train_per_class = {i: len(case_ids_by_class[i]) for i in case_ids_by_class}
        round_robin_case_order = _round_robin(
            *[list(v) for k, v in case_ids_by_class.items()]
        )
        cases_to_visit = [(i, y[i]) for i in round_robin_case_order]
        # this dictionary will be used to store all possible starting
        # positions and shapelet lengths for a give series length. This
        # is because we enumerate all possible candidates and sample without
        # replacement when assessing a series. If we have two series
        # of the same length then they will obviously have the same valid
        # shapelet starting positions and lengths (especially in standard
        # datasets where all series are equal length) so it makes sense to
        # store the possible candidates and reuse, rather than
        # recalculating each time

        # Initially the dictionary will be empty, and each time a new series
        # length is seen the dict will be updated. Next time that length
        # is used the dict will have an entry so can simply reuse
        possible_candidates_per_series_length = {}

        # a flag to indicate if extraction should stop (contract has ended)
        time_finished = False

        # max time calculating a shapelet
        # for timing the extraction when contracting
        start_time = time.time()

        def time_taken():
            return time.time() - start_time

        max_time_calc_shapelet = -1
        time_last_shapelet = time_taken()

        # for every series
        case_idx = 0
        while case_idx < len(cases_to_visit):

            series_id = cases_to_visit[case_idx][0]
            this_class_val = cases_to_visit[case_idx][1]

            # minus 1 to remove this candidate from sums
            binary_ig_this_class_count = num_train_per_class[this_class_val] - 1
            binary_ig_other_class_count = num_ins - binary_ig_this_class_count - 1

            if self.verbose:
                print(  # noqa
                    "visiting series: "
                    + str(series_id)
                    + " (#"
                    + str(case_idx + 1)
                    + ")"
                )

            this_series_len = len(X[series_id][0])

            # The bound on possible shapelet lengths will differ
            # series-to-series if using unequal length data.
            # However, shapelets cannot be longer than the series, so set to
            # the minimum of the series length
            # and max shapelet length (which is inf by default)
            if self.max_shapelet_length == -1:
                this_shapelet_length_upper_bound = this_series_len
            else:
                this_shapelet_length_upper_bound = min(
                    this_series_len, self.max_shapelet_length
                )

            # all possible start and lengths for shapelets within this
            # series (calculates if series length is new, a simple look-up
            # if not)
            # enumerate all possible candidate starting positions and lengths.

            # First, try to reuse if they have been calculated for a series
            # of the same length before.
            candidate_starts_and_lens = possible_candidates_per_series_length.get(
                this_series_len
            )
            # else calculate them for this series length and store for
            # possible use again
            if candidate_starts_and_lens is None:
                candidate_starts_and_lens = [
                    [start, length]
                    for start in range(
                        0, this_series_len - self.min_shapelet_length + 1
                    )
                    for length in range(
                        self.min_shapelet_length, this_shapelet_length_upper_bound + 1
                    )
                    if start + length <= this_series_len
                ]
                possible_candidates_per_series_length[
                    this_series_len
                ] = candidate_starts_and_lens

            # default for full transform
            candidates_to_visit = candidate_starts_and_lens
            num_candidates_per_case = len(candidate_starts_and_lens)

            # limit search otherwise:
            if hasattr(self, "num_candidates_to_sample_per_case"):
                num_candidates_per_case = min(
                    self.num_candidates_to_sample_per_case, num_candidates_per_case
                )
                cand_idx = list(
                    self._random_state.choice(
                        list(range(0, len(candidate_starts_and_lens))),
                        num_candidates_per_case,
                        replace=False,
                    )
                )
                candidates_to_visit = [candidate_starts_and_lens[x] for x in cand_idx]

            for candidate_idx in range(num_candidates_per_case):

                # if shapelet heap for this class is not full yet, set entry
                # criteria to be the predetermined IG threshold
                ig_cutoff = self.predefined_ig_rejection_level
                # otherwise if we have max shapelets already, set the
                # threshold as the IG of the current 'worst' shapelet we have
                if (
                    shapelet_heaps_by_class[this_class_val].get_size()
                    >= self.max_shapelets_to_store_per_class
                ):
                    ig_cutoff = max(
                        shapelet_heaps_by_class[this_class_val].peek()[0], ig_cutoff
                    )

                cand_start_pos = candidates_to_visit[candidate_idx][0]
                cand_len = candidates_to_visit[candidate_idx][1]

                candidate = ShapeletTransform.zscore(
                    X[series_id][:, cand_start_pos : cand_start_pos + cand_len]
                )

                # now go through all other series and get a distance from
                # the candidate to each
                orderline = []

                # initialise here as copy, decrease the new val each time we
                # evaluate a comparison series
                num_visited_this_class = 0
                num_visited_other_class = 0

                candidate_rejected = False

                for comparison_series_idx in range(len(cases_to_visit)):
                    i = cases_to_visit[comparison_series_idx][0]

                    if y[i] != cases_to_visit[comparison_series_idx][1]:
                        raise ValueError("class match sanity test broken")

                    if i == series_id:
                        # don't evaluate candidate against own series
                        continue

                    if y[i] == this_class_val:
                        num_visited_this_class += 1
                        binary_class_identifier = 1  # positive for this class
                    else:
                        num_visited_other_class += 1
                        binary_class_identifier = -1  # negative for any
                        # other class

                    bsf_dist = np.inf

                    start_left = cand_start_pos
                    start_right = cand_start_pos + 1

                    if X_lens[i] == cand_len:
                        start_left = 0
                        start_right = 0

                    for _ in range(
                        max(1, int(np.ceil((X_lens[i] - cand_len) / 2)))
                    ):  # max
                        # used to force iteration where series len ==
                        # candidate len
                        if start_left < 0:
                            start_left = X_lens[i] - 1 - cand_len

                        comparison = ShapeletTransform.zscore(
                            X[i][:, start_left : start_left + cand_len]
                        )
                        dist_left = np.linalg.norm(candidate - comparison)
                        bsf_dist = min(dist_left * dist_left, bsf_dist)

                        # for odd lengths
                        if start_left == start_right:
                            continue

                        # right
                        if start_right == X_lens[i] - cand_len + 1:
                            start_right = 0
                        comparison = ShapeletTransform.zscore(
                            X[i][:, start_right : start_right + cand_len]
                        )
                        dist_right = np.linalg.norm(candidate - comparison)
                        bsf_dist = min(dist_right * dist_right, bsf_dist)

                        start_left -= 1
                        start_right += 1

                    orderline.append((bsf_dist, binary_class_identifier))
                    # sorting required after each add for early IG abandon.
                    # timsort should be efficient as array is almost in
                    # order - insertion-sort like behaviour in this case.
                    # Can't use heap as need to traverse in order multiple
                    # times, not just access root
                    orderline.sort()

                    if len(orderline) > 2:
                        ig_upper_bound = ShapeletTransform.calc_early_binary_ig(
                            orderline,
                            num_visited_this_class,
                            num_visited_other_class,
                            binary_ig_this_class_count - num_visited_this_class,
                            binary_ig_other_class_count - num_visited_other_class,
                        )
                        # print("upper: "+str(ig_upper_bound))
                        if ig_upper_bound <= ig_cutoff:
                            candidate_rejected = True
                            break

                candidates_evaluated += 1
                if self.verbose > 3 and candidates_evaluated % 100 == 0:
                    print("candidates evaluated: " + str(candidates_evaluated))  # noqa

                # only do if candidate was not rejected
                if candidate_rejected is False:
                    final_ig = ShapeletTransform.calc_binary_ig(
                        orderline,
                        binary_ig_this_class_count,
                        binary_ig_other_class_count,
                    )
                    accepted_candidate = Shapelet(
                        series_id, cand_start_pos, cand_len, final_ig, candidate
                    )

                    # add to min heap to store shapelets for this class
                    shapelet_heaps_by_class[this_class_val].push(accepted_candidate)

                    # informal, but extra 10% allowance for self similar later
                    if (
                        shapelet_heaps_by_class[this_class_val].get_size()
                        > self.max_shapelets_to_store_per_class * 3
                    ):
                        shapelet_heaps_by_class[this_class_val].pop()

                # Takes into account the use of the MAX shapelet calculation
                # time to not exceed the time_limit (not exact, but likely a
                # good guess).
                if (
                    hasattr(self, "time_contract_in_mins")
                    and self.time_contract_in_mins > 0
                ):
                    time_now = time_taken()
                    time_this_shapelet = time_now - time_last_shapelet
                    if time_this_shapelet > max_time_calc_shapelet:
                        max_time_calc_shapelet = time_this_shapelet
                        if self.verbose > 0:
                            print(max_time_calc_shapelet)  # noqa
                    time_last_shapelet = time_now

                    # add a little 1% leeway to the timing incase one run was
                    # slightly faster than
                    # another based on the CPU.
                    time_in_seconds = self.time_contract_in_mins * 60
                    max_shapelet_time_percentage = (
                        max_time_calc_shapelet / 100.0
                    ) * 0.75
                    if (time_now + max_shapelet_time_percentage) > time_in_seconds:
                        if self.verbose > 0:
                            print(  # noqa
                                "No more time available! It's been {0:02d}:{"
                                "1:02}".format(
                                    int(round(time_now / 60, 3)),
                                    int(
                                        (
                                            round(time_now / 60, 3)
                                            - int(round(time_now / 60, 3))
                                        )
                                        * 60
                                    ),
                                )
                            )
                        time_finished = True
                        break
                    else:
                        if self.verbose > 0:
                            if candidate_rejected is False:
                                print(  # noqa
                                    "Candidate finished. {0:02d}:{1:02} "
                                    "remaining".format(
                                        int(
                                            round(
                                                self.time_contract_in_mins
                                                - time_now / 60,
                                                3,
                                            )
                                        ),
                                        int(
                                            (
                                                round(
                                                    self.time_contract_in_mins
                                                    - time_now / 60,
                                                    3,
                                                )
                                                - int(
                                                    round(
                                                        (
                                                            self.time_contract_in_mins
                                                            - time_now
                                                        )
                                                        / 60,
                                                        3,
                                                    )
                                                )
                                            )
                                            * 60
                                        ),
                                    )
                                )
                            else:
                                print(  # noqa
                                    "Candidate rejected. {0:02d}:{1:02} "
                                    "remaining".format(
                                        int(
                                            round(
                                                (self.time_contract_in_mins - time_now)
                                                / 60,
                                                3,
                                            )
                                        ),
                                        int(
                                            (
                                                round(
                                                    (
                                                        self.time_contract_in_mins
                                                        - time_now
                                                    )
                                                    / 60,
                                                    3,
                                                )
                                                - int(
                                                    round(
                                                        (
                                                            self.time_contract_in_mins
                                                            - time_now
                                                        )
                                                        / 60,
                                                        3,
                                                    )
                                                )
                                            )
                                            * 60
                                        ),
                                    )
                                )

            # stopping condition: in case of iterative transform (i.e.
            # num_cases_to_sample have been visited)
            #                     in case of contracted transform (i.e. time
            #                     limit has been reached)
            case_idx += 1

            if case_idx >= num_series_to_visit:
                if hasattr(self, "time_contract_in_mins") and time_finished is not True:
                    case_idx = 0
            elif case_idx >= num_series_to_visit or time_finished:
                if self.verbose > 0:
                    print("Stopping search")  # noqa
                break

        # remove self similar here
        # for each class value
        #       get list of shapelets
        #       sort by quality
        #       remove self similar

        self.shapelets = []
        for class_val in distinct_class_vals:
            by_class_descending_ig = sorted(
                shapelet_heaps_by_class[class_val].get_array(),
                key=itemgetter(0),
                reverse=True,
            )

            if self.remove_self_similar and len(by_class_descending_ig) > 0:
                by_class_descending_ig = (
                    ShapeletTransform.remove_self_similar_shapelets(
                        by_class_descending_ig
                    )
                )
            else:
                # need to extract shapelets from tuples
                by_class_descending_ig = [x[2] for x in by_class_descending_ig]

            # if we have more than max_shapelet_per_class, trim to that
            # amount here
            if len(by_class_descending_ig) > self.max_shapelets_to_store_per_class:
                max_n = self.max_shapelets_to_store_per_class
                by_class_descending_ig = by_class_descending_ig[:max_n]

            self.shapelets.extend(by_class_descending_ig)

        # final sort so that all shapelets from all classes are in
        # descending order of information gain
        self.shapelets.sort(key=lambda x: x.info_gain, reverse=True)

        # warn the user if fit did not produce any valid shapelets
        if len(self.shapelets) == 0:
            warnings.warn(
                "No valid shapelets were extracted from this dataset and "
                "calling the transform method "
                "will raise an Exception. Please re-fit the transform with "
                "other data and/or "
                "parameter options."
            )

        return self

    @staticmethod
    def remove_self_similar_shapelets(shapelet_list):
        """Remove self-similar shapelets from an input list.

        Note: this method assumes that shapelets are pre-sorted in descending order
        of quality (i.e. if two candidates are self-similar, the one with the later
        index will be removed)

        Parameters
        ----------
        shapelet_list: list of Shapelet objects

        Returns
        -------
        shapelet_list: list of Shapelet objects
        """
        # IMPORTANT: it is assumed that shapelets are already in descending
        # order of quality. This is preferable in the fit method as removing
        # self-similar
        # shapelets may be False so the sort needs to happen there in those
        # cases, and avoids a second redundant sort here if it is set to True
        def is_self_similar(shapelet_one, shapelet_two):
            # not self similar if from different series
            if shapelet_one.series_id != shapelet_two.series_id:
                return False

            if (shapelet_one.start_pos >= shapelet_two.start_pos) and (
                shapelet_one.start_pos <= shapelet_two.start_pos + shapelet_two.length
            ):
                return True
            if (shapelet_two.start_pos >= shapelet_one.start_pos) and (
                shapelet_two.start_pos <= shapelet_one.start_pos + shapelet_one.length
            ):
                return True

        # [s][2] will be a tuple with (info_gain,id,Shapelet), so we need to
        # access [2]
        to_return = [shapelet_list[0][2]]  # first shapelet must be ok
        for s in range(1, len(shapelet_list)):
            can_add = True
            for c in range(0, s):
                if is_self_similar(shapelet_list[s][2], shapelet_list[c][2]):
                    can_add = False
                    break
            if can_add:
                to_return.append(shapelet_list[s][2])

        return to_return

    # transform a set of data into distances to each extracted shapelet
    def _transform(self, X, y=None):
        """Transform X according to the extracted shapelets (self.shapelets).

        Parameters
        ----------
        X : pandas DataFrame
            The input dataframe to transform

        Returns
        -------
        output : pandas DataFrame
            The transformed dataframe in tabular format.
        """
        if len(self.shapelets) == 0:
            raise RuntimeError(
                "No shapelets were extracted in fit that exceeded the "
                "minimum information gain threshold. Please retry with other "
                "data and/or parameter settings."
            )

        # may need to pad with nans here for uneq length, look at later
        output = np.zeros(
            [len(X), len(self.shapelets)],
            dtype=np.float32,
        )

        # for the i^th series to transform
        for i in range(0, len(X)):
            this_series = X[i]

            # get the s^th shapelet
            for s in range(0, len(self.shapelets)):
                # find distance between this series and each shapelet
                min_dist = np.inf
                this_shapelet_length = self.shapelets[s].length

                for start_pos in range(
                    0, len(this_series[0]) - this_shapelet_length + 1
                ):
                    comparison = ShapeletTransform.zscore(
                        this_series[:, start_pos : (start_pos + this_shapelet_length)]
                    )

                    dist = np.linalg.norm(self.shapelets[s].data - comparison)
                    dist = dist * dist
                    dist = 1.0 / this_shapelet_length * dist
                    min_dist = min(min_dist, dist)

                    output[i][s] = min_dist

        return pd.DataFrame(output)

    def get_shapelets(self):
        """Accessor method to return the extracted shapelets.

        Returns
        -------
        shapelets: a list of Shapelet objects
        """
        return self.shapelets

    @staticmethod
    def binary_entropy(num_this_class, num_other_class):
        """Binary entropy."""
        ent = 0
        if num_this_class != 0:
            ent -= (
                num_this_class
                / (num_this_class + num_other_class)
                * np.log2(num_this_class / (num_this_class + num_other_class))
            )
        if num_other_class != 0:
            ent -= (
                num_other_class
                / (num_this_class + num_other_class)
                * np.log2(num_other_class / (num_this_class + num_other_class))
            )
        return ent

    # could cythonise
    @staticmethod
    def calc_binary_ig(orderline, total_num_this_class, total_num_other_class):
        """Binary information gain."""
        # def entropy(ent_class_counts, all_class_count):

        initial_ent = ShapeletTransform.binary_entropy(
            total_num_this_class, total_num_other_class
        )
        bsf_ig = 0

        count_this_class = 0
        count_other_class = 0

        total_all = total_num_this_class + total_num_other_class

        # evaluate each split point
        for split in range(0, len(orderline) - 1):
            next_class = orderline[split][1]  # +1 if this class, -1 if other
            if next_class > 0:
                count_this_class += 1
            else:
                count_other_class += 1

            # optimistically add this class to left side first and other to
            # right
            left_prop = (split + 1) / total_all
            ent_left = ShapeletTransform.binary_entropy(
                count_this_class, count_other_class
            )

            right_prop = 1 - left_prop  # because right side must
            # optimistically contain everything else
            ent_right = ShapeletTransform.binary_entropy(
                total_num_this_class - count_this_class,
                total_num_other_class - count_other_class,
            )

            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

        return bsf_ig

    # could cythonise
    @staticmethod
    def calc_early_binary_ig(
        orderline,
        num_this_class_in_orderline,
        num_other_class_in_orderline,
        num_to_add_this_class,
        num_to_add_other_class,
    ):
        """Early binary IG."""
        # def entropy(ent_class_counts, all_class_count):

        initial_ent = ShapeletTransform.binary_entropy(
            num_this_class_in_orderline + num_to_add_this_class,
            num_other_class_in_orderline + num_to_add_other_class,
        )
        bsf_ig = 0

        # actual observations in orderline
        count_this_class = 0
        count_other_class = 0

        total_all = (
            num_this_class_in_orderline
            + num_other_class_in_orderline
            + num_to_add_this_class
            + num_to_add_other_class
        )

        # evaluate each split point
        for split in range(0, len(orderline) - 1):
            next_class = orderline[split][1]  # +1 if this class, -1 if other
            if next_class > 0:
                count_this_class += 1
            else:
                count_other_class += 1

            # optimistically add this class to left side first and other to
            # right
            left_prop = (split + 1 + num_to_add_this_class) / total_all
            ent_left = ShapeletTransform.binary_entropy(
                count_this_class + num_to_add_this_class, count_other_class
            )

            right_prop = 1 - left_prop  # because right side must
            # optimistically contain everything else
            ent_right = ShapeletTransform.binary_entropy(
                num_this_class_in_orderline - count_this_class,
                num_other_class_in_orderline
                - count_other_class
                + num_to_add_other_class,
            )

            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

            # now optimistically add this class to right, other to left
            left_prop = (split + 1 + num_to_add_other_class) / total_all
            ent_left = ShapeletTransform.binary_entropy(
                count_this_class, count_other_class + num_to_add_other_class
            )

            right_prop = 1 - left_prop  # because right side must
            # optimistically contain everything else
            ent_right = ShapeletTransform.binary_entropy(
                num_this_class_in_orderline - count_this_class + num_to_add_this_class,
                num_other_class_in_orderline - count_other_class,
            )
            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

        return bsf_ig

    @staticmethod
    def zscore(a, axis=0, ddof=0):
        """Return the normalised version of series.

        This mirrors the scipy implementation
        with a small difference - rather than allowing /0, the function
        returns output = np.zeroes(len(input)).
        This is to allow for sensible processing of candidate
        shapelets/comparison subseries that are a straight
        line. Original version:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats
        .zscore.html

        Parameters
        ----------
        a : array_like
            An array like object containing the sample data.

        axis : int or None, optional
            Axis along which to operate. Default is 0. If None, compute over
            the whole array a.

        ddof : int, optional
            Degrees of freedom correction in the calculation of the standard
            deviation. Default is 0.

        Returns
        -------
        zscore : array_like
            The z-scores, standardized by mean and standard deviation of
            input array a.
        """
        zscored = np.empty(a.shape)
        for i, j in enumerate(a):
            # j = np.asanyarray(j)
            sstd = j.std(axis=axis, ddof=ddof)

            # special case - if shapelet is a straight line (i.e. no
            # variance), zscore ver should be np.zeros(len(a))
            if sstd == 0:
                zscored[i] = np.zeros(len(j))
            else:
                mns = j.mean(axis=axis)
                if axis and mns.ndim < j.ndim:
                    zscored[i] = (j - np.expand_dims(mns, axis=axis)) / np.expand_dims(
                        sstd, axis=axis
                    )
                else:
                    zscored[i] = (j - mns) / sstd
        return zscored

    @staticmethod
    def euclidean_distance_early_abandon(u, v, min_dist):
        """Euclidean distance with early abandon."""
        sum_dist = 0
        for i in range(0, len(u[0])):
            for j in range(0, len(u)):
                u_v = u[j][i] - v[j][i]
                sum_dist += np.dot(u_v, u_v)
                if sum_dist >= min_dist:
                    # The distance is higher, so early abandon.
                    return min_dist
        return sum_dist


class RandomShapeletTransform(BaseTransformer):
    """Random Shapelet Transform.

    Implementation of the binary shapelet transform along the lines of [1]_[2]_, with
    randomly extracted shapelets.

    Overview: Input "n" series with "d" dimensions of length "m". Continuously extract
    candidate shapelets and filter them in batches.
        For each candidate shapelet
            - Extract a shapelet from an instance with random length, position and
              dimension
            - Using its distance to train cases, calculate the shapelets information
              gain
            - Abandon evaluating the shapelet if it is impossible to obtain a higher
              information gain than the current worst
        For each shapelet batch
            - Add each candidate to its classes shapelet heap, removing the lowest
              information gain shapelet if the max number of shapelets has been met
            - Remove self-similar shapelets from the heap
    Using the final set of filtered shapelets, transform the data into a vector of
    of distances from a series to each shapelet.

    Parameters
    ----------
    n_shapelet_samples : int, default=10000
        The number of candidate shapelets to be considered for the final transform.
        Filtered down to <= max_shapelets, keeping the shapelets with the most
        information gain.
    max_shapelets : int or None, default=None
        Max number of shapelets to keep for the final transform. Each class value will
        have its own max, set to n_classes / max_shapelets. If None uses the min between
        10 * n_instances and 1000
    min_shapelet_length : int, default=3
        Lower bound on candidate shapelet lengths.
    max_shapelet_length : int or None, default= None
        Upper bound on candidate shapelet lengths. If None no max length is used.
    remove_self_similar : boolean, default=True
        Remove overlapping "self-similar" shapelets when merging candidate shapelets.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_shapelet_samples.
        Default of 0 means n_shapelet_samples is used.
    contract_max_n_shapelet_samples : int, default=np.inf
        Max number of shapelets to extract when time_limit_in_minutes is set.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform`.
        ``-1`` means using all processors.
    parallel_backend : str, ParallelBackendBase instance or None, default=None
        Specify the parallelisation backend implementation in joblib, if None a 'prefer'
        value of "threads" is used by default.
        Valid options are "loky", "multiprocessing", "threading" or a custom backend.
        See the joblib Parallel documentation for more details.
    batch_size : int or None, default=100
        Number of shapelet candidates processed before being merged into the set of best
        shapelets.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    n_instances : int
        The number of train cases.
    n_dims : int
        The number of dimensions per case.
    series_length : int
        The length of each series.
    classes_ : list
        The classes labels.
    shapelets : list
        The stored shapelets and relating information after a dataset has been
        processed.
        Each item in the list is a tuple containing the following 7 items:
        (shapelet information gain, shapelet length, start position the shapelet was
        extracted from, shapelet dimension, index of the instance the shapelet was
        extracted from in fit, class value of the shapelet, The z-normalised shapelet
        array)

    See Also
    --------
    ShapeletTransformClassifier

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/transformers/ShapeletTransform.java>`_.

    References
    ----------
    .. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
       Data Mining and Knowledge Discovery, 28(4), 851--881, 2014.
    .. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
       Series Classification", Transactions on Large-Scale Data and Knowledge Centered
       Systems, 32, 2017.

    Examples
    --------
    >>> from sktime.transformations.panel.shapelet_transform import (
    ...     RandomShapeletTransform
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> t = RandomShapeletTransform(
    ...     n_shapelet_samples=500,
    ...     max_shapelets=10,
    ...     batch_size=100,
    ... )
    >>> t.fit(X_train, y_train)
    RandomShapeletTransform(...)
    >>> X_t = t.transform(X_train)
    """

    _tags = {
        "fit_is_empty": False,
        "univariate-only": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "numpy1D",  # and for y?
        "requires_y": True,
    }

    def __init__(
        self,
        n_shapelet_samples=10000,
        max_shapelets=None,
        min_shapelet_length=3,
        max_shapelet_length=None,
        remove_self_similar=True,
        time_limit_in_minutes=0.0,
        contract_max_n_shapelet_samples=np.inf,
        n_jobs=1,
        parallel_backend=None,
        batch_size=100,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.remove_self_similar = remove_self_similar

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples

        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.batch_size = batch_size
        self.random_state = random_state

        # The following set in method fit
        self.n_classes = 0
        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.classes_ = []
        self.shapelets = []

        self._n_shapelet_samples = n_shapelet_samples
        self._max_shapelets = max_shapelets
        self._max_shapelet_length = max_shapelet_length
        self._n_jobs = n_jobs
        self._batch_size = batch_size
        self._class_counts = []
        self._class_dictionary = {}
        self._sorted_indicies = []

        super(RandomShapeletTransform, self).__init__()

    def _fit(self, X, y=None):
        """Fit the shapelet transform to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : RandomShapeletTransform
            This estimator.
        """
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.classes_, self._class_counts = np.unique(y, return_counts=True)
        self.n_classes = self.classes_.shape[0]
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        self.n_instances, self.n_dims, self.series_length = X.shape

        if self.max_shapelets is None:
            self._max_shapelets = min(10 * self.n_instances, 1000)
        if self._max_shapelets < self.n_classes:
            self._max_shapelets = self.n_classes
        if self.max_shapelet_length is None:
            self._max_shapelet_length = self.series_length

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        fit_time = 0

        max_shapelets_per_class = int(self._max_shapelets / self.n_classes)
        if max_shapelets_per_class < 1:
            max_shapelets_per_class = 1

        shapelets = List(
            [List([(-1.0, -1, -1, -1, -1, -1)]) for _ in range(self.n_classes)]
        )
        n_shapelets_extracted = 0

        if time_limit > 0:
            while (
                fit_time < time_limit
                and n_shapelets_extracted < self.contract_max_n_shapelet_samples
            ):
                candidate_shapelets = Parallel(
                    n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
                )(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                        shapelets,
                        max_shapelets_per_class,
                    )
                    for i in range(self._batch_size)
                )

                for i, heap in enumerate(shapelets):
                    self._merge_shapelets(
                        heap,
                        List(candidate_shapelets),
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = self._remove_self_similar_shapelets(heap)
                        shapelets[i] = List([n for (n, b) in zip(heap, to_keep) if b])

                n_shapelets_extracted += self._batch_size
                fit_time = time.time() - start_time
        else:
            while n_shapelets_extracted < self._n_shapelet_samples:
                n_shapelets_to_extract = (
                    self._batch_size
                    if n_shapelets_extracted + self._batch_size
                    <= self._n_shapelet_samples
                    else self._n_shapelet_samples - n_shapelets_extracted
                )

                candidate_shapelets = Parallel(
                    n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
                )(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                        shapelets,
                        max_shapelets_per_class,
                    )
                    for i in range(n_shapelets_to_extract)
                )

                for i, heap in enumerate(shapelets):
                    self._merge_shapelets(
                        heap,
                        List(candidate_shapelets),
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = self._remove_self_similar_shapelets(heap)
                        shapelets[i] = List([n for (n, b) in zip(heap, to_keep) if b])

                n_shapelets_extracted += n_shapelets_to_extract

        self.shapelets = [
            (
                s[0],
                s[1],
                s[2],
                s[3],
                s[4],
                self.classes_[s[5]],
                z_normalise_series(X[s[4], s[3], s[2] : s[2] + s[1]]),
            )
            for class_shapelets in shapelets
            for s in class_shapelets
            if s[0] > 0
        ]
        self.shapelets.sort(reverse=True, key=lambda s: (s[0], s[1], s[2], s[3], s[4]))

        to_keep = self._remove_identical_shapelets(List(self.shapelets))
        self.shapelets = [n for (n, b) in zip(self.shapelets, to_keep) if b]

        self._sorted_indicies = []
        for s in self.shapelets:
            sabs = np.abs(s[6])
            self._sorted_indicies.append(
                np.array(
                    sorted(range(s[1]), reverse=True, key=lambda j, sabs=sabs: sabs[j])
                )
            )
        return self

    def _transform(self, X, y=None):
        """Transform X according to the extracted shapelets.

        Parameters
        ----------
        X : pandas DataFrame or np.ndarray
            The input data to transform.

        Returns
        -------
        output : pandas DataFrame
            The transformed dataframe in tabular format.
        """
        output = np.zeros((len(X), len(self.shapelets)))

        for i, series in enumerate(X):
            dists = Parallel(
                n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
            )(
                delayed(_online_shapelet_distance)(
                    series[shapelet[3]],
                    shapelet[6],
                    self._sorted_indicies[n],
                    shapelet[2],
                    shapelet[1],
                )
                for n, shapelet in enumerate(self.shapelets)
            )

            output[i] = dists

        return pd.DataFrame(output)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {"max_shapelets": 5, "n_shapelet_samples": 50, "batch_size": 20}

    def _extract_random_shapelet(self, X, y, i, shapelets, max_shapelets_per_class):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (i + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        inst_idx = i % self.n_instances
        cls_idx = int(y[inst_idx])
        worst_quality = (
            shapelets[cls_idx][0][0]
            if len(shapelets[cls_idx]) == max_shapelets_per_class
            else -1
        )

        length = (
            rng.randint(0, self._max_shapelet_length - self.min_shapelet_length)
            + self.min_shapelet_length
        )
        position = rng.randint(0, self.series_length - length)
        dim = rng.randint(0, self.n_dims)

        shapelet = z_normalise_series(X[inst_idx, dim, position : position + length])
        sabs = np.abs(shapelet)
        sorted_indicies = np.array(
            sorted(range(length), reverse=True, key=lambda j: sabs[j])
        )

        quality = self._find_shapelet_quality(
            X,
            y,
            shapelet,
            sorted_indicies,
            position,
            length,
            dim,
            inst_idx,
            self._class_counts[cls_idx],
            self.n_instances - self._class_counts[cls_idx],
            worst_quality,
        )

        return quality, length, position, dim, inst_idx, cls_idx

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _find_shapelet_quality(
        X,
        y,
        shapelet,
        sorted_indicies,
        position,
        length,
        dim,
        inst_idx,
        this_cls_count,
        other_cls_count,
        worst_quality,
    ):
        # todo optimise this more, we spend 99% of time here
        orderline = []
        this_cls_traversed = 0
        other_cls_traversed = 0

        for i, series in enumerate(X):
            if i != inst_idx:
                distance = _online_shapelet_distance(
                    series[dim], shapelet, sorted_indicies, position, length
                )
            else:
                distance = 0

            if y[i] == y[inst_idx]:
                cls = 1
                this_cls_traversed += 1
            else:
                cls = -1
                other_cls_traversed += 1

            orderline.append((distance, cls))
            orderline.sort()

            if worst_quality > 0:
                quality = _calc_early_binary_ig(
                    orderline,
                    this_cls_traversed,
                    other_cls_traversed,
                    this_cls_count - this_cls_traversed,
                    other_cls_count - other_cls_traversed,
                    worst_quality,
                )

                if quality <= worst_quality:
                    return -1

        quality = _calc_binary_ig(orderline, this_cls_count, other_cls_count)

        return round(quality, 12)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _merge_shapelets(
        shapelet_heap, candidate_shapelets, max_shapelets_per_class, cls_idx
    ):
        for shapelet in candidate_shapelets:
            if shapelet[5] == cls_idx and shapelet[0] > 0:
                if (
                    len(shapelet_heap) == max_shapelets_per_class
                    and shapelet[0] < shapelet_heap[0][0]
                ):
                    continue

                heapq.heappush(shapelet_heap, shapelet)

                if len(shapelet_heap) > max_shapelets_per_class:
                    heapq.heappop(shapelet_heap)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_self_similar_shapelets(shapelet_heap):
        to_keep = [True] * len(shapelet_heap)

        for i in range(len(shapelet_heap)):
            if to_keep[i] is False:
                continue

            for n in range(i + 1, len(shapelet_heap)):
                if to_keep[n] and _is_self_similar(shapelet_heap[i], shapelet_heap[n]):
                    if (shapelet_heap[i][0], shapelet_heap[i][1]) >= (
                        shapelet_heap[n][0],
                        shapelet_heap[n][1],
                    ):
                        to_keep[n] = False
                    else:
                        to_keep[i] = False
                        break

        return to_keep

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_identical_shapelets(shapelets):
        to_keep = [True] * len(shapelets)

        for i in range(len(shapelets)):
            for n in range(i + 1, len(shapelets)):
                if (
                    to_keep[n]
                    and shapelets[i][1] == shapelets[n][1]
                    and np.array_equal(shapelets[i][6], shapelets[n][6])
                ):
                    to_keep[n] = False

        return to_keep


@njit(fastmath=True, cache=True)
def _online_shapelet_distance(series, shapelet, sorted_indicies, position, length):
    subseq = series[position : position + length]

    sum = 0.0
    sum2 = 0.0
    for i in subseq:
        sum += i
        sum2 += i * i

    mean = sum / length
    std = (sum2 - mean * mean * length) / length
    if std > 0:
        subseq = (subseq - mean) / std
    else:
        subseq = np.zeros(length)

    best_dist = 0
    for i, n in zip(shapelet, subseq):
        temp = i - n
        best_dist += temp * temp

    i = 1
    traverse = [True, True]
    sums = [sum, sum]
    sums2 = [sum2, sum2]

    while traverse[0] or traverse[1]:
        for n in range(2):
            mod = -1 if n == 0 else 1
            pos = position + mod * i
            traverse[n] = pos >= 0 if n == 0 else pos <= len(series) - length

            if not traverse[n]:
                continue

            start = series[pos - n]
            end = series[pos - n + length]

            sums[n] += mod * end - mod * start
            sums2[n] += mod * end * end - mod * start * start

            mean = sums[n] / length
            std = math.sqrt((sums2[n] - mean * mean * length) / length)

            dist = 0
            use_std = std != 0
            for j in range(length):
                val = (series[pos + sorted_indicies[j]] - mean) / std if use_std else 0
                temp = shapelet[sorted_indicies[j]] - val
                dist += temp * temp

                if dist > best_dist:
                    break

            if dist < best_dist:
                best_dist = dist

        i += 1

    return best_dist if best_dist == 0 else 1 / length * best_dist


@njit(fastmath=True, cache=True)
def _calc_early_binary_ig(
    orderline,
    c1_traversed,
    c2_traversed,
    c1_to_add,
    c2_to_add,
    worst_quality,
):
    initial_ent = _binary_entropy(
        c1_traversed + c1_to_add,
        c2_traversed + c2_to_add,
    )

    total_all = c1_traversed + c2_traversed + c1_to_add + c2_to_add

    bsf_ig = 0
    # actual observations in orderline
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        # optimistically add this class to left side first and other to right
        left_prop = (split + 1 + c1_to_add) / total_all
        ent_left = _binary_entropy(c1_count + c1_to_add, c2_count)

        # because right side must optimistically contain everything else
        right_prop = 1 - left_prop

        ent_right = _binary_entropy(
            c1_traversed - c1_count,
            c2_traversed - c2_count + c2_to_add,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        # now optimistically add this class to right, other to left
        left_prop = (split + 1 + c2_to_add) / total_all
        ent_left = _binary_entropy(c1_count, c2_count + c2_to_add)

        # because right side must optimistically contain everything else
        right_prop = 1 - left_prop

        ent_right = _binary_entropy(
            c1_traversed - c1_count + c1_to_add,
            c2_traversed - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        if bsf_ig > worst_quality:
            return bsf_ig

    return bsf_ig


@njit(fastmath=True, cache=True)
def _calc_binary_ig(orderline, c1, c2):
    initial_ent = _binary_entropy(c1, c2)

    total_all = c1 + c2

    bsf_ig = 0
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        left_prop = (split + 1) / total_all
        ent_left = _binary_entropy(c1_count, c2_count)

        right_prop = 1 - left_prop
        ent_right = _binary_entropy(
            c1 - c1_count,
            c2 - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

    return bsf_ig


@njit(fastmath=True, cache=True)
def _binary_entropy(c1, c2):
    ent = 0
    if c1 != 0:
        ent -= c1 / (c1 + c2) * np.log2(c1 / (c1 + c2))
    if c2 != 0:
        ent -= c2 / (c1 + c2) * np.log2(c2 / (c1 + c2))
    return ent


@njit(fastmath=True, cache=True)
def _is_self_similar(s1, s2):
    # not self similar if from different series or dimension
    if s1[4] == s2[4] and s1[3] == s2[3]:
        if s2[2] <= s1[2] <= s2[2] + s2[1]:
            return True
        if s1[2] <= s2[2] <= s1[2] + s1[1]:
            return True

    return False
