# WORSHOP NOTE: v3 initially is a full shapelet transform (extension to random and contract should be ready to implement, but not
# enabled yet). Specific TO-DOs:
#   - TESTING:          HAS NOT BEEN TESTED THOROUGHLY. Good to test against shapelets from the java UEA implementation to check for parity.
#                       Important to make sure info gain calcs are correct with binary info gain (this has been rewritten from v2 to avoid
#                       multiple sorts and some calls that were not necessary. I believe it is correct, but good to test further)
#
#                       Note: known issue is that transform does not currently return shapelets - run main to see where it's up to :)
#
#   - distance measure: it has been rewired with the latest early abandon for evaluating a candidate (i.e. start in the same location of
#                       a comparison series as the candidate, then move left and right). However, full distance of the subsequence is
#                       currently calculated using np.linalg.norm(candidate-comparison)^2 as it's faster than doing a Euclidean distance
#                       early abandon! This means there's no benefit in the left and right search currently, but it is there and ready to
#                       go as the plan is to do the ED calculation as a cython function with early abandon (which should add benefit).
#                       Cython dist to-do however, and timing experiments necessary to settle on which approach is fastest
#
#   - contracting:      to validate correctness the transform below should be a full, sequential search of candidate shapelets. This should
#     (also random)     be extended to  random enumeration and also a random time contract transforms. I think the best way to do this is
#     (subclasses)      to create two subclasses, RandomShapeletTransform and RandomContractedShapeletTransform (or something more catchy).
#                       These should use the same fit method (plus others) from the full/base class; the subclasses should effectively
#                       handle setting up Full/Random/Contract separately to avoid overlapping arguements (such as number of cases to visit
#                       doesn't make sense for full, time contract doesn't make sense for random enumeration, etc.)


import os
import time
import warnings
import numpy as np

from collections import Counter
from itertools import zip_longest
from sklearn.base import TransformerMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import heapq

warnings.filterwarnings("ignore", category=FutureWarning)

# TO-DO: thorough testing (some initial testing completed, but passing the code to David to develop
#        before everything has been fully verified)

# TO-DO: in case of unequal length time series, we have two options:
# 1) fix the maximum length for the shapelets to the minimum length time series (of train dataset,
#      which can also fails if there is a smaller time series in the test set).
# 2) use another time series distance measure such as DTW to avoid this, since you can compare unequal
#      time series (we lose the early abandon in the distance measurement).

# TO-DO: Currently extends TransformerMixin - class should extend the sktime transformer base class (not on dev at
#        branch at time of writing however so this can be updated later)


class ShapeletTransform(TransformerMixin):

    def __init__(self,
                 min_shapelet_length=3,
                 max_shapelet_length=np.inf,
                 max_shapelets_per_class=200,
                 dims_to_use=0,
                 random_state=None,
                 verbose=0,
                 remove_self_similar=True,
                 use_binary_info_gain=True,
                 independent_dimensions=False
                 ):

        self.shapelet_search_algo = 'full'
        self.shapelets = None

        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length

        self.max_shapelets_per_class = max_shapelets_per_class

        if isinstance(dims_to_use, (list, tuple, np.ndarray)) is True:
            self.dims_to_use = dims_to_use
        else:
            self.dims_to_use = [dims_to_use]

        self.random_state = random_state
        self.verbose = verbose

        self.remove_self_similar = remove_self_similar
        self.use_binary_info_gain = use_binary_info_gain
        self.independent_dimensions = independent_dimensions

        self.predefined_ig_rejection_level = 0.3


    def fit(self, X, y, **fit_params):
        """A method to fit the shapelet transform to a specified X and y

        Parameters
        ----------
        X: pandas DataFrame
            The training input samples.
        y: array-like or list
            The class values for X

        Returns
        -------
        self : FullShapeletTransform
            This estimator
        """

        X_lens = np.array([len(X.iloc[r,self.dims_to_use[0]]) for r in range(len(X))])
        X = np.array([[X.iloc[r,c].values for c in self.dims_to_use] for r in range(len(X))]) # may need to pad with nans here for uneq length, look at later


        num_ins = len(y)
        distinct_class_vals = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        if type(self) is RandomEnumerationShapeletTransform:
            num_series_to_visit = min(self.num_cases_to_sample, len(y))
        else:
            num_series_to_visit = num_ins

        # shapelet_heaps_by_class = {i: ShapeletHeap() for i in distinct_class_vals}
        shapelet_heaps_by_class = {i: ShapeletPQ() for i in distinct_class_vals}
        class_counts = dict(Counter(y))

        self.random_state = check_random_state(self.random_state)

        # Here we establish the order of cases to sample. We need to sample x cases and y shapelets from each (where x = num_cases_to_sample
        # and y = num_shapelets_to_sample_per_case). We could simply sample x cases without replacement and y shapelets from each case, but
        # the idea is that if we are using a time contract we may extract all y shapelets from each x candidate and still have time remaining.
        # Therefore, if we get a list of the indices of the series and shuffle them appropriately, we can go through the list again and extract
        # another y shapelets from each series (if we have time).

        # We also want to ensure that we visit all classes so we will visit in round-robin order. Therefore, the code below extracts the indices
        # of all series by class, shuffles the indices for each class independently, and then combines them in alternating order. This results in
        # a shuffled list of indices that are in alternating class order (e.g. 1,2,3,1,2,3,1,2,3,1...)

        def _round_robin(*iterables):
            sentinel = object()
            return (a for x in zip_longest(*iterables, fillvalue=sentinel) for a in x if a != sentinel)

        # case_ids_by_class = {i: shuffle(np.where(y == i)[0], random_state=self.random_state) for i in distinct_class_vals}
        case_ids_by_class = {i: np.where(y == i)[0] for i in distinct_class_vals}
        num_train_per_class = {i : len(case_ids_by_class[i]) for i in case_ids_by_class}
        round_robin_case_order = _round_robin(*[list(v) for k, v in case_ids_by_class.items()])
        cases_to_visit = [(i, y[i]) for i in round_robin_case_order]
        # this dictionary will be used to store all possible starting positions and shapelet lengths for a give series length. This
        # is because we enumerate all possible candidates and sample without replacement when assessing a series. If we have two series
        # of the same length then they will obviously have the same valid shapelet starting positions and lengths (especially in standard
        # datasets where all series are equal length) so it makes sense to store the possible candidates and reuse, rather than
        # recalculating each time

        # Initially the dictionary will be empty, and each time a new series length is seen the dict will be updated. Next time that length
        # is used the dict will have an entry so can simply reuse
        possible_candidates_per_series_length = {}

        # a flag to indicate if extraction should stop (contract has ended)
        time_finished = False
        can_ig_early_abandon = False # can only abandon IG calculations when there is a limit, aka IG of worst shapelet if max shapelets have been reached

        # max time calculating a shapelet
        # for timing the extraction when contracting
        start_time = time.time()
        time_taken = lambda: time.time() - start_time
        max_time_calc_shapelet = -1
        time_last_shapelet = time_taken()

        # for every series
        # for idx, series_id_and_class in enumerate(cases_to_visit):
        for case_idx in range(len(cases_to_visit)):

            series_id = cases_to_visit[case_idx][0]
            this_class_val = cases_to_visit[case_idx][1]

            # minus 1 to remove this candidate from sums
            binary_ig_this_class_count = num_train_per_class[this_class_val]-1
            binary_ig_other_class_count = num_ins-binary_ig_this_class_count-1

            if self.verbose:
                if type(self) == ContractedShapeletTransform:
                    print("visiting series: " + str(series_id) + " (#" + str(case_idx + 1) + "/" + str(num_series_to_visit) + ")")
                else:
                    print("visiting series: " + str(series_id) + " (#" + str(case_idx + 1) + ")")

            this_series_len = len(X[series_id][0])

            # The bohund on possible shapelet lengths will differ series-to-series if using unequal length data.
            #             # However, shapelets cannot be longer than te series, so set to the minimum of the series length
            # and max shapelet length (which is inf by default, unless set)
            if self.max_shapelet_length == -1:
                this_shapelet_length_upper_bound = this_series_len
            else:
                this_shapelet_length_upper_bound = min(this_series_len, self.max_shapelet_length)

            # all possible start and lengths for shapelets within this series (calculates if series length is new, a simple look-up if not)
            # enumerate all possible candidate starting positions and lengths.

            # First, try to reuse if they have been calculated for a series of the same length before.
            candidate_starts_and_lens = possible_candidates_per_series_length.get(this_series_len)
            # else calculate them for this series length and store for possible use again
            if candidate_starts_and_lens is None:
                candidate_starts_and_lens = [
                    [start, length] for start in range(0, this_series_len - self.min_shapelet_length + 1)
                    for length in range(self.min_shapelet_length, this_shapelet_length_upper_bound + 1) if start + length <= this_series_len]
                possible_candidates_per_series_length[this_series_len] = candidate_starts_and_lens

            # default for full transform
            candidates_to_visit = candidate_starts_and_lens
            num_candidates_per_case = len(candidate_starts_and_lens)

            # limit search otherwise:
            if hasattr(self,"num_shapelets_to_sample_per_case"):
                num_candidates_per_case = min(self.num_shapelets_to_sample_per_case, num_candidates_per_case)
                cand_idx = list(self.random_state.choice(list(range(0, len(candidate_starts_and_lens))), num_candidates_per_case, replace=False))
                candidates_to_visit = [candidate_starts_and_lens[x] for x in cand_idx]

            for candidate_idx in range(num_candidates_per_case):
                can_ig_early_abandon = False
                if shapelet_heaps_by_class[this_class_val].get_size() >= self.max_shapelets_per_class:
                    can_ig_early_abandon = True
                    #get the ig of the current worst shapelet for this class as a bound
                    ig_cutoff = shapelet_heaps_by_class[this_class_val].peek().info_gain

                cand_start_pos = candidates_to_visit[candidate_idx][0]
                cand_len = candidates_to_visit[candidate_idx][1]

                candidate = ShapeletTransform.zscore(X[series_id][:,cand_start_pos: cand_start_pos + cand_len])
                stop = False

                # now go through all other series and get a distance from the candidate to each
                orderline = []

                # initialise here as copy, decrease the new val each time we evaliate a comparison series
                num_visited_this_class = 0
                num_visited_other_class = 0

                ig_upper_bound = 1.0 # bsf starts at 1
                candidate_rejected = False

                for comparison_series_idx in range(len(cases_to_visit)):
                    # print("i: "+str(comparison_series_idx))
                    i = cases_to_visit[comparison_series_idx][0]

                    if y[i] != cases_to_visit[comparison_series_idx][1]:
                        raise ValueError("class match sanity test broken")

                    if i == series_id:
                        # don't evaluate candidate against own series
                        continue

                    if y[i]==this_class_val:
                        num_visited_this_class += 1
                        binary_class_identifier = 1 # positive for this class
                    else:
                        num_visited_other_class += 1
                        binary_class_identifier = -1 # negative for any other class

                    bsf_dist = np.inf

                    overlap = False
                    start_left = cand_start_pos
                    start_right = cand_start_pos+1

                    if X_lens[i]==cand_len:
                        start_left = 0
                        start_right = 0

                    for num_cals in range(max(1,int(np.ceil((X_lens[i]-cand_len)/2)))): # max used to force iteration where series len == candidate len
                        if start_left < 0:
                            start_left = X_lens[i]-1-cand_len

                        comparison = ShapeletTransform.zscore(X[i][:,start_left: start_left+ cand_len])
                        dist_left = np.linalg.norm(candidate-comparison)
                        bsf_dist = min(dist_left*dist_left, bsf_dist)

                        # for odd lengths
                        if start_left == start_right:
                            continue

                        #right
                        if start_right == X_lens[i]-cand_len+1:
                            start_right = 0
                        comparison = ShapeletTransform.zscore(X[i][:,start_right: start_right+ cand_len])
                        dist_right = np.linalg.norm(candidate-comparison)
                        bsf_dist = min(dist_right*dist_right, bsf_dist)

                        start_left-=1
                        start_right+=1

                    orderline.append((bsf_dist,binary_class_identifier))
                    # sorting required after each add for early IG abandon.
                    # timsort should be efficient as array is almost in order - insertion-sort like behaviour in this case.
                    # Can't use heap as need to traverse in order multiple times, not just access root
                    orderline.sort()

                    # print("order-len: "+str(len(orderline)))

                    # no point trying to abandon with 2 or less distances, also no point early abandoning when the last thing is added (just calc full ig after)
                    if can_ig_early_abandon and len(orderline) > 2 and comparison_series_idx != len(cases_to_visit)-1:
                        # calc optimistic ig bound here, reject candidate if possible
                        ig_upper_bound = ShapeletTransform.calc_early_binary_ig(num_visited_this_class, num_visited_other_class, binary_ig_this_class_count-num_visited_this_class, binary_ig_other_class_count-num_visited_other_class)
                        if ig_upper_bound <= ig_cutoff or ig_upper_bound < self.predefined_ig_rejection_level:
                            candidate_rejected = True
                            break

                if candidate_rejected:
                    continue
                else:
                    final_ig = ShapeletTransform.calc_binary_ig(orderline, binary_ig_this_class_count, binary_ig_other_class_count)

                accepted_candidate = Shapelet(series_id, self.dims_to_use, cand_start_pos, cand_len, final_ig, candidate)
                # print("series: "+str(accepted_candidate.series_id))
                # print("start:  "+str(accepted_candidate.start_pos))
                # print("length: "+str(accepted_candidate.length))
                # print("info:   "+str(accepted_candidate.info_gain))
                # print()

                # add to min heap to store shapelets for this class
                shapelet_heaps_by_class[this_class_val].push(accepted_candidate)

                if shapelet_heaps_by_class[this_class_val].get_size() > self.max_shapelets_per_class*1.1:
                    shapelet_heaps_by_class[this_class_val].pop()

                # Takes into account the use of the MAX shapelet calculation time to don't exceed the time_limit.
                if hasattr(self,'time_limit') and self.time_limit > 0:
                    time_now = time_taken()
                    time_this_shapelet = (time_now - time_last_shapelet)
                    if time_this_shapelet > max_time_calc_shapelet:
                        max_time_calc_shapelet = time_this_shapelet
                    time_last_shapelet = time_now
                    if (time_now + max_time_calc_shapelet) > self.time_limit:
                        if self.verbose > 0:
                            print("No more time available! It's been {0:02d}:{1:02}".format(int(round(time_now / 60, 3)), int((round(time_now / 60, 3) - int(round(time_now / 60, 3))) * 60)))
                        time_finished = True
                        break
                    else:
                        if self.verbose > 0:
                            print("Candidate finished. {0:02d}:{1:02} remaining".format(int(round((self.time_limit - time_now) / 60, 3)),
                                                                                        int((round((self.time_limit - time_now) / 60, 3) - int(round((self.time_limit - time_now) / 60, 3))) * 60)))

            # stopping condition: in case of iterative transform (i.e. num_cases_to_sample have been visited)
            #                     in case of contracted transform (i.e. time limit has been reached)
            if case_idx >= num_series_to_visit or time_finished:
                if self.verbose > 0:
                    print("Stopping search")
                break


        # remove self similar here
        #
        # # sort all shapelets by quality
        # all_shapelets_classes = [item for k, v in shapelets_by_class.items() for item in v]
        # all_shapelets_classes.sort(key=lambda x: x.info_gain, reverse=True)
        #
        # # moved to end as it is now possible to visit the same series multiple times, and a better series may be found in the second visit that removes
        # # the best from the first (and then means previously similar shapelets with that may again be eligible)
        # if self.remove_self_similar:
        #     all_shapelets_classes = ShapeletTransform.remove_self_similar(all_shapelets_classes)
        #
        # # we keep the best num_shapelets_to_trim_to shapelets
        # if self.num_shapelets_to_keep < len(all_shapelets_classes):
        #     all_shapelets_classes = all_shapelets_classes[:self.num_shapelets_to_keep]

        # self.shapelets = all_shapelets_classes
        # return self.shapelets

    @staticmethod
    def remove_self_similar(shapelet_list):
        """Remove self-similar shapelets from an input list. Note: this method assumes
        that shapelets are pre-sorted in descending order of quality (i.e. if two candidates
        are self-similar, the one with the later index will be removed)

        Parameters
        ----------
        shapelet_list: list of Shapelet objects

        Returns
        -------
        shapelet_list: list of Shapelet objects
        """

        # IMPORTANT: shapelets must be in descending order of quality. This is preferable in the fit method as removing self-similar
        # shapelets may be False. However, could be dangerous if something else uses this code later.

        def is_self_similar(shapelet_one, shapelet_two):
            # not self similar if from different series
            if shapelet_one.series_id != shapelet_two.series_id:
                return False

            if (shapelet_one.start_pos >= shapelet_two.start_pos) and (shapelet_one.start_pos <= shapelet_two.start_pos + shapelet_two.length):
                return True
            if (shapelet_two.start_pos >= shapelet_one.start_pos) and (shapelet_two.start_pos <= shapelet_one.start_pos + shapelet_one.length):
                return True

        to_return = [shapelet_list[0]]  # first shapelet must be ok
        for s in range(1, len(shapelet_list)):
            can_add = True
            for c in range(0, s):
                if is_self_similar(shapelet_list[s], shapelet_list[c]):
                    can_add = False
                    break
            if can_add:
                to_return.append(shapelet_list[s])

        return to_return
    # two "self-similar" shapelets are subsequences from the same series that are overlapping. This method

    # transform a set of data into distances to each extracted shapelet
    def transform(self, X, **transform_params):
        """Transforms X according to the extracted shapelets (self.shapelets)

        Parameters
        ----------
        X : pandas DataFrame
            The input dataframe to transform

        Returns
        -------
        output : pandas DataFrame
            The transformed dataframe in tabular format.
        """
        if self.shapelets is None:
            raise Exception("Fit not called yet - no shapelets exist!")

        X = X.iloc[:, self.dims_to_use]
        X_aux = [[]] * len(X)
        for i in range(0, len(X)):
            X_aux[i] = np.array([np.asarray(x) for x in np.asarray(X.iloc[i, :])])
        X = X_aux.copy()

        output = np.zeros([len(X), len(self.shapelets)], dtype=np.float32, )

        for i in range(0, len(X)):
            this_series = X[i]

            for s in range(0, len(self.shapelets)):
                # find distance between this series and each shapelet
                min_dist = np.inf
                this_shapelet_length = self.shapelets[s].length

                # TODO: We have to think about how to work when we have shapelets with length higher than min(lengths_test_set)
                if (self.independent_dimensions):
                    # In this case, as we are measuring distances between shapelets and series at different points, the use of a nested list is
                    # needed to use the same code for zscore and euclideanDistanceEarlyAbandon
                    for dim in range(0, len(self.dims_to_use)):
                        for start_pos in range(0, len(this_series[dim]) - this_shapelet_length + 1):
                            comp_2 = this_series[dim, start_pos:start_pos + this_shapelet_length]
                            comp = ShapeletTransform.zscore([comp_2])
                            min_dist = ShapeletTransform.euclideanDistanceEarlyAbandon([self.shapelets[s].data[dim]], comp, min_dist)

                else:
                    for start_pos in range(0, len(this_series[0]) - this_shapelet_length + 1):
                        comp_2 = this_series[:, start_pos:start_pos + this_shapelet_length]
                        comp = ShapeletTransform.zscore(comp_2)
                        min_dist = ShapeletTransform.euclideanDistanceEarlyAbandon(self.shapelets[s].data, comp, min_dist)

                try:
                    output[i][s] = min_dist.astype(np.float32)
                except Exception:
                    output[i][s] = np.float32(min_dist)


        return output

    def fit_transform(self, X, y=None, **fit_params):
        """Fits and transforms a given input X and y

        Parameters
        ----------
        X: pandas.DataFrame the input data to transform
        y: list or array like of class values corresponding to the indices in X

        Returns
        -------
        Xt : pandas DataFrame
            The transformed pandas DataFrame.
        """
        if self.shapelets is None and y is not None:
            self.fit(X, y)
        elif self.shapelets is not None:
            raise Exception("Trying to fit but shapelets already exist.")
        else:
            raise Exception("No class values specified - shapelet extraction is supervised")

        return self.transform(X)

    def get_shapelets(self):
        """An accessor method to return the extracted shapelets

        Returns
        -------
        shapelets: a list of Shapelet objects
        """
        return self.shapelets

    @staticmethod
    def binary_entropy(num_this_class, num_other_class):
            ent = 0
            if num_this_class != 0:
                ent -= num_this_class / (num_this_class + num_other_class) * np.log2(num_this_class / (num_this_class + num_other_class))
            if num_other_class != 0:
                ent -= num_other_class / (num_this_class + num_other_class) * np.log2(num_other_class / (num_this_class + num_other_class))
            return ent

    @staticmethod
    def calc_binary_ig(orderline, total_num_this_class, total_num_other_class):
        # def entropy(ent_class_counts, all_class_count):

        initial_ent = ShapeletTransform.binary_entropy(total_num_this_class, total_num_other_class)
        bsf_ig = 0

        count_this_class = 0
        count_other_class = 0

        total_all = total_num_this_class+total_num_other_class

        # evaluate each split point
        for split in range(0, len(orderline) - 1):
            next_class = orderline[split][1] # +1 if this class, -1 if other
            if next_class > 0:
                count_this_class += 1
            else:
                count_other_class += 1

            # optimistically add this class to left side first and other to right
            left_prop = (split + 1) / total_all
            ent_left = ShapeletTransform.binary_entropy(count_this_class,count_other_class)

            right_prop = 1-left_prop # because right side must optimistically contain everything else
            ent_right = ShapeletTransform.binary_entropy(total_num_this_class-count_this_class,total_num_other_class-count_other_class)

            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

        return bsf_ig

    @staticmethod
    def calc_early_binary_ig(orderline, num_this_class_in_orderline, num_other_class_in_orderline, num_to_add_this_class, num_to_add_other_class):
        # def entropy(ent_class_counts, all_class_count):

        initial_ent = ShapeletTransform.binary_entropy(num_this_class_in_orderline+num_to_add_this_class, num_other_class_in_orderline+num_to_add_other_class)
        bsf_ig = 0

        # actual observations in orderline
        count_this_class = 0
        count_other_class = 0

        total_all = num_this_class_in_orderline+num_other_class_in_orderline+num_to_add_this_class+num_other_class_in_orderline

        # evaluate each split point
        for split in range(0, len(orderline) - 1):
            next_class = orderline[split][1] # +1 if this class, -1 if other
            if next_class > 0:
                count_this_class += 1
            else:
                count_other_class += 1

            # optimistically add this class to left side first and other to right
            left_prop = (split + 1 + num_to_add_this_class) / total_all
            ent_left = ShapeletTransform.binary_entropy(count_this_class+num_to_add_this_class,count_other_class)

            right_prop = 1-left_prop # because right side must optimistically contain everything else
            ent_right = ShapeletTransform.binary_entropy(num_this_class_in_orderline-count_this_class,num_other_class_in_orderline-count_other_class+num_to_add_other_class)

            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

            # now optimistically add this class to right, other to left
            left_prop = (split + 1 + num_to_add_other_class) / total_all
            ent_left = ShapeletTransform.binary_entropy(count_this_class,count_other_class+num_to_add_other_class)

            right_prop = 1-left_prop # because right side must optimistically contain everything else
            ent_right = ShapeletTransform.binary_entropy(num_this_class_in_orderline-count_this_class+num_to_add_this_class,num_other_class_in_orderline-count_other_class)
            ig = initial_ent - left_prop * ent_left - right_prop * ent_right
            bsf_ig = max(ig, bsf_ig)

        return bsf_ig


    @staticmethod
    def zscore(a, axis=0, ddof=0):
        """ A static method to return the normalised version of series.  This mirrors the scipy implementation
        with a small difference - rather than allowing /0, the function returns output = np.zeroes(len(input)).
        This is to allow for sensible processing of candidate shapelets/comparison subseries that are a straight
        line. Original version: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html

        Parameters
        ----------
        a : array_like
            An array like object containing the sample data.

        axis : int or None, optional
            Axis along which to operate. Default is 0. If None, compute over the whole array a.

        ddof : int, optional
            Degrees of freedom correction in the calculation of the standard deviation. Default is 0.

        Returns
        -------
        zscore : array_like
            The z-scores, standardized by mean and standard deviation of input array a.
        """
        zscored = np.empty(a.shape)
        for i, j in enumerate(a):
            # j = np.asanyarray(j)
            sstd = j.std(axis=axis, ddof=ddof)

            # special case - if shapelet is a straight line (i.e. no variance), zscore ver should be np.zeros(len(a))
            if sstd == 0:
                zscored[i] = np.zeros(len(j))
            else:
                mns = j.mean(axis=axis)
                if axis and mns.ndim < j.ndim:
                    zscored[i] = ((j - np.expand_dims(mns, axis=axis)) /
                                    np.expand_dims(sstd, axis=axis))
                else:
                    zscored[i] = (j - mns) / sstd
        return zscored

    @staticmethod
    def euclideanDistanceEarlyAbandon(u, v, min_dist):
        sum_dist = 0
        for i in range(0, len(u[0])):
            for j in range(0, len(u)):
                u_v = u[j][i] - v[j][i]
                sum_dist += np.dot(u_v, u_v)
                if sum_dist >= min_dist:
                    # The distance is higher, so early abandon.
                    return min_dist
        return sum_dist

    # @staticmethod
    # cdef ed_early_abandon(np.ndarray[double, ndim=2] candidate, np.ndarray[double, ndim=2] comparison, double bsf_dist=np.inf):
    #     cdef double dist = 0
    #     cdef double point_dist = 0
    #
    #     for dim in range(len(candidate)):
    #         for val in range(len(candidate[dim])):
    #             point_dist = candidate[dim][val]
    #             dist+=point_dist*point_dist
    #             if dist > bsf_dist:
    #                 return np.inf
    #     return dist

class ContractedShapeletTransform(ShapeletTransform):
    pass

class RandomEnumerationShapeletTransform(ShapeletTransform):
    pass

class Shapelet:
    """A simple class to model a Shapelet with associated information

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
        The (z-normalised) data of this shapelet
    """

    def __init__(self, series_id, dims, start_pos, length, info_gain, data):
        self.series_id = series_id
        self.dims = dims
        self.start_pos = start_pos
        self.length = length
        self.info_gain = info_gain
        self.data = data

    def __str__(self):
        return "Series ID: {0}, num_dim: {1}, start_pos: {2}, length: {3}, info_gain: {4}, ".format(self.series_id, self.dims, self.start_pos, self.length, self.info_gain)

# class ShapeletHeap(object):
#    def __init__(self, ):
#        self._data = []
#
#    def push(self, shapelet_to_push):
#        # heapq.heappush(self._data, shapelet_to_push.info_gain)
#        # exit()
#        heapq.heappush(self._data, (shapelet_to_push.info_gain, shapelet_to_push))
#
#    def pop(self):
#        return heapq.heappop(self._data)[1]
#
#    def peek(self):
#        return self._data[0]
#
#    def get_size(self):
#        return len(self._data)
#
#    def get_list(self):
#        return self._data

class ShapeletPQ:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, shapelet):
        heapq.heappush(self._queue, (shapelet.info_gain, self._index, shapelet))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def peek(self):
        return self._queue[0]

    def get_size(self):
        return len(self._queue)


def saveTransform(transform, labels, file_name):
    """ A simple function to save the transform obtained in arff format

    Parameters
    ----------
    transform: array-like
        The transform obtained for a dataset
    labels: array-like
        The labels of the dataset
    file_name: string
        The directory to save the transform
    """
    # Create directory in case it doesn't exists
    directory = '/'.join(file_name.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    num_shapelets = transform.shape[1]
    unique_labels = np.unique(labels).tolist()

    with open(file_name, 'w+') as f:
        # Headers
        f.write("@Relation Shapelets" + file_name.split('/')[-1].split('_')[0] + '\n\n')
        for i in range(0, num_shapelets):
            f.write("@attribute Shapelet_" + str(i) + " numeric\n")
        f.write("@attribute target {" + ",".join(unique_labels) + "}\n")
        f.write("\n@data\n")
        # Patterns
        for i, j in enumerate(transform):
            pattern = j.tolist() + [int(float(labels[i]))]
            f.write(",".join(map(str, pattern)) + "\n")
    f.close()


def saveShapelets(shapelets, data, dim_to_use, time, file_name):
    """ A simple function to save the shapelets obtained in csv format

    Parameters
    ----------
    shapelets: array-like
        The shapelets obtained for a dataset
    data: array-like
        The original data
    time: fload
        The time spent obtaining shapelets
    file_name: string
        The directory to save the set of shapelets
    """
    data = data.iloc[:, dim_to_use]

    data_aux = [[]] * len(data)
    for i in range(0, len(data)):
        data_aux[i] = np.array([np.asarray(x) for x in np.asarray(data.iloc[i, :])])
    data = data_aux.copy()

    # Create directory in case it doesn't exists
    directory = '/'.join(file_name.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, 'w+') as f:
        # Number of shapelets and time extracting
        f.write(str(len(shapelets)) + "," + str(time) + "\n")
        for i, j in enumerate(shapelets):
            f.write(str(j.info_gain) + "," + str(j.series_id) + "," + ''.join(str(j.dims)).replace(', ', ':') + "," + str(j.start_pos) + "," + str(j.length) + "\n")
            for k in range(0, len(dim_to_use)):
                f.write(",".join(map(str, data[j.series_id][k, j.start_pos:j.start_pos + j.length])) + "\n")
    f.close()




#
# if __name__ == "__main__":
#
#     train_x, train_y = load_from_tsfile_to_dataframe("../datasets/data/GunPoint/GunPoint_TRAIN.ts")
#
#     st = ShapeletTransform(algo='full', verbose=2)
#     st.fit(train_x, train_y)

if __name__ == "__main__":

    from sktime.utils.load_data import load_from_arff_to_dataframe
    dataset = "GunPoint"
    # dataset = "BasicMotions"

    train_x, train_y = load_from_tsfile_to_dataframe("../../datasets/data/"+dataset+"/"+dataset+"_TRAIN.ts")
    test_x, test_y = load_from_tsfile_to_dataframe("../../datasets/data/"+dataset+"/"+dataset+"_TRAIN.ts")
    # X, y = load_from_arff_to_dataframe("../datasets/data/BasicMotions/BasicMotions_TRAIN.arff")

    a = ShapeletTransform(random_state=0, dims_to_use=0, verbose=3, min_shapelet_length=150, max_shapelet_length=150)
    start_time = time.time()
    print(train_y[0:10])
    shapelets = a.fit(train_x[0:10], train_y[0:10])
    # shapelets = a.fit(train_x[0:10], train_y[0:10])
    end_time = time.time()
    print("time: "+str(end_time-start_time))
    print()
    for s in a.shapelets:
        print(s)
    exit()

