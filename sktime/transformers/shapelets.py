import numpy as np
from scipy.spatial.distance import sqeuclidean
import time
from .base import BaseTransformer

# TO-DO: thorough testing (some initial testing completed, but passing the code to David to develop
#        before everything has been fully verified)
# TO-DO: check the validity of the binary info gain method and implement the early abandon as
#        in original shapelet paper (not possible for non-binary IG however)
# TO-DO: revisit contract timing - currently it's lazy (will process the last shapelet and go over the limit)
#        A more sensible approach would be to estimate the average/median/max shapelet calculation time and
#        only process a new candidate if there is enough time left. Max would be safest but most conservative
#        estimate, mean is easiest to calculate without storing/sorting run-times. Investigation req.
# TO-DO: Random and ContractedRandom transforms use all of the same logic but have different class types for
#        distinctions (Contracted extends Random with flags to set). However, Tony would now prefer to have a
#        single class with a flag for contract or not, so code should be refactored into a single class
# TO-DO: Transform currently only for univariate data - once all of the above is reconciled we should extend
#        to the multi-variate case (should be fairly simple - done in Java, though not sure on the specifics/how
#        complete it is and if we could use mutivariate data in other ways too
# TO-DO: Currently extends TransformerMixin - class should extend the sktime transformer base class (not on dev at
#        branch at time of writing however so this can be updated later)
# TO-DO: Add a parameter to cap the number of shapelets to use in the final transform. Not done yet as we should
#        look at the Java implementation to see what the default was. If there isn't one, we could use an
#        arbitrary value such as 200 as a default? The limit would be after all shapelets are extracted and sorted
#        in descending order of information gain (e.g. keep the top 200 shapelets, not the first 200 that
#        were visited)
# TO-DO: verbose output of time remaining for contract is currently in decimal format, e.g.:
#        > Candidate finished. 0.418 minutes remaining.
#        it would be nicer (but not that important) to convert this to mins:secs rather than a decimal for minutes
#        e.g. 5:45 left, rather than 5.75


class RandomShapeletTransform(BaseTransformer):
    """Random Shapelet Transform.

    A transformer to extract shapelets from a training dataset, then transform any
    passed dataset into primitives using each extracted shapelet. For k shapelets,
    an input case q is transformed by calculating the distance from q to each shapelet
    and the transformed output case is composed of k features, where the kth attribute
    is the distance from q to the kth shapelet.

    This implementation of the transform is based on the findings in [2]; it uses random
    sampling of candidate shapelets, rather than a full enumeration of candidates as
    originally proposed in [1], as it has been shown that there is no significant
    decrease in accuracy but a significant reduction in runtime through this approach.

    There are two versions of the transform: RandomShapeletTransform and
    ContractedRandomShapeletTransform. For each training series visited, both
    implementations assess a specified number of candidate shapelets per series. However,
    this version visits an explict number of cases to extract these shapelets from (from
    1 to len(X), whereas ContractedRandomShapeletTransform visits training cases while
    time remains on a contact (specified in minutes).

    Parameters
    ----------
    min_shapelet_length : int (default = 3)
        The minimum candidate shapelet length
    max_shapelet_length :int (default = np.inf)
        The maximum candidate shapelet length (default = np.inf). This value is
        an upper-bound and is automatically capped to the series length if the
        specified value exceeds the length of the shapelet.
    num_cases_to_sample: int (default = 5)
        The number of training cases to extract candidate shapelets from.
    num_shapelets_to_sample_per_case: int (default = 15)
        The number of shapelet candidates to evaluate per series in the fit method
    seed: int (default = 0)
        To seed the shapelet discovery to ensure deterministic results across multiple runs
    dim_to_use: int (default = 0)
        Which dimension of the data to use
    remove_self_similar: bool (default = True)
        Whether to remove self-similar shapelets before transforming. A candidate shapelet
        is considered to be self-similar to another candidate if they are taken from the
        same dimension of the same training series. With this set to True, any overlapping
        candidates will be removed to preserve the best candidate (i.e. highest info gain)
    verbose: bool (default = False)
        Whether to print information during shapelet extraction
    use_binary_info_gain: bool (default = True)
        Whether to use one-vs-all information gain (as in [2]) or consider each class
        independently when calculating the information gain of a candidate (as in [1])
    trim_shapelets: bool (default = True)
        Whether to crop the final list of extracted shapelets to the top k. If specified,
        this is an upper-bound; if less than k shapelets are found then this parameter will
        have no effect
    num_shapelets_to_trim_to: int (default = 200)
        If trim_shapelets = True, this is the number of shapelets to trim to.

    Attributes
    ----------
    shapelets: list of Shapelet objects
        The list of extracted shapelets after fit is called (initially shapelets = None)


    References:
    -----------
    [1] Hills, Jon, Jason Lines, Edgaras Baranauskas, James Mapp, and Anthony Bagnall.
    "Classification of time series by shapelet transformation." Data Mining and Knowledge
    Discovery 28, no. 4 (2014): 851-881.

    [2] Bostrom, Aaron, and Anthony Bagnall.
    "Binary shapelet transform for multiclass time series classification."
    In Transactions on Large-Scale Data-and Knowledge-Centered Systems XXXII,
    pp. 24-46. Springer, Berlin, Heidelberg, 2017.
    """
    def __init__(self,
                 min_shapelet_length=3,
                 max_shapelet_length=np.inf,
                 num_cases_to_sample=5,
                 num_shapelets_to_sample_per_case=15,
                 seed=0,
                 dim_to_use=0,
                 remove_self_similar=True,
                 verbose=False,
                 use_binary_info_gain=True,
                 trim_shapelets=True,
                 num_shapelets_to_trim_to=200
                 ):
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.num_cases_to_sample = num_cases_to_sample
        self.num_shapelets_to_sample_per_case = num_shapelets_to_sample_per_case
        self.seed = seed
        self.dim_to_use = dim_to_use
        self.remove_self_similar = remove_self_similar
        self.shapelets = None
        self.time_limit_on = False
        self.time_limit = np.inf
        self.verbose = verbose
        self.use_binary_info_gain = use_binary_info_gain
        self.trim_shapelets = trim_shapelets
        self.num_shapelets_to_trim_to = num_shapelets_to_trim_to

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
        self : RandomShapeletTransform
            This estimator
        """
        X = np.array([np.asarray(x) for x in X.iloc[:, self.dim_to_use]])
        num_ins = len(y)
        class_vals = np.sort(np.array([x for x in set(y)]))

        cases_to_sample = self.num_cases_to_sample
        if self.time_limit_on is False and self.num_cases_to_sample > len(y):
            cases_to_sample = len(y)

        class_counts = {x:0 for x in class_vals}
        for c_val in y:
            class_counts[c_val] +=1

        rand = np.random.RandomState(seed=self.seed)
        rand.seed(self.seed)

        # Here we establish the order of cases to sample. We need to sample x cases and y shapelets from each (where x = num_cases_to_sample
        # and y = num_shapelets_to_sample_per_case). We could simply sample x cases without replacement and y shapelets from each case, but
        # the idea is that if we are using a time contract we may extract all y shapelets from each x candidate and still have time remaining.
        # Therefore, if we get a list of the indices of the series and shuffle them appropriately, we can go through the list again and extract
        # another y shapelets from each series (if we have time).
        #
        # We also want to ensure that we visit all classes so we will visit in round-robin order. Therefore, the code below extracts the indices
        # of all series by class, shuffles the indices for each class independently, and then combines them in alternating order. This results in
        # a shuffled list of indices that are in alternating class order (e.g. 1,2,3,1,2,3,1,2,3,1...)

        # indices by class
        idxs_to_sample_by_class = {x:[] for x in class_vals}
        for i in range(0, len(y)):
            idxs_to_sample_by_class[y[i]].append((i,y[i]))  # (index, class_val)

        # shuffle lists and get iterators for each class list
        class_val_iterators = {}
        for c in range(0, len(class_vals)):
            rand.shuffle(idxs_to_sample_by_class[class_vals[c]])
            class_val_iterators[class_vals[c]] = (iter(idxs_to_sample_by_class[class_vals[c]]))

        # now iterate through each and add to a single list of indices (with class vals as tuples for convenience)
        to_add = len(y)
        idxs_to_sample = []
        while to_add > 0:
            start_quant = to_add
            for c in range(0,len(class_vals)):
                try:
                    idxs_to_sample.append(next(class_val_iterators[class_vals[c]]))
                    to_add -= 1
                except StopIteration:
                    pass
            if to_add == start_quant:
                raise IndexError("Unexpected end of data - more class labels than instances")

        # once extracted we will add all shapelets to this list
        all_shapelets = []

        # this dictionary will be used to store all possible starting positions and shapelet lengths for a give series length. This
        # is because we enumerate all possible candidates and sample without replacement when assessing a series. If we have two series
        # of the same length then they will obviously have the same valid shapelet starting positions and lengths (especially in standard
        # datasets where all series are equal length) so it makes sense to store the possible candidates and reuse, rather than
        # recalculating each time
        #
        # Initially the dictionary will be empty, and each time a new series length is seen the dict will be updated. Next time that length
        # is used the dict will have an entry so can simply reuse
        possible_candidates_per_series_length = {}

        # for timing the extraction when contracting
        start_time = time.time()
        time_taken = lambda : time.time()-start_time

        # a flag to indicate if extraction should stop (either contract has ended or we've visited all required cases)
        continue_extraction = True

        idx = 0
        while continue_extraction:
            for series_id_and_class in idxs_to_sample:
                series_id = series_id_and_class[0]
                idx+=1
                if self.verbose:
                    if self.time_limit_on is False:
                        print("visiting series: "+str(series_id)+" (#"+str(idx)+"/"+str(self.cases_to_sample)+")")
                    else:
                        print("visiting series: "+str(series_id)+" (#"+str(idx)+")")
                this_series_len = len(X[series_id])

                # The bound on possible shapelet lengths will differ series-to-series if using unequal length data.
                # However, shapelets cannot be longer than the series, so set to the minimum of the series length
                # and max shapelet length (which is inf by default, unless set)
                this_shapelet_length_upper_bound = min(this_series_len,self.max_shapelet_length)

                series_shapelets = []

                # enumerate all possible candidate starting positions and lengths.
                # First, try to reuse if they have been calculated for a series of the same length before
                candidate_starts_and_lens = possible_candidates_per_series_length.get(this_series_len)
                # else calculate them for this series length and store for possible use again
                if candidate_starts_and_lens is None:
                    candidate_starts_and_lens = [
                        [start, length] for start in range(0, this_series_len - self.min_shapelet_length + 1)
                        for length in range(self.min_shapelet_length, this_shapelet_length_upper_bound + 1) if start + length <= this_series_len]
                    possible_candidates_per_series_length[this_series_len] = candidate_starts_and_lens

                # from the possible start and lengths, sample without replacement the specified number of shapelets and evaluate
                cand_idx = list(rand.choice(list(range(0,len(candidate_starts_and_lens))), self.num_shapelets_to_sample_per_case, replace=False))
                cands = [candidate_starts_and_lens[x] for x in cand_idx]

                # evaluate each candidate
                for candidate_info in cands:
                    # for convenience, extract candidate data from series_id and znorm it
                    candidate = X[series_id][candidate_info[0]:candidate_info[0]+candidate_info[1]]

                    candidate = RandomShapeletTransform.zscore(candidate)

                    # now go through all other series and get a distance from the candidate to each
                    loop_dists = []
                    for i in range(0,len(X)):
                        # don't calculate distance to self
                        if i != series_id:

                            min_dist = np.inf
                            comparison = X[i] # comparison series

                            # two implementations for comparing a candidate to all series - for loop and list comprehension (lc).
                            # Timing results to follow but there seems to be very marginal difference from informal testing

                            # for loop vs list comprehension
                            # loop
                            for start in range(0, len(comparison)-candidate_info[1]):
                                comp = X[i][start:start+candidate_info[1]]

                                comp = RandomShapeletTransform.zscore(comp)

                                dist = sqeuclidean(candidate, comp)
                                if dist < min_dist:
                                    min_dist = dist
                            if self.use_binary_info_gain:
                                # if doing binary info gain we need to make it a 1 vs all encoding
                                # if this series is from the same class as the candidate, add to the orderline with the class value
                                if y[i]==series_id_and_class[1]:
                                    loop_dists.append((min_dist, y[i]))
                                # else, the series came from another class so combine into an "other" class:
                                else:
                                    loop_dists.append((min_dist, 'otherClassForBinary'))

                            else:
                                loop_dists.append((min_dist,y[i]))

                            #lc - does the same as above without a for loop
                            # a = np.min([sqeuclidean(candidate, zscore(X[i][start:start + candidate_info[1]])) for start in range(0, len(comparison) - candidate_info[1])])
                            # loop_dists.append((a, y[i]))

                            # timing (consistent fixed params for both timing results, didn't record what they were though!)
                            # with loop:
                            # time = 6.166342258453369
                            # time = 7.506842136383057

                            # with lc:
                            # time = 5.624400854110718
                            # time = 7.173776388168335

                    loop_dists.sort(key=lambda tup: tup[0])

                    if self.use_binary_info_gain:
                        # add the class counts of the current class to the dictionary
                        # and then add num_ins-class_counts[series_id_and_class[1]] to an "other" class
                        binary_class_counts = {
                            series_id_and_class[1] :  class_counts[series_id_and_class[1]],
                            'otherClassForBinary' : num_ins-class_counts[series_id_and_class[1]]
                        }

                        # we can then simply reuse the info gain calculation without editing, but..
                        # TO-DO: implement early abandon info gain for 2 classes, as per the original Ye and Keogh shapelet paper
                        quality = self.calc_info_gain(loop_dists, binary_class_counts, num_ins)
                    else:
                        # otherwise calculate information gain for all classes vs all
                        quality = self.calc_info_gain(loop_dists, class_counts, num_ins,)

                    series_shapelets.append(Shapelet(series_id, candidate_info[0], candidate_info[1], quality, candidate))

                    if self.time_limit_on:
                        time_now = time_taken()
                        if time_now > self.time_limit:
                            if self.verbose:
                                print("time to stop! It's been "+str(round(time_now/60,3))+" minutes")
                            continue_extraction = False
                            break
                        else:
                            if self.verbose:
                                print("Candidate finished. "+str(round((self.time_limit-time_now)/60,3))+" minutes remaining.")

                # add shapelets from this series to the collection for all
                all_shapelets.extend(series_shapelets)

                # stopping condition for contracted transform
                if continue_extraction is False:
                    break

                # stopping condition for iterative transform (i.e. num_cases_to_sample have been visited)
                if not self.time_limit_on and idx >= cases_to_sample:
                    continue_extraction = False
                    break

        # sort all shapelets by quality
        all_shapelets.sort(key=lambda x: x.info_gain, reverse=True)

        # moved to end as it is now possible to visit the same series multiple times, and a better series may be found in the second visit that removes
        # the best from the first (and then means previously similar shapelets with that may again be eligible)
        if self.remove_self_similar:
            all_shapelets = RandomShapeletTransform.remove_self_similar(all_shapelets)

        self.shapelets = all_shapelets

        return self

    # two "self-similar" shapelets are subsequences from the same series that are overlapping. This method
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
        # shapelets may be False. However, could be dangerous if something else uses this code later. TO-DO: decide the best place to sort
        #
        # shapelet_list.sort(key=lambda x: x.info_gain, reverse=True)

        def is_self_similar(shapelet_one, shapelet_two):
            # not self similar if from different series
            if shapelet_one.series_id != shapelet_two.series_id:
                return False

            if (shapelet_one.start_pos >= shapelet_two.start_pos) and (shapelet_one.start_pos <= shapelet_two.start_pos + shapelet_two.length):
                return True
            if (shapelet_two.start_pos >= shapelet_one.start_pos) and (shapelet_two.start_pos <= shapelet_one.start_pos + shapelet_one.length):
                return True

        to_return = [shapelet_list[0]] # first shapelet must be ok
        for s in range(1,len(shapelet_list)):
            can_add = True
            for c in range(0, s):
                if is_self_similar(shapelet_list[s],shapelet_list[c]):
                    can_add = False
                    break
            if can_add:
                to_return.append(shapelet_list[s])

        return to_return

    # transform a set of data into distances to each extracted shapelet
    def transform(self, X, y=None, **transform_params):
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
        start = time.time()
        if self.shapelets is None:
            raise Exception("Fit not called yet - no shapelets exist!")
        X = np.array([np.asarray(x) for x in X.iloc[:, self.dim_to_use]])
        output = np.zeros([len(X),len(self.shapelets)],dtype=np.float32,)

        for i in range(0, len(X)):
            this_series = X[i]

            for s in range(0,len(self.shapelets)):
                # find distance between this series and each shapelet
                min_dist = np.inf
                this_shapelet_length = self.shapelets[s].length

                for start_pos in range(0, len(this_series) - this_shapelet_length):
                    comp = this_series[start_pos:start_pos + this_shapelet_length]
                    comp = RandomShapeletTransform.zscore(comp)

                    dist = sqeuclidean(self.shapelets[s].data, comp)
                    if dist < min_dist:
                        min_dist = dist
                try:
                    output[i][s] = min_dist.astype(np.float32)
                except Exception:
                    output[i][s] = np.float32(min_dist)

        end = time.time()
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
            self.fit(X,y)
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
    def calc_info_gain(orderline, class_counts, total):
        """A static method to calculate the information gain of a candidate shapelet
        when given an orderline and class distribution of distances to a dataset.

        Parameters
        ----------
        orderline: pandas DataFrame
            The input dataframe to transform

        Returns
        -------
        output: pandas DataFrame
            The transformed dataframe in tabular format.
        """

        def entropy(ent_class_counts, all_class_count):
            ent = 0
            for class_val in ent_class_counts.keys():
                if ent_class_counts[class_val] == 0:
                    continue
                ent -= ent_class_counts[class_val]/all_class_count*np.log2(ent_class_counts[class_val]/all_class_count)
            return ent

        less_counts = {x: 0 for x in class_counts.keys()}
        more_counts = class_counts.copy()

        initial_ent = entropy(class_counts, total)

        max_ig = 0

        for split in range(0,len(orderline)-1):
            next_class = orderline[split][1]
            less_counts[next_class] += 1
            more_counts[next_class] -= 1

            left_prop = (split+1)/total
            ent_left = entropy(less_counts, split+1)

            right_prop = (total-(split+1))/total
            ent_right = entropy(more_counts, total-(split+1))

            together = initial_ent - (left_prop*ent_left + right_prop*ent_right)

            if together > max_ig:
                max_ig = together

        return max_ig

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
        a = np.asanyarray(a)
        sstd = a.std(axis=axis, ddof=ddof)

        # special case - if shapelet is a straight line (i.e. no variance), zscore ver should be np.zeroes(len(a))
        if sstd==0:
            return np.zeros(len(a))

        mns = a.mean(axis=axis)
        if axis and mns.ndim < a.ndim:
            return ((a - np.expand_dims(mns, axis=axis)) /
                    np.expand_dims(sstd, axis=axis))
        else:
            return (a - mns) / sstd


class ContractedRandomShapeletTransform(RandomShapeletTransform):
    """
    Contracted Random Shapelet Transform (extends Random Shapelet Transform).

    A transformer to extract shapelets from a training dataset, then transform any
    passed dataset into primitives using each extracted shapelet. For k shapelets,
    an input case q is transformed by calculating the distance from q to each shapelet
    and the transformed output case is composed of k features, where the kth attribute
    is the distance from q to the kth shapelet.

    This implementation of the transform is based on the findings in [2]; it uses random
    sampling of candidate shapelets, rather than a full enumeration of candidates as
    originally proposed in [1], as it has been shown that there is no significant
    decrease in accuracy but a significant reduction in runtime through this approach.

    There are two versions of the transform: RandomShapeletTransform and
    ContractedRandomShapeletTransform. For each training series visited, both
    implementations assess a specified number of candidate shapelets per series. However,
    this version uses a contracted time limit; it visits a series, extracts an explict
    number of shapelets, and then moves onto the next series. This continues until the
    time limit is breached, and if the end of the data is reached first, the algorithm
    will loop back round to the first series and extract the specified number of shapelets
    again. The time limit is specified in minutes and is a lower-bound; if a shapelet is
    being evaluated when the time limit is reached, it will complete before the extraction
    process exits.

    Parameters
    ----------
    min_shapelet_length : int (default = 3)
        The minimum candidate shapelet length
    max_shapelet_length :int (default = np.inf)
        The maximum candidate shapelet length (default = np.inf). This value is
        an upper-bound and is automatically capped to the series length if the
        specified value exceeds the length of the shapelet.
    time_limit_in_mins: float (default = 60.0)
        The contract time limit to continue shapelet extraction. This is a lower-bound as a
        shapelet candidate will not be abandoned if the time limit runs out.
    initial_num_shapelets_per_case: int (default = 15)
        The number of shapelet candidates to evaluate per series in the fit method initially.
        If there is sufficient time remaining on the contract after this number of shapelets
        have been evaluated from all training series, the search will continue at the start
        of the data and extract this many shapelets from each series again.
    seed: int (default = 0)
        To seed the shapelet discovery to ensure deterministic results across multiple runs
    dim_to_use: int (default = 0)
        Which dimension of the data to use
    remove_self_similar: bool (default = True)
        Whether to remove self-similar shapelets before transforming. A candidate shapelet
        is considered to be self-similar to another candidate if they are taken from the
        same dimension of the same training series. With this set to True, any overlapping
        candidates will be removed to preserve the best candidate (i.e. highest info gain)
    verbose: bool (default = False)
        Whether to print information during shapelet extraction
    use_binary_info_gain: bool (default = True)
        Whether to use one-vs-all information gain (as in [2]) or consider each class
        independently when calculating the information gain of a candidate (as in [1])
    trim_shapelets: bool (default = True)
        Whether to crop the final list of extracted shapelets to the top k. If specified,
        this is an upper-bound; if less than k shapelets are found then this parameter will
        have no effect
    num_shapelets_to_trim_to: int (default = 200)
        If trim_shapelets = True, this is the number of shapelets to trim to.

    Attributes
    ----------
    shapelets: list of Shapelet objects
        The list of extracted shapelets after fit is called (initially shapelets = None)


    References
    ----------
    [1] Hills, Jon, Jason Lines, Edgaras Baranauskas, James Mapp, and Anthony Bagnall.
    "Classification of time series by shapelet transformation." Data Mining and Knowledge
    Discovery 28, no. 4 (2014): 851-881.

    [2] Bostrom, Aaron, and Anthony Bagnall.
    "Binary shapelet transform for multiclass time series classification."
    In Transactions on Large-Scale Data-and Knowledge-Centered Systems XXXII,
    pp. 24-46. Springer, Berlin, Heidelberg, 2017.
    """
    def __init__(self,
                 min_shapelet_length=3,
                 max_shapelet_length=np.inf,
                 initial_num_shapelets_per_case=15,
                 time_limit_in_mins = 60.0,
                 seed=0,
                 dim_to_use=0,
                 remove_self_similar=True,
                 verbose=False,
                 use_binary_info_gain=True,
                 trim_shapelets = True,
                 num_shapelets_to_trim_to = 200
                 ):

        self.time_limit_on = True
        self.time_limit = 60*time_limit_in_mins
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.num_cases_to_sample = -1
        self.num_shapelets_to_sample_per_case = initial_num_shapelets_per_case
        self.seed = seed
        self.dim_to_use = dim_to_use
        self.remove_self_similar = remove_self_similar
        self.shapelets = None
        self.verbose = verbose
        self.use_binary_info_gain = use_binary_info_gain
        self.trim_shapelets = trim_shapelets
        self.num_shapelets_to_trim_to = num_shapelets_to_trim_to


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

    def __init__(self, series_id, start_pos, length, info_gain, data):
        self.series_id = series_id
        self.start_pos = start_pos
        self.length = length
        self.info_gain = info_gain
        self.data = data

    def __str__(self):
        return "series id: " + str(self.series_id) + ", start_pos: " + str(self.start_pos) + ", length: " \
               + str(self.length) + ", info_gain: " + str(self.info_gain)
