from sktime.load_data import load_from_web_to_xdataframe
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
import time


class DynamicTimeWarping1NNTSClassifier(BaseEstimator):
    """ A 1-nearest neighbour classifier using dynamic time warping for time series data.

        NOTE: prototype code - thorough testing required
        TO-DO:
            add k-NN (currently 1-NN for simple testing)
            add type checks
    """

    def __init__(self, window_width=1.0, use_independent_dimension_distances=True):
        """ A constructor for a nearest neighbour classifier using dynamic time warping.
            Parameters
            ----------
            window_width : float
                The proportion of warping that is allowed between two series
                0 = Euclidean distance, 1 = full warping
            use_independent_dimension_distances : boolean
                Determine whether distances should be calculated independently for each dimension
                in multi-variate problems. True allows unique warping paths for each dimension,
                False uses a single path across all dimensions for calculating distances
            """
        self.window_width=window_width
        self.x_train = None
        self.y_train = None
        self.is_fitted_ = False
        self.use_independent_dimension_distances = use_independent_dimension_distances

    def distance(self, first, second, cutoff=None, use_single_dimension=True, single_dimension_to_use=0) -> float:
        """ For calculating the dynamic time warping distance between two Series. This can be two Series of the
            underlying data, or two Series containing Series of all dimensions of a case (e.g. a slice of a dataframe)
            Parameters
            ----------
            first : pandas.Series
                The first case in the comparison
            second : pandas.Series
                The second case in the comparison
            use_single_dimension : boolean
                Allows the distance to be calculated for a single dimension in multi-dimensional problems, if desired
            single_dimension_to_use : int
                If using a single dimension of a multivariate problem, this is the dimension to use when
                computing distances
            """
        window_proportion = self.window_width

        n = len(first[single_dimension_to_use])
        m = len(second[single_dimension_to_use])

        warp_matrix = np.full([n, m], np.inf)
        if n > m:
            window_size = n * window_proportion
        else:
            window_size = m * window_proportion
        window_size = int(window_size)

        dist = lambda x1, x2: ((x1 - x2) ** 2)

        if use_single_dimension:
            pairwise_distances = np.asarray([[dist(x1, x2) for x2 in second[single_dimension_to_use]] for x1 in first[single_dimension_to_use]])
        else:
            # distance matrix for first dimension
            pairwise_distances = np.asarray([[dist(p1, p2) for p2 in second[0]] for p1 in first[0]])
            # add distances for dimensions 1 to d-1 for d dimensions
            if len(first) > 1:
                for dim in range(1, len(first)):
                    pairwise_distances += np.asarray([[dist(p1, p2) for p2 in second[dim]] for p1 in first[dim]])

        # initialise edges of the warping matrix
        warp_matrix[0][0] = pairwise_distances[0][0]
        for i in range(1, window_size):
            warp_matrix[0][i] = pairwise_distances[0][i] + warp_matrix[0][i - 1]
            warp_matrix[i][0] = pairwise_distances[i][0] + warp_matrix[i - 1][0]

        # now visit all allowed cells, calculate the value as the distance in this cell + min(top, left, or top-left)
        # traverse each row,
        for row in range(1, n):
            cutoff_beaten = False
            # traverse left and right by the allowed window
            for column in range(row - window_size, row + 1 + window_size):
                if column < 1 or column >= m:
                    continue

                # find smallest entry in the warping matrix, either above, to the left, or diagonally left and up
                above = warp_matrix[row - 1][column]
                left = warp_matrix[row][column - 1]
                diag = warp_matrix[row-1][column-1]

                # add the pairwise distance for [row][column] to the minimum of the three possible potential cells
                warp_matrix[row][column] = pairwise_distances[row][column]+np.min([above, left, diag])

                # check for evidence that cutoff has been beaten on this row (if using)
                if cutoff is not None and warp_matrix[row][column] < cutoff:
                    cutoff_beaten = True

            # if using a cutoff, at least one calculated value on this row MUST be less than the cutoff otherwise the
            # final distance is guaranteed not to be less. Therefore, if the cutoff has not been beaten, early-abandon
            if cutoff is not None and cutoff_beaten is False:
                return float("inf")

        return warp_matrix[n - 1][m - 1]

    def fit(self, x, y):
        """ To initialise a DTW 1NN.
        Parameters
        ----------
        x : dataframe-like of pandas.Series objects
            The training input samples. Each column represents a dimension of the
            problem, each row represents a case/instance
        y : array-like, xpandas XdataFrame Xseries, shape (n_samples,)
            The target values (class labels)
        Returns
        -------
        self : object
            Returns self.
        """
        self.x_train = x
        self.y_train = y

        # let the model know that it is fitted
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, x):
        """ Predicting function.
        Parameters
        ----------
        X : dataframe-like of pandas.Series objects
            The test data. Each column represents a dimension of the
            problem, each row represents a case/instance
        Returns
        -------
        predictions : ndarray, shape (n_samples,)
            Returns the predicted class labels
        """
        if self.window_width < 0:
            raise Exception("Negative window value")

        predictions = []

        for test in range(0, len(x)):
            bsf_class_id = -1
            bsf_dist = float("inf")
            for train in range(0, len(self.x_train)):
                dist = 0

                # calculate distances for each dimension independently (withing specified warping window) and sum together (i.e. potentially unique warping path for each dimension)
                if self.use_independent_dimension_distances:
                    for dim in range(0,len(self.x_train.iloc[train])):
                        dist += self.distance(self.x_train.iloc[train], x.iloc[test], cutoff=bsf_dist-dist, use_single_dimension=True, single_dimension_to_use=dim)
                # otherwise, use a combined distance calculation across all dimensions (i.e. single warping path, combined pointwise matrix)
                else:
                    dist = self.distance(self.x_train.iloc[train], x.iloc[test], use_single_dimension=False, cutoff=bsf_dist)
                if dist < bsf_dist:
                    bsf_dist = dist
                    bsf_class_id = train_y[train]
            predictions.append(bsf_class_id)

        return predictions

# testing code to demonstrate the use of scikit-learn's GridSearchCV
def example_grid_search(train_x, train_y, num_folds = 10):
    classifier = DynamicTimeWarping1NNTSClassifier()

    # param_grid = {'window_width': [0.1, 0.5, 1.0], 'use_independent_dimension_distances': [True, False]}
    param_grid = {'window_width': [0.1, 0.5, 1.0]}
    grid = GridSearchCV(classifier, param_grid, cv=num_folds, scoring='accuracy')

    grid.fit(train_x, train_y)

    print(grid.cv_results_)
    print(grid.best_params_)
    print(grid.best_estimator_.get_params())
    return grid.best_estimator_


# main method includes a simple experiment
if __name__ == "__main__":
    start = time.time()
    dataset_name = "GunPoint"
    # dataset_name = "BasicMotions"
    train_x, train_y = load_from_web_to_xdataframe(dataset_name,is_train_file=True)
    test_x, test_y = load_from_web_to_xdataframe(dataset_name, is_test_file=True)

    # use the grid search example
    # trained_dtw = example_grid_search(train_x,train_y)

    # or build a simple classifier with no window optimisation
    trained_dtw = DynamicTimeWarping1NNTSClassifier()
    trained_dtw.fit(train_x,train_y)

    print("time to train: " + str(time.time() - start))
    preds = trained_dtw.predict(test_x)
    print("time to train+pred: "+str(time.time()-start))
    correct = 0
    for i in range(0,len(test_x)):
        print(str(test_y[i])+","+str(preds[i]))
        if preds[i] == test_y[i]:
            correct += 1
    print("\ncorrect: "+str(correct)+"/"+str(len(test_y)))

# Example output code - not to be included in release, obviously! Example from using grid search with GunPoint
# overall acc = 94% (timeseriesclassification.com says 0.947 with DTW 1NN and a warping window set through CV (100 param options))
#
# C:\Users\jason\Anaconda3\envs\sktime\python.exe C:/Users/jason/Documents/GitHub/sktime/sktime/classifiers/nearest_neighbour_classifiers.py
# C:\Users\jason\Anaconda3\envs\sktime\lib\site-packages\sklearn\externals\joblib\externals\cloudpickle\cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
#   import imp
# {'mean_fit_time': array([0.0007977 , 0.00089839, 0.0004988 ]), 'std_fit_time': array([0.00039885, 0.00029951, 0.0004988 ]), 'mean_score_time': array([ 4.18143818,  9.55397735, 12.51285794]), 'std_score_time': array([0.32151874, 1.13589433, 1.6812154 ]), 'param_window_width': masked_array(data=[0.1, 0.5, 1.0],
#              mask=[False, False, False],
#        fill_value='?',
#             dtype=object), 'params': [{'window_width': 0.1}, {'window_width': 0.5}, {'window_width': 1.0}], 'split0_test_score': array([0., 0., 0.]), 'split1_test_score': array([0.4, 0.4, 0.4]), 'split2_test_score': array([0.4, 0.2, 0.2]), 'split3_test_score': array([0.8, 0.8, 0.8]), 'split4_test_score': array([0.4, 0.4, 0.4]), 'split5_test_score': array([0.8, 0.6, 0.6]), 'split6_test_score': array([0.8, 0.8, 0.8]), 'split7_test_score': array([0.8, 0.8, 0.8]), 'split8_test_score': array([0.8, 0.8, 0.8]), 'split9_test_score': array([0.6, 0.6, 0.6]), 'mean_test_score': array([0.58, 0.54, 0.54]), 'std_test_score': array([0.26      , 0.26907248, 0.26907248]), 'rank_test_score': array([1, 2, 2]), 'split0_train_score': array([0.42222222, 0.42222222, 0.42222222]), 'split1_train_score': array([0.48888889, 0.48888889, 0.48888889]), 'split2_train_score': array([0.6, 0.6, 0.6]), 'split3_train_score': array([0.64444444, 0.64444444, 0.64444444]), 'split4_train_score': array([0.73333333, 0.73333333, 0.73333333]), 'split5_train_score': array([0.75555556, 0.75555556, 0.75555556]), 'split6_train_score': array([0.86666667, 0.86666667, 0.86666667]), 'split7_train_score': array([0.88888889, 0.88888889, 0.88888889]), 'split8_train_score': array([0.97777778, 0.97777778, 0.97777778]), 'split9_train_score': array([1., 1., 1.]), 'mean_train_score': array([0.73777778, 0.73777778, 0.73777778]), 'std_train_score': array([0.18850942, 0.18850942, 0.18850942])}
# {'window_width': 0.1}
# {'use_independent_dimension_distances': True, 'window_width': 0.1}
# time to train: 2190.3773922920227
# time to train+pred: 2322.36062002182
# 1,1
# 2,2
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 2,2
# 2,2
# 1,2
# 1,1
# 1,1
# 2,1
# 1,1
# 1,1
# 1,1
# 1,2
# 2,2
# 2,2
# 2,2
# 1,1
# 2,2
# 1,1
# 1,1
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 2,2
# 1,2
# 1,1
# 1,1
# 1,1
# 1,1
# 1,1
# 2,2
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 2,2
# 2,2
# 1,2
# 2,2
# 2,2
# 2,2
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 2,2
# 1,1
# 2,1
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 2,2
# 2,2
# 2,2
# 2,2
# 2,2
# 2,2
# 2,2
# 2,2
# 2,2
# 2,2
# 2,2
# 1,1
# 1,1
# 1,1
# 2,2
# 2,2
# 2,2
# 2,2
# 1,1
# 2,1
# 1,1
# 1,1
# 1,1
# 2,2
# 1,1
# 2,2
# 1,2
# 1,1
# 2,2
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 2,2
# 2,2
# 2,2
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 2,2
# 2,2
# 2,2
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 2,2
# 2,2
# 2,2
# 1,1
# 1,1
# 1,1
# 2,2
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 1,1
# 1,1
# 2,2
# 1,2
# 2,2
# 1,1
# 2,2
# 2,2
# 1,1
#
# correct: 141/150
#
# Process finished with exit code 0

