# -*- coding: utf-8 -*-
# Utilities
import numpy as np
import pandas as pd
from sktime.utils.validation.panel import check_X, check_X_y
from sktime.utils.data_container import from_nested_to_2d_array

# Tuning
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Transforms
from sktime.transformers.panel.segment import SlidingWindowSegmenter
from sktime.transformers.panel.dictionary_based._paa import PAA
from sktime.transformers.panel.dwt import DWTTransformer
from sktime.transformers.panel.slope import SlopeTransformer
from sktime.transformers.panel.summarize._extract import (
    DerivativeSlopeTransformer,
)
from sktime.transformers.panel.hog1d import HOG1DTransformer

# Classifiers
from sktime.classification.base import BaseClassifier
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier


class ShapeDTW(BaseClassifier):

    """

    The ShapeDTW classifier works by initially extracting a set of subsequences
    describing local neighbourhoods around each data point in a time series.
    These subsequences are then passed into a shape descriptor function that
    transforms these local neighbourhoods into a new representation. This
    new representation is then sent into DTW with 1-NN.

    Parameters
    ----------
    n_neighbours                : int, int, set k for knn (default =1).
    subsequence_length          : int, defines the length of the
                                  subsequences(default=sqrt(n_timepoints)).

    shape_descriptor_function   : string, defines the function to describe
                                  the set of subsequences
                                  (default = 'raw').


    The possible shape descriptor functions are as follows:

        - 'raw'                 : use the raw subsequence as the
                                  shape descriptor function.
                                - params = None

        - 'paa'                 : use PAA as the shape descriptor function.
                                - params = num_intervals_paa (default=8)

        - 'dwt'                 : use DWT (Discrete Wavelet Transform)
                                  as the shape descriptor function.
                                - params = num_levels_dwt (default=3)

        - 'slope'               : use the gradient of each subsequence
                                  fitted by a total least squares
                                  regression as the shape descriptor
                                  function.
                                - params = num_intervals_slope (default=8)

        - 'derivative'          : use the derivative of each subsequence
                                  as the shape descriptor function.
                                - params = None

        - 'hog1d'               : use a histogram of gradients in one
                                  dimension as the shape desciptor
                                  function.
                                - params = num_intervals_hog1d
                                                    (defualt=2)
                                         = num_bins_hod1d
                                                    (default=8)
                                         = scaling_factor_hog1d
                                                    (default=0.1)

        - 'compound'            : use a combination of two shape
                                  descriptors simultaneously.
                                - params = weighting_factor
                                          (default=None)
                                           Defines how to scale
                                           values of a shape
                                           descriptor.
                                           If a value is not given,
                                           this value is tuned
                                           by 10-fold cross-validation
                                           on the training data.


    shape_descriptor_functions  : string list, only applicable when the
                                  shape_descriptor_function is
                                  set to 'compound'.
                                  Use a list of shape descriptor
                                  functions at the same time.
                                  (default = ['raw','derivative'])

    metric_params               : dictionary for metric parameters
                                  (default = None).

    Notes
    _____
    ..[1] Jiaping Zhao and Laurent Itti, "shapeDTW: Shape Dynamic Time Warping",
        Pattern Recognition, 74, pp 171-184, 2018
        http://www.sciencedirect.com/science/article/pii/S0031320317303710},

    """

    def __init__(
        self,
        n_neighbors=1,
        subsequence_length=30,
        shape_descriptor_function="raw",
        shape_descriptor_functions=["raw", "derivative"],  # noqa from flake8 B006
        metric_params=None,
    ):
        self.n_neighbors = n_neighbors
        self.subsequence_length = subsequence_length
        self.shape_descriptor_function = shape_descriptor_function
        self.shape_descriptor_functions = shape_descriptor_functions
        self.metric_params = metric_params
        super(ShapeDTW, self).__init__()

    def fit(self, X, y):
        """
        Method to perform training on the classifier.

        Parameters
        ----------
        X - pandas dataframe of training data of shape [n_instances,1].
        y - list of class labels of shape [n_instances].

        Returns
        -------
        self : the shapeDTW object
        """
        # Perform preprocessing on params.
        if not (isinstance(self.shape_descriptor_function, str)):
            raise TypeError(
                "shape_descriptor_function must be an 'str'. \
                            Found '"
                + type(self.shape_descriptor_function).__name__
                + "' instead."
            )

        X, y = check_X_y(X, y, enforce_univariate=False)

        if self.metric_params is None:
            self.metric_params = {}

        # If the shape descriptor is 'compound',
        # calculate the appropriate weighting_factor
        if self.shape_descriptor_function == "compound":
            self._calculate_weighting_factor_value(X, y)

        # Fit the SlidingWindowSegmenter
        sw = SlidingWindowSegmenter(self.subsequence_length)
        sw.fit(X)
        self.sw = sw

        # Transform the training data.
        X = self._preprocess(X)

        # Fit the kNN classifier
        self.knn = KNeighborsTimeSeriesClassifier(n_neighbors=self.n_neighbors)
        self.knn.fit(X, y)
        self.classes_ = self.knn.classes_

        return self

    def _calculate_weighting_factor_value(self, X, y):
        """
        Helper function for calculating the appropriate
        weighting_factor for the compound shape descriptor.
        If a value is given, the weighting_factor is set
        as the given value. If not, its tuned via
        a 10-fold cross-validation on the training data.

        Parameters
        ----------
        X - training data in a dataframe of shape [n_instances,1]
        y - training data classes of shape [n_instances].
        """
        self.metric_params = {k.lower(): v for k, v in self.metric_params.items()}

        # Get the weighting_factor if one is provided
        if self.metric_params.get("weighting_factor") is not None:
            self.weighting_factor = self.metric_params.get("weighting_factor")
        else:
            # Tune it otherwise
            self._param_matrix = {
                "metric_params": [
                    {"weighting_factor": 0.1},
                    {"weighting_factor": 0.125},
                    {"weighting_factor": (1 / 6)},
                    {"weighting_factor": 0.25},
                    {"weighting_factor": 0.5},
                    {"weighting_factor": 1},
                    {"weighting_factor": 2},
                    {"weighting_factor": 4},
                    {"weighting_factor": 6},
                    {"weighting_factor": 8},
                    {"weighting_factor": 10},
                ]
            }

            n = self.n_neighbors
            sl = self.subsequence_length
            sdf = self.shape_descriptor_function
            sdfs = self.shape_descriptor_functions
            if sdfs is None or not (len(sdfs) == 2):
                raise ValueError(
                    "When using 'compound', "
                    + "shape_descriptor_functions must be a "
                    + "string array of length 2."
                )
            mp = self.metric_params

            grid = GridSearchCV(
                estimator=ShapeDTW(
                    n_neighbors=n,
                    subsequence_length=sl,
                    shape_descriptor_function=sdf,
                    shape_descriptor_functions=sdfs,
                    metric_params=mp,
                ),
                param_grid=self._param_matrix,
                cv=KFold(n_splits=10, shuffle=True),
                scoring="accuracy",
            )
            grid.fit(X, y)
            self.weighting_factor = grid.best_params_["metric_params"][
                "weighting_factor"
            ]

    def _preprocess(self, X):
        # private method for performing the transformations on
        # the test/training data. It extracts the subsequences
        # and then performs the shape descriptor function on
        # each subsequence.
        X = self.sw.transform(X)

        # Feed X into the appropriate shape descriptor function
        X = self._generate_shape_descriptors(X)

        return X

    def predict_proba(self, X):
        """
        Function to perform predictions on the testing data X. This function
        returns the probabilities for each class.

        Parameters
        ----------
        X - pandas dataframe of testing data of shape [n_instances,1].

        Returns
        -------
        output : numpy array of shape =
                [n_instances, num_classes] of probabilities
        """
        X = check_X(X, enforce_univariate=False)

        # Transform the test data in the same way as the training data.
        X = self._preprocess(X)

        # Classify the test data
        return self.knn.predict_proba(X)

    def predict(self, X):
        """
        Find predictions for all cases in X.
        Could do a wrap function for predict_proba,
        but this will do for now.
        ----------
        X : The testing input samples of shape [n_instances,1].

        Returns
        -------
        output : numpy array of shape = [n_instances]
        """
        X = check_X(X, enforce_univariate=False)

        # Transform the test data in the same way as the training data.
        X = self._preprocess(X)

        # Classify the test data
        return self.knn.predict(X)

    def _generate_shape_descriptors(self, data):
        """
        This function is used to convert a list of
        subsequences into a list of shape descriptors
        to be used for classification.
        """
        # Get the appropriate transformer objects
        if self.shape_descriptor_function != "compound":
            self.transformer = [self._get_transformer(self.shape_descriptor_function)]
        else:
            self.transformer = []
            for x in self.shape_descriptor_functions:
                self.transformer.append(self._get_transformer(x))
            if not (len(self.transformer) == 2):
                raise ValueError(
                    "When using 'compound', "
                    + "shape_descriptor_functions must be a "
                    + "string array of length 2."
                )

        # To hold the result of each transformer
        dataFrames = []
        col_names = [x for x in range(len(data.columns))]

        # Apply each transformer on the set of subsequences
        for t in self.transformer:
            if t is None:
                # Do no transformations
                dataFrames.append(data)
            else:
                # Do the transformation and extract the resulting data frame.
                t.fit(data)
                newData = t.transform(data)
                dataFrames.append(newData)

        # Combine the arrays into one dataframe
        if self.shape_descriptor_function == "compound":
            result = self._combine_data_frames(
                dataFrames, self.weighting_factor, col_names
            )
        else:
            result = dataFrames[0]
            result.columns = col_names

        return result

    def _get_transformer(self, tName):
        """
        Function to extract the appropriate transformer

        Parameters
        -------
        self   : the ShapeDTW object.
        tName  : the name of the required transformer.

        Returns
        -------
        output : Base Transformer object corresponding to the class
                 (or classes if its a compound transformer) of the
                 required transformer. The transformer is
                 configured with the parameters given in self.metric_params.

        throws : ValueError if a shape descriptor doesn't exist.
        """
        parameters = self.metric_params

        tName = tName.lower()

        if parameters is None:
            parameters = {}

        parameters = {k.lower(): v for k, v in parameters.items()}

        self._check_metric_params(parameters)

        if tName == "raw":
            return None
        elif tName == "paa":
            num_intervals = parameters.get("num_intervals_paa")
            if num_intervals is None:
                return PAA()
            return PAA(num_intervals)
        elif tName == "dwt":
            num_levels = parameters.get("num_levels_dwt")
            if num_levels is None:
                return DWTTransformer()
            return DWTTransformer(num_levels)
        elif tName == "slope":
            num_intervals = parameters.get("num_intervals_slope")
            if num_intervals is None:
                return SlopeTransformer()
            return SlopeTransformer(num_intervals)
        elif tName == "derivative":
            return DerivativeSlopeTransformer()
        elif tName == "hog1d":
            num_intervals = parameters.get("num_intervals_hog1d")
            num_bins = parameters.get("num_bins_hog1d")
            scaling_factor = parameters.get("scaling_factor_hog1d")

            # All 3 paramaters are None
            if num_intervals is None and num_bins is None and scaling_factor is None:
                return HOG1DTransformer()

            # 2 parameters are None
            if num_intervals is None and num_bins is None:
                return HOG1DTransformer(scaling_factor=scaling_factor)
            if num_intervals is None and scaling_factor is None:
                return HOG1DTransformer(num_bins=num_bins)
            if num_bins is None and scaling_factor is None:
                return HOG1DTransformer(num_intervals=num_intervals)

            # 1 parameter is None
            if num_intervals is None:
                return HOG1DTransformer(
                    scaling_factor=scaling_factor, num_bins=num_bins
                )
            if scaling_factor is None:
                return HOG1DTransformer(num_intervals=num_intervals, num_bins=num_bins)
            if num_bins is None:
                return HOG1DTransformer(
                    scaling_factor=scaling_factor, num_intervals=num_intervals
                )

            # All parameters are given
            return HOG1DTransformer(
                num_intervals=num_intervals,
                num_bins=num_bins,
                scaling_factor=scaling_factor,
            )
        else:
            raise ValueError("Invalid shape desciptor function.")

    def _check_metric_params(self, parameters):
        """
        Helper function for checking if a user has entered in
        an invalid metric_params.
        """
        valid_metric_params = [
            "num_intervals_paa",
            "num_levels_dwt",
            "num_intervals_slope",
            "num_intervals_hog1d",
            "num_bins_hog1d",
            "scaling_factor_hog1d",
            "weighting_factor",
        ]

        names = list(parameters.keys())

        for x in names:
            if not (x in valid_metric_params):
                raise ValueError(
                    x
                    + " is not a valid metric parameter."
                    + "Make sure the shape descriptor function"
                    + " name is at the end of the metric "
                    + "parameter name."
                )

    def _combine_data_frames(self, dataFrames, weighting_factor, col_names):
        """
        Helper function for the shape_dtw class to combine two dataframes
        together into a single dataframe.
        Used when the shape_descriptor_function is set to "compound".
        """
        first_desc = dataFrames[0]
        second_desc = dataFrames[1]

        first_desc_array = []
        second_desc_array = []

        # Convert the dataframes into arrays
        for x in first_desc.columns:
            first_desc_array.append(
                from_nested_to_2d_array(first_desc[x], return_numpy=True)
            )

        for x in second_desc.columns:
            second_desc_array.append(
                from_nested_to_2d_array(second_desc[x], return_numpy=True)
            )

        # Concatenate the arrays together
        res = []
        for x in range(len(first_desc_array)):
            dim1 = []
            for y in range(len(first_desc_array[x])):
                dim2 = []
                dim2.extend(first_desc_array[x][y])
                dim2.extend(second_desc_array[x][y] * weighting_factor)
                dim1.append(dim2)
            res.append(dim1)

        res = np.asarray(res)

        # Convert to pandas dataframe
        df = pd.DataFrame()

        for col in col_names:
            colToAdd = []
            for row in range(len(res[col])):
                inst = res[col][row]
                colToAdd.append(pd.Series(inst))
            df[col] = colToAdd

        return df
