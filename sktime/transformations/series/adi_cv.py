# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Feature transformer that returns features of time series including categories."""

import pandas as pd

from sktime.transformations.base import BaseTransformer

__author__ = ["shlok191"]


class ADICVTransformer(BaseTransformer):
    """
    Classifier based on Intermittent Demand Estimates paper by Syntetos/Boylan.

    We Set the ADI threshold to 1.32 and the CV2 threshold to 0.49 by default.
    Following is the description of the parameters mentioned above.

    1. Average Demand Interval (ADI): The average time period between
    time periods with non-zero demands

    2. Variance (CV2): Variance calculated on non-zero values
    in the time series

    3. Class: Classification of time series on basis of ADI threshold
    and CV2 threshold.

    The following are the classes we classify into:

    1. Smooth: If ADI <= ADI_threshold and CV2 <= CV2_threshold
    2. Erratic: If ADI <= ADI_threshold and CV2 > CV2_threshold
    3. Intermittent: If ADI > ADI_threshold and CV2 <= CV2_threshold
    4. Lumpy: if ADI > ADI_threshold and CV2 > CV2_threshold

    Parameters
    ----------
    adi_threshold : float (default = 1.32)
        Specifies the ADI threshold utilized for classifying the time series

    cv2_threshold : float (default = 0.49)
        Specifies the CV2 threshold utilized for classifying the time series

    features : list[str] | None (default = ['adi', 'cv2', 'class'])
        Specifies all of the feature values to be calculated

    Examples
    --------
    >>> from sktime.transformations.series.adi_cv import ADICVTransformer
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> transformer = ADICVTransformer()
    >>> y_hat = transformer.fit_transform(y)

    References
    ----------
    [1]: John E. Boylan, Aris Syntetos: The Accuracy of Intermittent
    Demand Estimates. International Journal of Forecasting, 1 Apr. 2005
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "univariate-only": True,  # Demand being the only variable
        "requires_y": False,
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": False,
        "handles-missing-data": False,
        "authors": ["shlok191"],
        "maintainers": ["shlok191"],
    }

    def __init__(self, features=None, adi_threshold=1.32, cv_threshold=0.49):
        """Initialize the transformer and processes any provided parameters.

        Parameters
        ----------
        features : List[str] | None, optional
            List of features to compute. Defaults to None (all features)

        adi_threshold : float, optional
            Threshold for Average Demand Interval. Defaults to 1.32.

        cv_threshold : float, optional
            Threshold for Variance. Defaults to 0.49.

        Raises
        ------
            ValueError: If features is provided and does not
            contain 'adi','cv2', or 'class'.
        """
        self.adi_threshold = adi_threshold
        self.cv_threshold = cv_threshold
        self.features = features

        self.features_internal = features

        # Initialize the parent class
        super().__init__()

        # Checking if the features parameter is valid
        if features is not None:
            if (
                "adi" not in features
                or "cv2" not in features
                or "class" not in features
            ):
                raise ValueError(
                    "The features list must either be None or include adi "
                    + "cv2, and class as elements."
                )

        else:
            # Helpful to transform None to default list for transform function
            self.features_internal = ["adi", "cv2", "class"]

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series
            Series of time series data this transform function aims to classify

        y : Series | None, default=None
            Not required for the classification of the time series

        Returns
        -------
        X_transformed : pd.DataFrame

            The returned DataFrame consists of the columns in the features list passed
            during initialization. Specifically, the columns include (by default):

                1. Average Demand Interval (ADI)
                2. Variance (CV2)
                3. categorical class
        """
        X_non_zero = X.to_numpy().nonzero()
        X_non_zero = X.iloc[X_non_zero]

        # Calculating ADI value based on formula from paper
        adi_value = (len(X) / len(X_non_zero)) - 1

        # Calculating variance for all non-zero values
        variance = X_non_zero.var().iloc[0]
        cv2_value = variance / len(X_non_zero)

        # Calculating the class type

        adi_low = adi_value <= self.adi_threshold
        cv2_low = cv2_value <= self.cv_threshold

        if adi_low:
            if cv2_low:
                class_type = "smooth"

            else:
                class_type = "erratic"

        elif cv2_low:
            class_type = "intermittent"

        else:
            class_type = "lumpy"

        # Collecting all values together into dict and converting to DF
        return_dict = {}

        if "adi" in self.features_internal:
            return_dict["adi"] = [adi_value]

        if "cv2" in self.features_internal:
            return_dict["cv2"] = [cv2_value]

        if "class" in self.features_internal:
            return_dict["class"] = [class_type]

        df = pd.DataFrame(return_dict)

        # Ordering the dataframe in the correct order
        df = df.loc[:, self.features_internal]

        return df

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.
            If no special parameters are defined for a value, will return
            `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test case
        """
        # Testing with 0 thresholds for both thresholds
        # in independent test cases!

        params = [
            {"features": None, "adi_threshold": 1.32, "cv_threshold": 0.49},
        ]

        return params
