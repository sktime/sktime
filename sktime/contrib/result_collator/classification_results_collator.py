# -*- coding: utf-8 -*-
__all__ = ["ClassificationResultCollator"]

from typing import List, Any, Union
from functools import lru_cache
import pandas as pd
import requests
import warnings

from sktime.contrib.result_collator import ResultCollator

# Types
ListOrStr = Union[List[str], str]

# Request urls
PROBLEM_REQUEST_URL = (
    "https://timeseriesclassification.com/JSON/datasetTable.json?order=asc"
)
CLASSIFIERS_REQUEST_URL = (
    "https://timeseriesclassification.com/JSON/algorithmTable.json?order=asc"
)

# Small subsection of datasets to fall back on in case server not available
FALLBACK_PROBLEM_LIST = [
    "EigenWorms",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Haptics",
    "InlineSkate",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Worms",
    "WormsTwoClass",
]
# Small subsection of classifiers to fall back on in case server not available
FALLBACK_CLASSIFIER_LIST = [
    "BOSS",
    "cBOSS",
    "S-BOSS",
    "WEASEL",
    "TSF",
    "RISE",
    "STC",
    "PF",
    "ResNet",
    "HIVE-COTE",
    "TS-CHIEF",
    "ROCKET",
    "TWED",
    "WDTW",
    "CID",
    "BOP",
    "MSM",
    "TSBF",
    "FS",
    "SAXVSM",
    "ST",
    "EE",
    "DTD_C",
    "Flat-COTE",
    "DD_DTW",
    "LPS",
    "LS",
    "DTW_F",
    "Catch22",
]

FALLBACK_DICT = {
    PROBLEM_REQUEST_URL: FALLBACK_PROBLEM_LIST,
    CLASSIFIERS_REQUEST_URL: FALLBACK_CLASSIFIER_LIST,
}


@lru_cache(maxsize=None)
def get_enum_from_url(url: str, key: str):
    """Method that is used to get a enum list from a url.

    Parameters
    ----------
    url: str
        Url to get enum values from
    key: str
        json key that will be the values

    """
    enum_arr = []
    try:
        responses: List[Any] = (ResultCollator.request_resource(url)).json()
    except (
        requests.exceptions.TooManyRedirects,
        requests.exceptions.Timeout,
        requests.exceptions.RequestException,
    ) as e:
        if url not in FALLBACK_DICT:
            raise e
        warnings.warn(
            f"\nAn error has occurred while getting the latest list of problems and or datasets. \n"
            f"To resolve this a limited fallback subset of problems and dataset will be used. \n"
            f"As a result this will not contain the most up to date results."
        )
        return FALLBACK_DICT[url]

    for response in responses:
        enum_arr.append(response[key])
    return enum_arr


class ClassificationResultCollator(ResultCollator):
    """ClassificationResultCollator collates results for classifiers from a url.

    Parameters
    ----------
    urls: List[str]
        Array of urls to get results from
    classifiers: ListOrStr, defaults = "*"
        List[str] or str of classifiers to get results for. If * then will
        get results for all classifiers
    problem_list: ListOrStr, defaults = "*"
        List[str] or str of problems names to get results for. If * then will
        get results for all problem sets
    metric: str, defaults = "accuracy"
        Metric to measure classifiers over.
    resamples: int, defaults = 1
        Int that is the number of resmaples/folds. Max is 30
    toolkit: str, defaults = "sktime"
        Str that is the source of the results (i.e. results from sktime or tsml
        implementations)
    """

    # Valid parameter constants
    VALID_TOOLKITS = ["sktime", "tsml"]
    VALID_METRICS = ["accuracy", "f1"]  # TODO: find and add the rest of the results
    VALID_CLASSIFIERS: List[str] = get_enum_from_url(CLASSIFIERS_REQUEST_URL, "Acronym")
    VALID_PROBLEMS: List[str] = get_enum_from_url(PROBLEM_REQUEST_URL, "Dataset")
    VALID_DOMAIN = "https://timeseriesclassification.com"

    def __init__(
        self,
        urls: List[str],
        classifiers: ListOrStr = "*",
        problem_list: ListOrStr = "*",
        metric: str = "accuracy",
        resamples: int = 1,
        toolkit: str = "sktime",
    ):
        super(ClassificationResultCollator, self).__init__(urls)

        self.classifiers: ListOrStr = classifiers
        self.problem_list: ListOrStr = problem_list
        self.metric: str = metric
        self.resamples: int = resamples
        self.toolkit: str = toolkit

        self._classifiers: List[str] = []
        self._problem_list: List[str] = []

    def get_results(self) -> List[Any]:
        """Method used to get results by requesting and formatting response from urls.

        Returns
        -------
        formatted_response: List[Any]
            List of formatted responses
        """
        # Check domain is valid
        for url in self.urls:
            if ClassificationResultCollator.VALID_DOMAIN not in url:
                raise TypeError(
                    f"The url {url} is invalid. Please use a url "
                    f"from the domain {ClassificationResultCollator.VALID_DOMAIN}"
                )

        # Check classifiers are valid
        self._classifiers = ClassificationResultCollator._check_valid_parameters(
            self.classifiers,
            ClassificationResultCollator.VALID_CLASSIFIERS,
            "classifier",
        )

        # Check problems requested are valid
        self._problem_list = ClassificationResultCollator._check_valid_parameters(
            self.problem_list, ClassificationResultCollator.VALID_PROBLEMS, "problem"
        )

        # Check metric is valid
        ClassificationResultCollator._check_valid_parameters(
            self.metric, ClassificationResultCollator.VALID_METRICS, "metric"
        )

        # Check resamples is valid
        if self.resamples < 1 or self.resamples > 30:
            raise TypeError(
                f"The number of resamples {self.metric} is invalid. "
                f"The number of resamples must be between 1 and 30"
            )

        # Check toolkit is valid
        self.toolkit = ClassificationResultCollator._check_valid_parameters(
            self.toolkit, ClassificationResultCollator.VALID_TOOLKITS, "toolkit"
        )

        return super(ClassificationResultCollator, self).get_results()

    def _format_result(self, response: str) -> Any:
        """Method that is used to take the result and format it for your use case.

        Parameters
        ----------
        response: str
            str response

        Returns
        -------
        formatted_response: Any
            response in the format for use case can be of any type
        """
        df = pd.DataFrame([x.split(",") for x in response.split("\n")])
        return df

    @staticmethod
    def _check_valid_parameters(
        check_values: ListOrStr, valid_list: List[str], parameter_name: str
    ) -> ListOrStr:
        """Method used to check a parameter appears in the list of valid parameters.

        Parameters
        ----------
        check_values: ListOrStr
            List or str to check values of
        valid_list: List[str]
            List to check if the parameter appears in
        parameter_name: str
            Name of parameter
        """

        def check(value):
            if value is None:
                raise TypeError(f"The parameter {parameter_name} cannot be None")
            if value not in valid_list:
                raise TypeError(
                    f"The parameter {parameter_name} of value {value} is invalid. "
                    f"Please use a value from {valid_list}"
                )

        if isinstance(check_values, str):
            if check_values == "*":
                return valid_list
            check(check_values)
        else:
            for val in check_values:
                check(val)
        return check_values
