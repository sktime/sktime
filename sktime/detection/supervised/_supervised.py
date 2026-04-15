from sktime.detection.base import BaseDetector
from sktime.forecasting.base._base import _coerce_to_list
from sktime.datatypes import convert, check_is_scitype, check_is_error_msg
from sktime.utils.validation.panel import check_X, check_y

import pandas as pd

from sktime.datatypes import (
    VectorizedDF,
    check_is_scitype,
)
from sktime.forecasting.base._fh import ForecastingHorizon
from sktime.utils.validation.forecasting import check_cv
from sktime.utils.warnings import warn


class BaseSupervisedDetector(BaseDetector):
    """todo: write!
    """
    _tags = {
        # packaging info
        # --------------
        "authors": "CloseChoice",  # author(s) of the object
        "maintainers": "CloseChoice",  # current maintainer(s) of the object
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # str or list of str, package soft dependencies
        # estimator tags
        # --------------
        # todo 1.0.0 - remove series-annotator
        "object_type": ["detector", "series-annotator"],  # type of object
        "learning_type": "None",  # supervised, unsupervised
        "task": "None",  # anomaly_detection, change_point_detection, segmentation
        "capability:multivariate": False,
        "capability:missing_values": False,
        "capability:update": False,
        "capability:variable_identification": False,
        #
        # todo: distribution_type does not seem to be used - refactor or remove
        "distribution_type": "None",
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex"],
        "fit_is_empty": False,
        "y_inner_mtype": [
            # should this be a series or a dataframe?
            "pd.DataFrame",
            # "pd.Series",
        ],
    }

    def __init__(self):
        # todo: do we need anything else in here? If not we can omit this!
        super().__init__()
        self._state = "new"  # can be "new", "pretrained", "fitted"

    def _check_X_y(self, X=None, y=None, y_inner_mtype=None, multivariate=False):
        # this is basically copied from panel check_X_y but without the consistency checks, since y is not line-for-line target for X!
        # todo: we would need to add checks that y_inner_mtype is compatible and coerce appropriately
        if isinstance(y, pd.DataFrame):
            if len(y.columns) > 1:
                raise NotImplementedError(
                    "pretrain does not currently support multivariate targets. "
                    )
            else:
                y = y.iloc[:, 0]
        # just set these values somehow for now
        # enforce_univariate=False
        # enforce_min_instances=
        # enforce_min_columns=1
        coerce_to_numpy=False
        # coerce_to_pandas=False
        y = check_y(y, coerce_to_numpy=coerce_to_numpy)

        # X = check_X(
        #     X,
        #     enforce_univariate=enforce_univariate,
        #     enforce_min_columns=enforce_min_columns,
        #     enforce_min_instances=enforce_min_instances,
        #     coerce_to_numpy=coerce_to_numpy,
        #     coerce_to_pandas=coerce_to_pandas,
        # )
        return X, y

    # todo: needed?
    def _check_X(self, X):
        """Check input data.

        Parameters
        ----------
        X : pd.DataFrame, pd.Series or np.ndarray
            Data to be transformed

        Returns
        -------
        X : X_inner_mtype
            Data to be transformed
        """
        ALLOWED_SCITYPES = ["Series", "Panel"]
        X_valid, X_msg, X_metadata = check_is_scitype(
            X, scitype=ALLOWED_SCITYPES, return_metadata=[]
        )
        self._X_metadata = X_metadata
        if not X_valid:
            msg_start = (
                f"Unsupported input data type in {self.__class__.__name__}, input X"
            )
            allowed_msg = (
                "Allowed scitypes for X in detection are "
                f"{', '.join(ALLOWED_SCITYPES)}, "
                "for instance a pandas.DataFrame with sktime compatible time indices."
                " See the detection tutorial examples/07_detection.ipynb, or"
                " the data format tutorial examples/AA_datatypes_and_datasets.ipynb"
            )
            if not X_valid:
                check_is_error_msg(
                    X_msg,
                    var_name=msg_start,
                    allowed_msg=allowed_msg,
                    raise_exception=True,
                )

                
        X_inner_mtype = self.get_tag("X_inner_mtype")
        X_inner = convert(X, from_type=X_metadata["mtype"], to_type=X_inner_mtype)
        return X_inner

    def pretrain(self, X, y):
        """todo: write docsting.        """
        # pretrain always requires panel data, independent of what fit/predict
        # support via y_inner_mtype. Pass expanded mtypes directly to _check_X_y
        # to decouple pretrain's data requirements from the tag.
        _PRETRAIN_MTYPES = ["pd-multiindex", "pd_multiindex_hier"]
        orig_y_mtypes = _coerce_to_list(self.get_tag("y_inner_mtype"))
        pretrain_y_mtypes = list(set(orig_y_mtypes + _PRETRAIN_MTYPES))

        # pretrain accepts multivariate panel data even for univariate forecasters,
        # because _pretrain can split columns into separate univariate series.
        # Pass multivariate=True to prevent column vectorization.
        X_inner, y_inner = self._check_X_y(
            X=X, y=y, y_inner_mtype=pretrain_y_mtypes, multivariate=True
        )

        # pretrain does not support vectorization - global learning requires
        # the forecaster to handle panel data directly
        if isinstance(y_inner, VectorizedDF):
            raise TypeError(
                f"{type(self).__name__}.pretrain does not support automatic "
                "vectorization. Pretraining requires global learning across all "
                "instances, so the forecaster must natively support the input data."
            )

        if self._state == "new":
            self._pretrain(y=y_inner, X=X_inner)
        else:
            self._pretrain_update(y=y_inner, X=X_inner)

        if not hasattr(self, "_pretrained_attrs"):
            self._pretrained_attrs = []

        # Track new pretrained attributes (extend, not append, to avoid nested lists)
        new_attrs = [
            a
            for a in dir(self)
            if a.endswith("_")
            and not a.startswith("_")
            and a not in self._pretrained_attrs
        ]
        self._pretrained_attrs.extend(new_attrs)

        self._state = "pretrained"
        return self

    def get_pretrained_params(self, deep=True):
        """todo: write!        """
        if not hasattr(self, "_pretrained_attrs"):
            return {}

        params = {}
        for attr in self._pretrained_attrs:
            if hasattr(self, attr):
                value = getattr(self, attr)
                params[attr] = value

                # Handle nesting: if value is an estimator with pretrained params
                if deep and hasattr(value, "get_pretrained_params"):
                    nested = value.get_pretrained_params(deep=True)
                    for nested_key, nested_val in nested.items():
                        params[f"{attr}__{nested_key}"] = nested_val

        return params

    def update(self, X, y, update_params=True):
        """todo: write docstring        """
        self.check_is_fitted()

        if y is None or (hasattr(y, "__len__") and len(y) == 0):
            warn(
                f"empty y passed to update of {self}, no update was carried out",
                obj=self,
            )
            return self

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        # update internal X/y with the new X/y
        # this also updates cutoff from y
        self._update_y_X(y_inner, X_inner)

        # checks and conversions complete, pass to inner fit
        if not self._is_vectorized:
            self._update(y=y_inner, X=X_inner, update_params=update_params)
        else:
            self._vectorize("update", y=y_inner, X=X_inner, update_params=update_params)

        return self

    def update_predict(
        self,
        X,
        cv=None,
        y=None,
        update_params=True,
        reset_forecaster=True,
    ):
        """todo: write docstring        """
        from sktime.split import ExpandingWindowSplitter

        if cv is None:
            cv = ExpandingWindowSplitter(initial_window=1)

        self.check_is_fitted()

        # input checks and minor coercions on X, y
        X_inner, y_inner = self._check_X_y(X=X, y=y)

        cv = check_cv(cv)

        return self._predict_moving_cutoff(
            y=y_inner,
            cv=cv,
            X=X_inner,
            update_params=update_params,
            reset_forecaster=reset_forecaster,
        )

    def _predict(self, fh, X):
        """todo: write!        """
        raise NotImplementedError("abstract method")

    def _pretrain(self, y, X, fh=None):
        """todo: write!        """
        # the default simply discards the data, i.e., no pretraining happens
        return self
