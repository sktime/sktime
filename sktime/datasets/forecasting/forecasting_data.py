"""Forecasting Datasets."""

from inspect import signature

from sktime.datasets import load_forecastingdata
from sktime.datasets.forecasting._base import BaseForecastingDataset


class ForecastingData(BaseForecastingDataset):
    """Forecasting dataset loader.

    Examples
    --------
    >>> from sktime.datasets import ForecastingData
    >>> dataset = ForecastingData(name="cif_2016_dataset")
    >>> y = dataset.load("y")
    """

    _tags = {
        "name": "forecasting_data",
    }

    def __init__(self, name):
        super().__init__()
        self.name = name

        self.loader_func = load_forecastingdata

    def _encode_args(self, code):
        """Decide kwargs for the loader function."""
        kwargs = {}

        if code in ["X", "y"]:
            split = None
        elif code in ["X_train", "y_train"]:
            split = "TRAIN"
        elif code in ["X_test", "y_test"]:
            split = "TEST"
        else:
            split = None

        loader_func_params = signature(self.loader_func).parameters
        init_signature_params = signature(self.__init__).parameters
        init_param_values = {k: getattr(self, k) for k in init_signature_params.keys()}

        if (
            "test" in code.lower() or "train" in code.lower()
        ) and "split" not in loader_func_params:
            raise ValueError(
                "This dataset loader does not have a train/test split. "
                "Load the full dataset instead."
            )

        if "split" in loader_func_params:
            kwargs["split"] = split

        for init_param_name, init_param_value in init_param_values.items():
            if init_param_name in loader_func_params:
                kwargs[init_param_name] = init_param_value

        return kwargs

    def _load(self, *args):
        """Load the dataset.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            "X": exogeneous time series
            "y": time series
            "X_train": training instances only, for fixed single split
            "y_train": training labels only, for fixed single split
            "X_test": test instances only, for fixed single split
            "y_test": test labels only, for fixed single split

        Returns
        -------
        dataset, if args is empty or length one
            data container corresponding to string in args (see above)
        tuple, of same length as args, if args is length 2 or longer
            data containers corresponding to strings in args, in same order
        """
        if len(args) == 0:
            args = ("X", "y")

        cache = {}

        if "X" in args or "y" in args:
            X, y = self._load_dataset(**self._encode_args("X"))
            cache["X"] = X
            cache["y"] = y
        if "X_train" in args or "y_train" in args:
            X, y = self._load_dataset(**self._encode_args("X_train"))
            cache["X_train"] = X
            cache["y_train"] = y
        if "X_test" in args or "y_test" in args:
            X, y = self._load_dataset(**self._encode_args("X_test"))
            cache["X_test"] = X
            cache["y_test"] = y
        if "cv" in args:
            cv = self._load_simple_train_test_cv_split()
            cache["cv"] = cv

        res = [cache[key] for key in args]
        return res[0] if len(res) == 1 else tuple(res)

    def _load_dataset(self, **kwargs):
        """Call loader function with self.name included automatically."""
        if "name" in signature(self.loader_func).parameters and "name" not in kwargs:
            kwargs["name"] = self.name
        output = self.loader_func(**kwargs)
        y, X = self._split_into_y_and_X(output)
        return X, y

    def _split_into_y_and_X(self, loader_output):
        """Split the output of the loader into X and y.

        Parameters
        ----------
        loader_output: any
            Output of the loader function.

        Returns
        -------
        tuple
            Tuple containing y and X.
        """
        if isinstance(loader_output, tuple):
            return loader_output

        y = loader_output
        X = None
        return (y, X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter."""
        params_list = [
            {
                "name": "cif_2016_dataset",
            },
            {
                "name": "hospital_dataset",
            },
        ]

        return params_list
