# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""TimeCopilot forecaster interface.

TimeCopilot is an LLM-based forecasting agent that combines large language models
with time series foundation models for automated forecasting workflows.
"""

__author__ = ["kushvinth"]
__all__ = ["TimeCopilotForecaster"]

from unittest.mock import MagicMock

import pandas as pd

from sktime.base import _HeterogenousMetaEstimator
from sktime.forecasting.base import BaseForecaster


class TimeCopilotForecaster(_HeterogenousMetaEstimator, BaseForecaster):
    """TimeCopilot LLM-based forecasting agent.

    TimeCopilot combines large language models (LLMs) with time series
    foundation models (Chronos, Moirai, TimesFM, TimeGPT, etc.) for
    automated model selection and forecasting with natural language explanations.

    This wrapper integrates TimeCopilot with sktime, allowing users to:

    * Use sktime forecasters as candidate models for TimeCopilot's selection
    * Query the forecaster with natural language questions
    * Access detailed analysis and explanations of forecasts

    Parameters
    ----------
    forecasters : list of sktime forecasters, or None, default=None
        List of forecasters to use as candidates for model selection.
        If None, TimeCopilot uses its default model pool.
        Can be a list of forecasters or list of (name, forecaster) tuples.
    llm : str, default="openai:gpt-4o-mini"
        The LLM model to use for analysis and explanation.
        Uses pydantic-ai model format: ``"provider:model_name"``.

        Supported providers and example values:

        * ``"openai:gpt-4o"`` - OpenAI GPT-4o
        * ``"openai:gpt-4o-mini"`` - OpenAI GPT-4o-mini (default)
        * ``"anthropic:claude-3-5-sonnet-latest"`` - Anthropic Claude 3.5 Sonnet
        * ``"google-gla:gemini-1.5-flash"`` - Google Gemini 1.5 Flash

        See pydantic-ai documentation for full list of supported models:
        https://ai.pydantic.dev/models/
    query : str or None, default=None
        Natural language question to ask about the forecast.
        The response will be available via ``get_user_query_response()``.
    retries : int, default=3
        Number of retries for LLM API calls.
    verbose : bool, default=False
        Whether to print detailed logs during execution.
    api_key : str or None, default=None
        API key for the LLM provider. If None, will use environment variable.
        The expected environment variable depends on the provider in ``llm``:

        * OpenAI: ``OPENAI_API_KEY``
        * Anthropic: ``ANTHROPIC_API_KEY``
        * Google: ``GOOGLE_API_KEY``

        See pydantic-ai documentation for provider-specific configuration:
        https://ai.pydantic.dev/models/

    Attributes
    ----------
    forecasters_ : list of (str, forecaster) tuples
        The fitted forecasters with their names.
    result_ : TimeCopilotResult
        The full result object from TimeCopilot after fitting.
        Contains forecast, analysis, and explanations.

    See Also
    --------
    sktime.forecasting.chronos.ChronosForecaster :
        Amazon Chronos foundation model forecaster.
    sktime.forecasting.moirai_forecaster.MoiraiForecaster :
        Salesforce Moirai foundation model forecaster.

    References
    ----------
    .. [1] Garza, A., & Rosillo, R. (2025). TimeCopilot.
           arXiv preprint arXiv:2509.00616.
    .. [2] https://github.com/TimeCopilot/timecopilot
    .. [3] https://ai.pydantic.dev/models/ (pydantic-ai model specification)

    Examples
    --------
    >>> from sktime.forecasting.timecopilot import TimeCopilotForecaster
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = TimeCopilotForecaster(
    ...     llm="openai:gpt-4o-mini",
    ...     query="What is the expected trend for the next year?"
    ... )  # doctest: +SKIP
    >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
    >>> y_pred = forecaster.predict()  # doctest: +SKIP
    >>> # Access the LLM's response to the query
    >>> response = forecaster.get_user_query_response()  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        "authors": ["kushvinth"],
        "maintainers": ["kushvinth"],
        "python_dependencies": ["timecopilot"],
        # estimator type
        "y_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        "capability:exogenous": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
    }

    # for default get_params/set_params from _HeterogenousMetaEstimator
    _steps_attr = "_forecasters"
    _steps_fitted_attr = "forecasters_"

    def __init__(
        self,
        forecasters=None,
        llm="openai:gpt-4o-mini",
        query=None,
        retries=3,
        verbose=False,
        api_key=None,
        _mock=False,
    ):
        self.forecasters = forecasters
        self.llm = llm
        self.query = query
        self.retries = retries
        self.verbose = verbose
        self.api_key = api_key
        self._mock = _mock

        super().__init__()

        # Initialize forecaster tuples if forecasters provided
        if forecasters is not None:
            self._forecasters = self._check_estimators(
                forecasters,
                attr_name="forecasters",
                cls_type=BaseForecaster,
                clone_ests=False,
            )
        else:
            self._forecasters = []

    @property
    def _forecasters(self):
        """Return forecasters as name/estimator tuples."""
        if self.forecasters is None:
            return []
        return self._get_estimator_tuples(self.forecasters, clone_ests=False)

    @_forecasters.setter
    def _forecasters(self, value):
        self.forecasters = value

    def _fit(self, y, X=None, fh=None):
        """Fit the TimeCopilot forecaster.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Target time series to fit.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (not currently supported).
        fh : ForecastingHorizon
            The forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        # Handle mock mode for testing without API key
        if self._mock:
            self._fit_mock(y, fh)
            return self

        from timecopilot import TimeCopilot

        # Set API key if provided
        if self.api_key is not None:
            import os

            # Determine provider from llm string
            if self.llm.startswith("openai:"):
                os.environ["OPENAI_API_KEY"] = self.api_key
            elif self.llm.startswith("anthropic:"):
                os.environ["ANTHROPIC_API_KEY"] = self.api_key
            elif self.llm.startswith("google"):
                os.environ["GOOGLE_API_KEY"] = self.api_key

        # Convert y to the format expected by TimeCopilot
        # TimeCopilot expects: unique_id, ds, y columns
        df = self._convert_to_timecopilot_format(y)

        # Determine forecast horizon
        if fh is not None:
            fh_relative = fh.to_relative(self.cutoff)
            h = max(fh_relative._values)
        else:
            h = None

        # Initialize TimeCopilot
        tc = TimeCopilot(
            llm=self.llm,
            retries=self.retries,
        )

        # Build kwargs for forecast call
        # freq and seasonality are inferred by TimeCopilot from the data
        forecast_kwargs = {"df": df}
        if h is not None:
            forecast_kwargs["h"] = h
        if self.query is not None:
            forecast_kwargs["query"] = self.query

        # Run forecast
        self.result_ = tc.forecast(**forecast_kwargs)

        # Store fitted forecasters if any were passed
        if self.forecasters is not None:
            self.forecasters_ = self._get_estimator_tuples(
                self.forecasters, clone_ests=True
            )
        else:
            self.forecasters_ = []

        # Store the forecast DataFrame for predict
        self._fcst_df = self.result_.fcst_df

        return self

    def _fit_mock(self, y, fh):
        """Fit using mock data for testing without API key.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Target time series to fit.
        fh : ForecastingHorizon
            The forecasting horizon.
        """
        import numpy as np

        # Create mock result object
        mock_result = MagicMock()

        # Generate mock forecast
        if fh is not None:
            fh_relative = fh.to_relative(self.cutoff)
            h = max(fh_relative._values)
        else:
            h = 3

        # Use last value as naive forecast for mock
        if isinstance(y, pd.DataFrame):
            last_val = y.iloc[-1, 0]
        else:
            last_val = y.iloc[-1]

        # Create mock forecast DataFrame
        mock_fcst = pd.DataFrame(
            {
                "unique_id": ["series_0"] * h,
                "ds": pd.date_range(start=self.cutoff, periods=h + 1, freq="D")[1:],
                "mock_model": [last_val + np.random.randn() * 0.1 for _ in range(h)],
            }
        )

        mock_result.fcst_df = mock_fcst
        mock_result.output = MagicMock()
        mock_result.output.user_query_response = "Mock response for testing."
        mock_result.output.forecast_analysis = "Mock analysis for testing."
        mock_result.output.reason_for_selection = "Mock selection reason."
        mock_result.output.selected_model = "mock_model"

        self.result_ = mock_result
        self._fcst_df = mock_fcst

        # Store fitted forecasters if any were passed
        if self.forecasters is not None:
            self.forecasters_ = self._get_estimator_tuples(
                self.forecasters, clone_ests=True
            )
        else:
            self.forecasters_ = []

    def _predict(self, fh, X=None):
        """Return forecasted values.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables (not currently supported).

        Returns
        -------
        y_pred : pd.Series or pd.DataFrame
            Point predictions.
        """
        # Get forecast from stored result
        fcst_df = self._fcst_df

        # Convert from TimeCopilot format back to sktime format
        y_pred = self._convert_from_timecopilot_format(fcst_df, fh)

        return y_pred

    def get_user_query_response(self):
        """Get the response to the user query from TimeCopilot.

        Returns the LLM's natural language response to the query
        provided during initialization or fitting.

        Returns
        -------
        str or None
            The response to the user query, or None if no query was provided
            or the forecaster has not been fitted.

        Raises
        ------
        RuntimeError
            If the forecaster has not been fitted yet.

        Examples
        --------
        >>> from sktime.forecasting.timecopilot import TimeCopilotForecaster
        >>> from sktime.datasets import load_airline
        >>> y = load_airline()  # doctest: +SKIP
        >>> forecaster = TimeCopilotForecaster(
        ...     query="What is the expected growth rate?"
        ... )  # doctest: +SKIP
        >>> forecaster.fit(y, fh=[1, 2, 3])  # doctest: +SKIP
        >>> response = forecaster.get_user_query_response()  # doctest: +SKIP
        >>> print(response)  # doctest: +SKIP
        """
        if not hasattr(self, "result_"):
            raise RuntimeError(
                "Forecaster has not been fitted yet. "
                "Call fit() before accessing user_query_response."
            )

        if hasattr(self.result_, "output") and hasattr(
            self.result_.output, "user_query_response"
        ):
            return self.result_.output.user_query_response

        return None

    def get_forecast_analysis(self):
        """Get the forecast analysis from TimeCopilot.

        Returns the LLM's natural language analysis of the forecast results.

        Returns
        -------
        str or None
            The forecast analysis, or None if not available.

        Raises
        ------
        RuntimeError
            If the forecaster has not been fitted yet.
        """
        if not hasattr(self, "result_"):
            raise RuntimeError(
                "Forecaster has not been fitted yet. "
                "Call fit() before accessing forecast_analysis."
            )

        if hasattr(self.result_, "output") and hasattr(
            self.result_.output, "forecast_analysis"
        ):
            return self.result_.output.forecast_analysis

        return None

    def get_model_selection_reason(self):
        """Get the reason for model selection from TimeCopilot.

        Returns the LLM's explanation for why a particular model was selected.

        Returns
        -------
        str or None
            The reason for model selection, or None if not available.

        Raises
        ------
        RuntimeError
            If the forecaster has not been fitted yet.
        """
        if not hasattr(self, "result_"):
            raise RuntimeError(
                "Forecaster has not been fitted yet. "
                "Call fit() before accessing model_selection_reason."
            )

        if hasattr(self.result_, "output") and hasattr(
            self.result_.output, "reason_for_selection"
        ):
            return self.result_.output.reason_for_selection

        return None

    def get_selected_model(self):
        """Get the selected model name from TimeCopilot.

        Returns the name of the model that TimeCopilot selected as best.

        Returns
        -------
        str or None
            The name of the selected model, or None if not available.

        Raises
        ------
        RuntimeError
            If the forecaster has not been fitted yet.
        """
        if not hasattr(self, "result_"):
            raise RuntimeError(
                "Forecaster has not been fitted yet. "
                "Call fit() before accessing selected_model."
            )

        if hasattr(self.result_, "output") and hasattr(
            self.result_.output, "selected_model"
        ):
            return self.result_.output.selected_model

        return None

    def _convert_to_timecopilot_format(self, y):
        """Convert sktime format to TimeCopilot format.

        Parameters
        ----------
        y : pd.Series or pd.DataFrame
            Time series in sktime format.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: unique_id, ds, y
        """
        if isinstance(y, pd.DataFrame):
            # Take first column if DataFrame
            y_series = y.iloc[:, 0]
        else:
            y_series = y

        # Get series name for unique_id
        series_name = y_series.name if y_series.name is not None else "series_0"

        # Create TimeCopilot format DataFrame
        df = pd.DataFrame(
            {
                "unique_id": series_name,
                "ds": y_series.index,
                "y": y_series.values,
            }
        )

        return df

    def _convert_from_timecopilot_format(self, fcst_df, fh):
        """Convert TimeCopilot format back to sktime format.

        Parameters
        ----------
        fcst_df : pd.DataFrame
            Forecast DataFrame from TimeCopilot.
        fh : ForecastingHorizon
            The forecasting horizon.

        Returns
        -------
        pd.Series or pd.DataFrame
            Forecast in sktime format.
        """
        # Get forecast values - TimeCopilot returns with columns like:
        # unique_id, ds, and model name column
        forecast_cols = [c for c in fcst_df.columns if c not in ["unique_id", "ds"]]

        if len(forecast_cols) == 0:
            raise ValueError("No forecast column found in TimeCopilot result")

        # Use the first forecast column (selected model)
        forecast_col = forecast_cols[0]

        # Get the forecast horizon indices
        fh_abs = fh.to_absolute_index(self.cutoff)

        # Determine series name from stored y
        series_name = None
        y_stored = self._y
        if hasattr(y_stored, "columns") and len(y_stored.columns) > 0:
            series_name = y_stored.columns[0]

        # Create forecast series
        # Match the indices from fh_abs with the forecast
        y_pred = pd.Series(
            fcst_df[forecast_col].values[: len(fh_abs)],
            index=fh_abs,
            name=series_name,
        )

        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        # Use _mock=True for testing without API key
        params1 = {
            "llm": "openai:gpt-4o-mini",
            "retries": 1,
            "_mock": True,
        }

        # Test with forecasters list
        from sktime.forecasting.naive import NaiveForecaster

        params2 = {
            "llm": "openai:gpt-4o-mini",
            "forecasters": [NaiveForecaster()],
            "query": "What is the trend?",
            "retries": 1,
            "_mock": True,
        }

        return [params1, params2]
