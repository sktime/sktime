"""MLFLow callback for logging metrics and plots to MLFlow."""
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sktime.forecasting.callbacks.callback import Callback


class MLFlowCallback(Callback):
    """MLFlow callback for logging metrics and plots to MLFlow.

    Parameters
    ----------
    forecaster : Forecaster.
        Forecaster being evaluated.
    scores : list of sktime.performance_metrics.base.PerformanceMetric
        List of performance metrics to calculate.
    tracking_uri : str
        URI of the MLFlow tracking server.
    run_name : str
        Name of the MLFlow run.
    experiment_id : str
        ID of the MLFlow experiment.
    nested : bool
        Whether to create a nested run. If set to True, make sure to have a parent run.

    Examples
    --------
    # Import necessary libraries
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_evaluation import evaluate
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.forecasting.callbacks.mlflow import MLFlowCallback
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import PolynomialTrendForecaster
    >>> import mlflow

    # Start your mlflow server locally with `mlflow server --host 127.0.0.1 --port 8080`

    # Load data
    >>> y = load_airline()[:48]

    # Create forecaster
    >>> forecaster = NaiveForecaster(strategy="mean", sp=3)

    # Create expanding window splitter
    >>> cv = ExpandingWindowSplitter(initial_window=12, step_length=6, fh=[1, 2, 3])

    # Set MLflow tracking URI
    >>> mlflow.set_tracking_uri("http://127.0.0.1:8080")

    # Set MLflow experiment
    >>> experiment_id = mlflow.set_experiment("Airline_Models").experiment_id

    # Start parent run if nested is set to True:
    >>> with mlflow.start_run(run_name="parent_run"):
    ...     # Evaluate NaiveForecaster with MLFlowCallback
    ...     results_naive = evaluate(
    ...         forecaster=NaiveForecaster(strategy="mean", sp=3),
    ...         y=y,
    ...         cv=cv,
    ...         strategy="update",
    ...         callbacks=[MLFlowCallback(experiment_id=experiment_id, nested=True)]
    ...     )
    ...
    ...     # Evaluate PolynomialTrendForecaster with MLFlowCallback
    ...     results_polynomial = evaluate(
    ...         forecaster=PolynomialTrendForecaster(degree=2),
    ...         y=y,
    ...         cv=cv,
    ...         strategy="update",
    ...         callbacks=[MLFlowCallback(experiment_id=experiment_id, nested=True)]
    ...     )
    """

    def __init__(
        self,
        forecaster=None,
        scores=None,
        tracking_uri=None,
        run_name=None,
        experiment_id=None,
        nested=False,
    ):
        super().__init__()
        self.experiment_id = experiment_id
        self.nested = nested
        self.tracking_uri = tracking_uri
        self._forecaster = forecaster
        self.score_metrics = scores
        self.run_name = run_name

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    @property
    def forecaster(self):
        """Forecaster being evaluated."""
        return self._forecaster

    @forecaster.setter
    def forecaster(self, forecaster):
        self._forecaster = forecaster
        if not self.run_name:
            self.run_name = forecaster.__class__.__name__

    def on_iteration(self, iteration, y_pred, x, result, update=None):
        """Start MLFlow run or open existing run."""
        _, (y_train, y_test, X_train, X_test) = x
        scores = {}
        for score in self.score_metrics:
            scores[f"{score.name}"] = result[f"test_{score.name}"].iloc[0]
            mlflow.log_metric(
                f"{score.name}", value=result[f"test_{score.name}"], step=iteration
            )

        fig = self._plot_time_series(y_train, y_test, y_pred, scores)
        mlflow.log_figure(fig, f"time_series_plots/iteration_{iteration}.html")

    def on_iteration_start(self, update=None):
        """
        Log metrics and plots to MLFlow.

        Logging a histogram with all the scores.
        Logging the plots of training, prediction and true values.
        Logging all scores.
        """
        mlflow.start_run(
            run_name=self.run_name, experiment_id=self.experiment_id, nested=self.nested
        )
        mlflow.log_params(self.forecaster.get_params())

    def on_iteration_end(self, results=None):
        """Stop or close MlFlow run."""
        if isinstance(results, list):
            results = pd.concat(results, ignore_index=True)
        for score in self.score_metrics:
            fig = self._create_histogram(
                results[f"test_{score.name}"], column_name=f"test_{score.name}"
            )
            mlflow.log_figure(fig, f"histograms/{score.name}.html")
        mlflow.end_run()

    def _create_histogram(self, values, column_name):
        fig = px.histogram(values, x=column_name, title="Histogram of Scores")
        fig.update_traces(texttemplate="%{y}", textposition="outside")
        return fig

    def _plot_time_series(self, train, test, predictions, scores):
        train_trace = go.Scatter(
            x=train.index.astype(str),
            y=train.values,
            mode="lines",
            name="Train",
            line=dict(color="blue"),
        )
        test_trace = go.Scatter(
            x=test.index.astype(str),
            y=test.values,
            mode="lines",
            name="Test",
            line=dict(color="green"),
        )
        predictions_trace = go.Scatter(
            x=predictions.index.astype(str),
            y=predictions.values,
            mode="lines",
            name="Predictions",
            line=dict(color="red"),
        )

        # Create figure and add traces
        fig = go.Figure([train_trace, test_trace, predictions_trace])

        # Add score annotation
        for i, (score_name, score) in enumerate(scores.items()):
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98
                - i * 0.05,  # Top-left corner, increment y position for each score
                text=f"{score_name}: {score:.2f}",
                font=dict(color="black", size=16),
                showarrow=False,
            )

        # Update layout
        fig.update_layout(
            title="Time Series Plot",
            xaxis_title="Date",
            yaxis_title="Value",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            margin=dict(
                l=100, r=40, t=60, b=40
            ),  # Adjust margins for better visibility of annotations
            font=dict(size=16),  # Increase font size for dates and values
        )

        return fig
