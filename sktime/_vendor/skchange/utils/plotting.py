"""Plotting utilities."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..anomaly_detectors.base import BaseSegmentAnomalyDetector
from ..change_detectors.base import BaseChangeDetector


def _plot_time_series(
    df: pd.DataFrame,
    data_repr: str = "line",
    **kwargs,
) -> go.Figure:
    """Create a Plotly figure for a time series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The time series data to plot. The index should represent the time points, while
        the columns represent the values of the time series.

    data_repr : str, optional, default="line"
        The representation of the data to plot. Can be one of the following:

        * `"line"`: Line plot with different colors for each variable.
        * `"subplot-line"`: Line plot with subplots for each variable.
        * `"point"`: Scatter plot with different colors for each variable.
        * `"subplot-point"`: Scatter plot with subplots for each variable.
        * `"heatmap"`: Heatmap representation of the time series.

    **kwargs
        Additional keyword arguments to pass to the Plotly Express plotting functions.
        See the Plotly documentation for more details.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure with the time series plotted.
    """
    index_name = df.index.name if df.index.name is not None else "index"
    long_df = (
        df.stack()
        .reset_index()
        .rename({"level_0": index_name, "level_1": "variable", 0: "value"}, axis=1)
    )

    if data_repr == "line":
        fig = px.line(long_df, x=index_name, y="value", color="variable", **kwargs)
    elif data_repr == "subplot-line":
        fig = px.line(long_df, x=index_name, y="value", facet_row="variable", **kwargs)
    elif data_repr == "point":
        fig = px.scatter(long_df, x=index_name, y="value", color="variable", **kwargs)
    elif data_repr == "subplot-point":
        fig = px.scatter(
            long_df, x=index_name, y="value", facet_row="variable", **kwargs
        )
    elif data_repr == "heatmap":
        fig = px.imshow(
            df.T,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"x": "index", "y": "variable", "color": "value"},
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown data representation: {data_repr}."
            " Must be one of: 'line', 'scatter', 'heatmap'."
        )

    return fig


def _get_data_repr(
    data_repr: str | None,
    n_variables: int,
    max_variables_for_line_plot: int = 10,
):
    """Determine the data representation for plotting.

    Parameters
    ----------
    data_repr : str or None
        The representation of the data to plot. Can be one of the following:

        * `"line"`: Line plot with different colors for each variable.
        * `"subplot-line"`: Line plot with subplots for each variable.
        * `"point"`: Scatter plot with different colors for each variable.
        * `"subplot-point"`: Scatter plot with subplots for each variable.
        * `"heatmap"`: Heatmap representation of the time series.

        If None, the function will choose "heatmap" if the number of variables is
        greater than `max_variables_for_line_plot`, and "subplot-line" otherwise.

    n_variables : int
        The number of variables (columns) in the DataFrame.

    max_variables_for_line_plot : int
        The maximum number of variables (columns) in the DataFrame to use line plots
        instead of heatmaps when `data_repr` is None.

    Returns
    -------
    str
        The determined data representation.
    """
    if data_repr is None:
        data_repr = (
            "heatmap" if n_variables > max_variables_for_line_plot else "subplot-line"
        )
    return data_repr


def plot_detections(
    df: pd.DataFrame,
    detections: pd.DataFrame,
    data_repr: str | None = None,
    **kwargs,
) -> go.Figure:
    """Plot detected change points or segment anomalies on a time series.

    Parameters
    ----------
    df : pd.DataFrame
        The time series data to plot. The index should represent the time points, while
        the columns represent the values of the time series.

    detections : pd.DataFrame
        The detections to plot, on the format returned by detectors' `predict` method.

    data_repr : str
        The representation of the data to plot. Can be one of the following:

        * `"line"`: Line plot with different colors for each variable.
        * `"subplot-line"`: Line plot with subplots for each variable.
        * `"point"`: Scatter plot with different colors for each variable.
        * `"subplot-point"`: Scatter plot with subplots for each variable.
        * `"heatmap"`: Heatmap representation of the time series.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure with the time series and highlighed detected events in red.
    """
    df = df.copy()

    n_vars = df.shape[1]
    data_repr = _get_data_repr(data_repr, n_vars)
    fig = _plot_time_series(df, data_repr, **kwargs)
    n_subplots = len([k for k in fig.layout if k.startswith("yaxis")])

    has_affected_components = "icolumns" in detections.columns
    visual_cpt_adjustment = -0.5 if data_repr == "heatmap" else 0
    for event in detections.itertuples():
        if n_subplots > 1:
            columns = event.icolumns if has_affected_components else range(df.shape[1])
        else:
            columns = [0]

        for col in columns:
            if isinstance(event.ilocs, int):
                fig.add_vline(
                    x=event.ilocs + visual_cpt_adjustment,
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    row=n_subplots - col,
                    col=1,
                )
            elif isinstance(event.ilocs, pd.Interval):
                fig.add_vrect(
                    x0=event.ilocs.left,
                    x1=event.ilocs.right,
                    fillcolor="red",
                    opacity=0.2,
                    line_width=0,
                    layer="below",
                    row=n_subplots - col,
                    col=1,
                )
            else:
                raise ValueError(
                    "The 'ilocs' column in detections must contain either integers "
                    "or pd.Intervals."
                )
    return fig


def _get_segment_ranges(
    segment_labels: pd.DataFrame, anomalies: bool = False
) -> pd.Series:
    """Create a string representation of segment ranges."""
    segment_ranges = (
        segment_labels.reset_index()
        .groupby("labels")
        .agg(start_inclusive=("index", "first"), end_inclusive=("index", "last"))
        .reset_index()
    )
    segment_ranges["end_exclusive"] = segment_ranges["end_inclusive"] + 1
    segment_ranges["range"] = segment_ranges.apply(
        lambda row: f"[{int(row['start_inclusive'])}, {int(row['end_exclusive'])})",
        axis=1,
    )

    out_ranges = segment_ranges.set_index("labels")["range"]
    if anomalies:
        out_ranges.loc[0] = "no anomaly"
    return out_ranges


def plot_scatter_segmentation(
    df: pd.DataFrame, detections: pd.DataFrame, x_var=None, y_var=None
):
    """Plot segmentation of a time series.

    Parameters
    ----------
    df : pd.DataFrame
        The time series data to plot. The index should represent the time points, while
        the columns represent the values of the time series.

    detections : pd.DataFrame
        The detections, on the format returned by detectors' `predict` method.

    x_var : str, optional
        The name of the column to use for the x-axis. If None, the first column will be
        used.

    y_var : str, optional
        The name of the column to use for the y-axis. If None, the second column will be
        used.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure with the time series and segments highlighted by different
        colors.
    """
    df = df.copy()
    original_columns = df.columns.tolist()
    if len(original_columns) < 2:
        raise ValueError("The DataFrame must contain at least two columns.")

    if x_var is None:
        x_var = original_columns[0]
    if y_var is None:
        y_var = original_columns[1]

    is_anomalies = isinstance(detections["ilocs"].iloc[0], pd.Interval)
    sparse_to_dense = (
        BaseSegmentAnomalyDetector.sparse_to_dense
        if is_anomalies
        else BaseChangeDetector.sparse_to_dense
    )

    segment_labels = sparse_to_dense(detections, df.index)
    segment_ranges = _get_segment_ranges(segment_labels, anomalies=is_anomalies)
    df["labels"] = segment_labels["labels"]
    df = df.join(segment_ranges, on="labels").rename(columns={"range": "segment"})
    df = df[original_columns + ["segment"]]
    fig = px.scatter(df, x=x_var, y=y_var, color="segment")
    return fig
