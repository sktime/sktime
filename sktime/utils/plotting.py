#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Common timeseries plotting functionality."""

__all__ = ["plot_series", "plot_correlations", "plot_windows", "plot_calibration"]
__author__ = ["mloning", "RNKuhns", "Dbhasin1", "chillerobscuro", "benheid"]

import math
from warnings import simplefilter, warn

from sktime.datatypes._check import check_is_scitype
from sktime.forecasting.model_evaluation._functions import gen_y_X_train_test_global


def plot_series(
    *series,
    labels=None,
    markers=None,
    colors=None,
    title=None,
    x_label=None,
    y_label=None,
    ax=None,
    pred_interval=None,
):
    """Plot one or more time series.

    This function allows you to plot one or more
    time series on a single figure via ``series``.
    Used for making comparisons between different series.

    The resulting figure includes the time series data plotted on a graph with
    x-axis as time by default and can be changed via ``x_label`` and
    y-axis as value of time series can be renamed via ``y_label`` and
    labels explaining the meaning of each series via ``labels``,
    markers for data points via ``markers``.
    You can also specify custom colors via ``colors`` for each series and
    add a title to the figure via ``title``.
    If prediction intervals are available add them using ``pred_interval``,
    they can be overlaid on the plot to visualize uncertainty.

    Parameters
    ----------
    series : pd.Series or iterable of pd.Series
        One or more time series
    labels : list, default = None
        Names of series, will be displayed in figure legend
    markers: list, default = None
        Markers of data points, if None the marker "o" is used by default.
        The length of the list has to match with the number of series.
    colors: list, default = None
        The colors to use for plotting each series. Must contain one color per series
    title: str, default = None
        The text to use as the figure's suptitle
    pred_interval: pd.DataFrame, default = None
        Output of ``forecaster.predict_interval()``. Contains columns for lower
        and upper boundaries of confidence interval.
    ax : matplotlib axes, optional
        Axes to plot on, if None, a new figure is created and returned

    Returns
    -------
    fig : plt.Figure
        It manages the final visual appearance and layout.
        Create a new figure, or activate an existing figure.
    ax : plt.Axis
        Axes containing the plot
        If ax was None, a new figure is created and returned
        If ax was not None, the same ax is returned with plot added

    Examples
    --------
    >>> from sktime.utils.plotting import plot_series
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> fig, ax = plot_series(y)  # doctest: +SKIP
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    from sktime.datatypes import convert_to
    from sktime.utils.validation.forecasting import check_interval_df, check_y
    from sktime.utils.validation.series import check_consistent_index_type

    for y in series:
        check_y(y)

    l_series = list(series)
    l_series = [convert_to(y, "pd.Series", "Series") for y in l_series]
    for i in range(len(l_series)):
        if isinstance(list(series)[i], pd.DataFrame):
            l_series[i].name = list(series)[i].columns[0]
        elif isinstance(list(series)[i], pd.Series):
            l_series[i].name = list(series)[i].name

    n_series = len(l_series)
    _ax_kwarg_is_none = True if ax is None else False

    # labels
    if labels is not None:
        if n_series != len(labels):
            raise ValueError(
                """There must be one label for each time series,
                but found inconsistent numbers of series and
                labels."""
            )
        legend = True
    else:
        labels = ["" for _ in range(n_series)]
        legend = False

    # markers
    if markers is not None:
        if n_series != len(markers):
            raise ValueError(
                """There must be one marker for each time series,
                but found inconsistent numbers of series and
                markers."""
            )
    else:
        markers = ["o" for _ in range(n_series)]

    for y in l_series[1:]:
        check_consistent_index_type(l_series[0].index, y.index)

    if isinstance(l_series[0].index, pd.core.indexes.period.PeriodIndex):
        from copy import deepcopy

        tmp = deepcopy(l_series)  # local copy
        l_series = tmp

    for y in l_series:
        # check index types
        if isinstance(y.index, pd.core.indexes.period.PeriodIndex):
            y.index = y.index.to_timestamp()

    # create figure if no ax provided for plotting
    if _ax_kwarg_is_none:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))

    # colors
    if colors is None:
        colors = sns.color_palette("colorblind", n_colors=n_series)

    # plot series
    for y, color, label, marker in zip(l_series, colors, labels, markers):
        # scatter if little data is available or index is not complete
        if len(y) <= 3:  # or not np.array_equal(np.arange(x[0], x[-1] + 1), x):
            ax.scatter(y.index, y.values, marker=marker, label=label, color=color, s=4)
        else:
            ax.plot(
                y.index, y.values, marker=marker, label=label, color=color, markersize=4
            )

    # Set the axes title
    if title is not None:
        ax.set_title(title, size="xx-large")

    # Label the x and y axes
    if x_label is not None:
        ax.set_xlabel(x_label)

    _y_label = y_label if y_label is not None else l_series[0].name
    ax.set_ylabel(_y_label)

    if legend:
        ax.legend()
    if pred_interval is not None:
        if isinstance(pred_interval.index, pd.core.indexes.period.PeriodIndex):
            pred_interval.index = pred_interval.index.to_timestamp()
        check_interval_df(pred_interval, l_series[-1].index)

        ax = plot_interval(ax, pred_interval)
    if _ax_kwarg_is_none:
        return fig, ax
    else:
        return ax


def plot_interval(ax, interval_df):
    """Plot prediction intervals on an existing matplotlib axes.

    This function overlays prediction intervals on an existing plot to visualize
    forecast uncertainty.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the prediction intervals to.
    interval_df : pd.DataFrame
        A multi-index DataFrame containing prediction intervals.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes with the prediction intervals added.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.utils.plotting import plot_series, plot_interval

    >>> data = load_airline()
    >>> y_train, y_test = temporal_train_test_split(data, test_size=12)

    >>> forecaster = NaiveForecaster(strategy="last")
    >>> _ = forecaster.fit(y_train)

    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> interval_df = forecaster.predict_interval(fh=fh)

    >>> y_train.index = y_train.index.to_timestamp()
    >>> y_test.index = y_test.index.to_timestamp()
    >>> interval_df.index = interval_df.index.to_timestamp()

    >>> fig, ax = plot_series(
    ...     y_train, y_test, labels=["Train", "Test"],
    ...     pred_interval=interval_df,
    ... )  # doctest: +SKIP
    >>> plot_interval(ax, interval_df)  # doctest: +SKIP

    >>> ax.set_title('Predictions with Confidence Intervals')  # doctest: +SKIP
    >>> ax.set_xlabel('Date')  # doctest: +SKIP
    >>> ax.set_ylabel('Passengers')  # doctest: +SKIP
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("seaborn")

    import seaborn as sns

    var_name = interval_df.columns.levels[0][0]

    n = len(interval_df.columns.levels[1])
    if n == 1:
        colors = [ax.get_lines()[-1].get_c()]
    else:
        colors = sns.color_palette("colorblind", n_colors=n)

    for i, cov in enumerate(interval_df.columns.levels[1]):
        ax.fill_between(
            interval_df.index,
            interval_df[var_name][cov]["lower"].astype("float64").to_numpy(),
            interval_df[var_name][cov]["upper"].astype("float64").to_numpy(),
            alpha=0.2,
            color=colors[i],
            label=f"{int(cov * 100)}% prediction interval",
        )
    ax.legend()
    return ax


def plot_lags(series, lags=1, suptitle=None):
    """Plot one or more lagged versions of a time series.

    Parameters
    ----------
    series : pd.Series
        Time series for plotting lags.
    lags : int or array-like, default=1
        The lag or lags to plot.

        - int plots the specified lag
        - array-like  plots specified lags in the array/list

    suptitle : str, default=None
        The text to use as the Figure's suptitle. If None, then the title
        will be "Plot of series against lags {lags}"

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : np.ndarray
        Array of the figure's Axe objects

    Examples
    --------
    >>> from sktime.utils.plotting import plot_lags
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> fig, ax = plot_lags(y, lags=2) # plot of y(t) with y(t-2)  # doctest: +SKIP
    >>> fig, ax = plot_lags(y, lags=[1,2,3]) # y(t) & y(t-1), y(t-2).. # doctest: +SKIP
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sktime.utils.validation.forecasting import check_y

    check_y(series)

    if isinstance(lags, int):
        single_lag = True
        lags = [lags]
    elif isinstance(lags, (tuple, list, np.ndarray)):
        single_lag = False
    else:
        raise ValueError("`lags should be an integer, tuple, list, or np.ndarray.")

    length = len(lags)
    n_cols = min(3, length)
    n_rows = math.ceil(length / n_cols)
    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(8, 6 * n_rows),
        sharex=True,
        sharey=True,
    )
    if single_lag:
        axes = ax
        pd.plotting.lag_plot(series, lag=lags[0], ax=axes)
    else:
        axes = ax.ravel()
        for i, val in enumerate(lags):
            pd.plotting.lag_plot(series, lag=val, ax=axes[i])

    if suptitle is None:
        fig.suptitle(
            f"Plot of series against lags {', '.join([str(lag) for lag in lags])}",
            size="xx-large",
        )
    else:
        fig.suptitle(suptitle, size="xx-large")

    return fig, np.array(fig.get_axes())


def plot_correlations(
    series,
    lags=24,
    alpha=0.05,
    zero_lag=True,
    acf_fft=False,
    acf_adjusted=True,
    pacf_method="ywadjusted",
    suptitle=None,
    series_title=None,
    acf_title="Autocorrelation",
    pacf_title="Partial Autocorrelation",
):
    """Plot series and its ACF and PACF values.

    Parameters
    ----------
    series : pd.Series
        A time series.

    lags : int, default = 24
        Number of lags to include in ACF and PACF plots

    alpha : int, default = 0.05
        Alpha value used to set confidence intervals. Alpha = 0.05 results in
        95% confidence interval with standard deviation calculated via
        Bartlett's formula.

    zero_lag : bool, default = True
        If True, start ACF and PACF plots at 0th lag

    acf_fft : bool,  = False
        Whether to compute ACF via FFT.

    acf_adjusted : bool, default = True
        If True, denominator of ACF calculations uses n-k instead of n, where
        n is number of observations and k is the lag.

    pacf_method : str, default = 'ywadjusted'
        Method to use in calculation of PACF.

    suptitle : str, default = None
        The text to use as the Figure's suptitle.

    series_title : str, default = None
        Used to set the title of the series plot if provided. Otherwise, series
        plot has no title.

    acf_title : str, default = 'Autocorrelation'
        Used to set title of ACF plot.

    pacf_title : str, default = 'Partial Autocorrelation'
        Used to set title of PACF plot.

    Returns
    -------
    fig : matplotlib.figure.Figure

    axes : np.ndarray
        Array of the figure's Axe objects

    Examples
    --------
    >>> from sktime.utils.plotting import plot_correlations
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> fig, ax = plot_correlations(y)  # doctest: +SKIP
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("matplotlib", "statsmodels")
    import matplotlib.pyplot as plt
    import numpy as np
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    from sktime.datatypes import convert_to
    from sktime.utils.validation.forecasting import check_y

    series = check_y(series)
    series = convert_to(series, "pd.Series", "Series")

    # Setup figure for plotting
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = fig.add_gridspec(2, 2)
    f_ax1 = fig.add_subplot(gs[0, :])
    if series_title is not None:
        f_ax1.set_title(series_title)
    f_ax2 = fig.add_subplot(gs[1, 0])
    f_ax3 = fig.add_subplot(gs[1, 1])

    # Create expected plots on their respective Axes
    plot_series(series, ax=f_ax1)
    plot_acf(
        series,
        ax=f_ax2,
        lags=lags,
        zero=zero_lag,
        alpha=alpha,
        title=acf_title,
        adjusted=acf_adjusted,
        fft=acf_fft,
    )
    plot_pacf(
        series,
        ax=f_ax3,
        lags=lags,
        zero=zero_lag,
        alpha=alpha,
        title=pacf_title,
        method=pacf_method,
    )
    if suptitle is not None:
        fig.suptitle(suptitle, size="xx-large")

    return fig, np.array(fig.get_axes())


def _check_colors(colors, n_series):
    """Verify color list is correct length and contains only colors."""
    from matplotlib.colors import is_color_like

    if n_series == len(colors) and all([is_color_like(c) for c in colors]):
        return True
    warn(
        "Color list must be same length as `series` and contain only matplotlib colors"
    )
    return False


def _get_windows(cv, y):
    """Generate cv split windows, utility function."""
    train_windows = []
    test_windows = []
    for train, test in cv.split(y):
        train_windows.append(train)
        test_windows.append(test)
    return train_windows, test_windows


def plot_folds_global_forecasting(cv, cv_global, cv_global_temporal, y):
    """Plot training and test windows for global forecasting.

    cv_global_temporal splits the Panel temporally
    before the instance split from cv_global is applied. This avoids
    temporal leakage in the global evaluation across time series.
    cv is applied on the test set of the combined application of
    cv_global and cv_global_temporal.
    The resulting train and test windows are plotted for each fold.


    Pararameters
    ----------
    cv : sktime splitter object, descendant of BaseSplitter
        Time series splitter, e.g., temporal cross-validation iterator
    cv_global : sktime splitter object, descendant of BaseSplitter
        the ``cv_global`` splitter is used to split data at instance level,
        into a global training set ``y_train``,
        and a global test set ``y_test_global``.
    cv_global_temporal : SingleWindowSplitter
        Time series splitter, e.g., temporal cross-validation iterator.
        splits the Panel temporally before the instance split from cv_global
        is applied.
    y : pd.DataFrame
        Time series to split

    Returns
    -------
    fig : matplotlib.figure.Figure
        matplotlib figure object
    axes : np.ndarray
        matplotlib axes object with the figure
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("matplotlib")

    import matplotlib.pyplot as plt

    assert len(y.columns) == 1, "y should be univariate"
    assert check_is_scitype(y, scitype="Panel"), "y should be Panel data"

    ins = list(
        gen_y_X_train_test_global(y, None, cv, None, cv_global, cv_global_temporal)
    )

    column_name = y.columns[0]

    fig, axes = plt.subplots(len(ins), 2, figsize=(16, 3 * len(ins)))
    for i, _ins in enumerate(ins):
        for idx in _ins[0].index.get_level_values(0).unique():
            _ins[0].loc[idx].rename(columns={column_name: idx}).plot(
                ax=axes[i, 0], label=idx
            )
        test_title = _ins[1].index.get_level_values(0).unique()[0]
        ax = (
            _ins[1]
            .rename(columns={column_name: "Context"})
            .droplevel(0)
            .plot(ax=axes[i, 1], title=test_title)
        )
        _ins[2].rename(columns={column_name: "Target"}).droplevel(0).plot(ax=ax)
        axes[i, 0].legend(ncol=len(_ins[0].index.get_level_values(0).unique()))
        axes[i, 1].legend(ncol=2)

    return fig, axes


def plot_windows(cv, y, title="", ax=None):
    """Plot training and test windows.

    Plots the training and test windows for each split of a time series,
    subject to an sktime time series splitter.

    x-axis: time, ranging from start to end of ``y``

    y-axis: window number, starting at 0

    plot elements: training split (orange) and test split (blue)

        dots indicate index in the training or test split
        will be plotted on top of each other if train/test split is not disjoint

    Parameters
    ----------
    y : pd.Series
        Time series to split
    cv : sktime splitter object, descendant of BaseSplitter
        Time series splitter, e.g., temporal cross-validation iterator
    title : str
        Plot title
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes on which to plot. If None, axes will be created and returned.

    Returns
    -------
    fig : matplotlib.figure.Figure, returned only if ax is None
        matplotlib figure object
    ax : matplotlib.axes.Axes
        matplotlib axes object with the figure

    Examples
    --------
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.utils.plotting import plot_windows
    >>> from sktime.datasets import load_airline
    >>> import numpy as np

    >>> fh = np.arange(1, 13)
    >>> cv = ExpandingWindowSplitter(step_length=1, fh=fh, initial_window=24)
    >>> y = load_airline()
    >>> plot_windows(cv, y.iloc[:50])  # doctest: +SKIP
    """
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("matplotlib", "seaborn")
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    simplefilter("ignore", category=UserWarning)

    _ax_kwarg_is_none = True if ax is None else False

    # create figure if no ax provided for plotting
    if _ax_kwarg_is_none:
        fig, ax = plt.subplots(figsize=plt.figaspect(0.3))

    train_windows, test_windows = _get_windows(cv, y)

    def get_y(length, split):
        # Create a constant vector based on the split for y-axis."""
        return np.ones(length) * split

    n_splits = len(train_windows)
    n_timepoints = len(y)
    len_test = len(test_windows[0])

    train_color, test_color = sns.color_palette("colorblind")[:2]

    for i in range(n_splits):
        train = train_windows[i]
        test = test_windows[i]

        ax.plot(
            np.arange(n_timepoints), get_y(n_timepoints, i), marker="o", c="lightgray"
        )
        ax.plot(
            train,
            get_y(len(train), i),
            marker="o",
            c=train_color,
            label="Window",
        )
        ax.plot(
            test,
            get_y(len_test, i),
            marker="o",
            c=test_color,
            label="Forecasting horizon",
        )
    ax.invert_yaxis()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    xtickslocs = [tick for tick in ax.get_xticks() if tick in np.arange(n_timepoints)]
    ax.set(
        title=title,
        ylabel="Window number",
        xlabel="Time",
        xticks=xtickslocs,
        xticklabels=y.iloc[xtickslocs].index,
    )
    # remove duplicate labels/handles
    handles, labels = ((leg[:2]) for leg in ax.get_legend_handles_labels())
    ax.legend(handles, labels)

    if _ax_kwarg_is_none:
        return fig, ax
    else:
        return ax


def plot_calibration(y_true, y_pred, ax=None):
    r"""Plot the calibration curve for a sample of quantile predictions.

    Visualizes calibration of the quantile predictions.

    Computes the following calibration plot:

    Let :math:`p_1, \dots, p_k` be the quantile points at which
    predictions in ``y_pred`` were queried,
    e.g., via ``alpha`` in ``predict_quantiles``.

    Let :math:`y_1, \dots, y_N` be the actual values in ``y_true``,
    and let :math:`\widehat{y}_{i,j}`, for :math:`i = 1, \dots, N, j = 1, \dots, k`
    be quantile predictions at quantile point :math:`p_j`,
    of the conditional distribution of :math:`y_i`, as contained in ``y_pred``.

    We compute the calibration indicators :math:`c_{i, j},`
    as :math:`c_{i, j} = 1, \text{ if } y_i \le \widehat{y}_{i,j} \text{ and } 0, \text{otherwise},`
    and calibration fractions as

    .. math:: \widehat{p}_j = \frac{1}{N} \sum_{i = 1}^N c_{i, j}.

    If the quantile predictions are well-calibrated, we expect :math:`\widehat{p}_j`
    to be close to :math:`p_j`.

    x-axis: interval from 0 to 1, quantile points

    y-axis: interval from 0 to 1, calibration fractions

    plot elements: calibration curve of the quantile predictions (blue) and the ideal
    calibration curve (orange), the curve with equation y = x.
        Calibration curve are points :math:`(p_i, \widehat{p}_i), i = 1 \dots, k`;

        Ideal curve is the curve with equation y = x,
        containing points :math:`(p_i, p_i)`.

    Parameters
    ----------
    y_true : pd.Series, single columned pd.DataFrame, or single columned np.array.
        The actual values
    y_pred : pd.DataFrame
        The quantile predictions,
        formatted as returned by ``BaseDistribution.quantile``,
        or ``predict_quantiles``
    ax : matplotlib.axes.Axes, optional (default=None)
        Axes on which to plot. If None, axes will be created and returned.

    Returns
    -------
    fig : matplotlib.figure.Figure, returned only if ax is None
        matplotlib figure object
    ax : matplotlib.axes.Axes
        matplotlib axes object with the figure

    Examples
    --------
    >>> import numpy as np
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.utils.plotting import plot_calibration

    >>> y_train = load_airline()[0:24]  # train on 24 months, 1949 and 1950
    >>> y_test = load_airline()[24:36]  # ground truth for 12 months in 1951

    >>> # try to forecast 12 months ahead, from y_train
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)

    >>> forecaster = NaiveForecaster(strategy="last")
    >>> forecaster.fit(y_train)  # doctest: +SKIP

    >>> pred_quantiles = forecaster.predict_quantiles(fh=fh, alpha=[0.1, 0.25, 0.5, 0.75, 0.9])  # doctest: +SKIP
    >>> plot_calibration(y_true=y_test.loc[pred_quantiles.index], y_pred=pred_quantiles)  # doctest: +SKIP
    """  # noqa: E501
    from sktime.utils.dependencies import _check_soft_dependencies

    _check_soft_dependencies("matplotlib", "statsmodels")
    import matplotlib.pyplot as plt
    import pandas as pd

    from sktime.datatypes import convert_to

    series = convert_to(y_true, "pd.Series", "Series")

    _ax_kwarg_is_none = True if ax is None else False

    if _ax_kwarg_is_none:
        fig, ax = plt.subplots(1, figsize=plt.figaspect(0.25))

    result = [0]
    ideal_calibration = [0]

    for col in y_pred.columns:
        if isinstance(col, tuple):
            q = col[1]
        else:
            q = col
        pred_q = convert_to(y_pred[[col]], "pd.Series", "Series")
        result.append(sum(series.values < pred_q.values) / len(pred_q.values))
        ideal_calibration.append(q)
    result.append(1)
    ideal_calibration.append(1)

    df = pd.DataFrame(
        {"Forecast's Calibration": result, "Ideal Calibration": ideal_calibration},
        index=ideal_calibration,
    )

    df.plot(ax=ax)

    if _ax_kwarg_is_none:
        return fig, ax
    else:
        return ax
