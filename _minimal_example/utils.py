from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_series_for_given_level(selected_first_level, y_train, y_test, forecasts=None):
    y_column_name = y_train.columns[0]
    second_level_values = y_train.loc[selected_first_level].index.get_level_values(0).unique()

    # Create subplots
    fig, axs = plt.subplots(
        nrows=int(np.ceil(len(second_level_values) / 2)),
        ncols=2,
        figsize=(12, 1.5 * len(second_level_values)),
    )

    y_train_selected = y_train.loc[selected_first_level]
    y_test_selected = y_test.loc[selected_first_level]

    if forecasts is not None:
        forecasts_selected = {k: forecast.loc[selected_first_level] for k, forecast in forecasts.items()}

    # Plot each purpose and remove empty subplots in one loop
    for ax, level_value in zip(axs.flatten(), second_level_values):
        y_train_selected.loc[level_value][y_column_name].rename("Train").plot(ax=ax, legend=True)
        y_test_selected.loc[level_value][y_column_name].rename("Test").plot(ax=ax, legend=True)

        if forecasts is not None:
            for name, forecast in forecasts_selected.items():
                forecast.loc[level_value][y_column_name].rename(name).plot(ax=ax, legend=True)

        ax.set_title(level_value)

    # Remove empty subplots
    for ax in axs.flatten()[len(second_level_values) :]:
        fig.delaxes(ax)

    fig.suptitle(selected_first_level)

    fig.tight_layout()
    plt.show()


def load_stallion(as_period=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv("data/stallion_data.csv")
    data["date"] = pd.to_datetime(data["date"])
    if as_period:
        data["date"] = data["date"].dt.to_period("M")
    data = data.set_index(["agency", "sku", "date"])
    y = data[["volume"]]
    X = data.drop(columns="volume")
    return X, y
