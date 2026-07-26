"""Fractional differentiation of S&P 500."""

import sys

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader
import seaborn

sys.path.append("../..")
from fracdiff import Fracdiff  # noqa: E402


def fetch_spx():
    """Fetch 'Adj Close' value of Yahoo stocks."""
    return pandas_datareader.data.DataReader(
        "^GSPC", "yahoo", "1999-10-01", "2020-09-30"
    )["Adj Close"]


if __name__ == "__main__":
    s = fetch_spx()

    f = Fracdiff(0.5, window=100, mode="valid")
    d = f.fit_transform(s.values.reshape(-1, 1)).reshape(-1)

    s = s[100 - 1 :]
    d = pd.Series(d, index=s.index)

    seaborn.set_style("white")
    fig, ax_s = plt.subplots(figsize=(16, 8))
    ax_d = ax_s.twinx()
    plot_s = ax_s.plot(s, color="blue", linewidth=0.6, label="S&P 500 (left)")
    plot_d = ax_d.plot(
        d, color="orange", linewidth=0.6, label="S&P 500, 0.5th differentiation (right)"
    )
    plots = plot_s + plot_d
    plt.title("S&P 500 and its fractional differentiation")
    ax_s.legend(plots, [p.get_label() for p in (plots)], loc=0)
    plt.savefig("spx.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
