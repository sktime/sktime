import sys

import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader
import seaborn

sys.path.append("../..")
from fracdiff import FracdiffStat  # noqa: E402


def fetch_nky():
    return pandas_datareader.data.DataReader(
        "^N225", "yahoo", "1999-10-01", "2020-09-30"
    )["Adj Close"]


if __name__ == "__main__":
    s = fetch_nky()

    f = FracdiffStat(window=100, mode="valid")
    d = f.fit_transform(s.values.reshape(-1, 1)).reshape(-1)

    s = s[100 - 1 :]
    d = pd.Series(d, index=s.index)

    seaborn.set_style("white")
    fig, ax_s = plt.subplots(figsize=(16, 8))
    ax_d = ax_s.twinx()
    plot_s = ax_s.plot(s, color="blue", linewidth=0.6, label="Nikkei 225 (left)")
    plot_d = ax_d.plot(
        d,
        color="orange",
        linewidth=0.6,
        label="Nikkei 225, {:.2f}th differentiation (right)".format(f.d_[0]),
    )
    plots = plot_s + plot_d
    plt.title("Nikkei 225 and its fractional differentiation")
    ax_s.legend(plots, [p.get_label() for p in (plots)], loc=0)
    plt.savefig("nky.png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
