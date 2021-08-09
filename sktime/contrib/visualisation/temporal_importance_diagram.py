#!/usr/bin/python
# -*- coding: utf-8 -*-

# Temporal importance curve diagram generator for interval forests.
# Applicable to other interval forests.
# Inputs: figure save path, also used to load in attribute/timepoint inforamtion gain from text file
#         seed, used in save path
#         number of attributes used in the forests
#         number of top attributes to plot
#
# Author: Matthew Middlehurst

import sys

import numpy as np
from matplotlib import pyplot as plt


# Temporal importance curve diagram generator for interval forests
def plot_curves(
    curves,
    curve_names,
    top_curves_shown=-1,
    plot_mean=True,
    normalise_time_points=False,
):
    num_atts = int(sys.argv[3])
    num_dims = int(sys.argv[4])

    curves = []
    names = []
    for i in range(num_atts * num_dims):
        names.append(f.readline().strip())
        curves.append(array_string_to_list_float(f.readline()))

    # find attributes to display by max information gain for any time point
    top_atts = top_curves_shown if True else num_atts
    max = [max(i) for i in curves]
    top = sorted(range(len(max)), key=lambda i: max[i], reverse=True)[:top_atts]

    top_curves = [curves[i] for i in top]
    top_names = [names[i] for i in top]

    # plot curves with highest max and mean information gain for each time point
    for i in range(0, top_atts):
        plt.plot(
            top_curves[i],
            label=top_names[i] if num_dims == 1 else top_names[i] + " " + top_dims[i],
        )
    if plot_mean and num_dims == 1:
        plt.plot(
            list(np.mean(curves, axis=0)),
            "--",
            linewidth=3,
            label="Mean Information Gain",
        )
    plt.legend(
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncol=2,
        mode="expand",
        borderaxespad=0.0,
    )
    plt.xlabel("Time Point")
    plt.ylabel("Information Gain")

    plt.savefig(sys.argv[1] + "vis" + sys.argv[2])
