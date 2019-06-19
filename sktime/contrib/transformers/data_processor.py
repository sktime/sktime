import numpy as np
import pandas as pd

__author__ = ["Jeremy Sellier"]

""" Prototype method to convert a text file into timeSerie dataFrame object
- must improve the robustness and extend the functionalities
"""


def toDF(file, dtype=float, starting_line=1):
    x = []
    n_row = 1
    with open(file) as f:
        for line in f:
            if n_row > starting_line:
                ln = line.split(",")
                ln = list(map(str.strip, ln))  # remove the '/n'
                ln[-1] = (ln[-1].split(':'))[0]  # process the last endline

                ln = np.array(ln)
                arr = ln.astype(dtype)
                x.append(arr)

            n_row += 1

    return pd.DataFrame(data=np.array(x))
