import numpy as np
import pandas as pd

__author__ = ["Jeremy Sellier"]

"""DataProcessor Skeleton
(comment to be added)

 process input files into timeSeries DataFrames object. Must be extended and more robust.
 Ex.  df = DataProcessor().process(file_location)

 Parameters
 ----------
 itype : specify the input type (string)
 otype : specify the output variable type (array of ...)
"""


class DataProcessor:

    def __init__(self, itype='txt', otype=float):
        self.itype_ = itype
        self.otype_ = otype

    def process(self, file_loc, starting_line=1):

        if self.itype_ == 'txt':
            return self.__process_txt(file_loc, starting_line)
        else:
            raise TypeError("unrecognized input type")

    def __process_txt(self, file_loc, starting_line=1):
        x = []
        n_row = 1
        with open(file_loc) as f:
            for line in f:
                if n_row > starting_line:
                    ln = line.split(",")
                    ln = list(map(str.strip, ln))  # remove the '/n'
                    ln[-1] = (ln[-1].split(':'))[0]  # process the last endline
                    ln = np.array(ln)
                    arr = ln.astype(self.otype_)
                    x.append(arr)
                n_row += 1

        return pd.DataFrame(data=np.array(x))
