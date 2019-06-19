import numpy as np
import pandas as pd

import sktime.contrib.transformers.mats_transformer as tr
from sktime.contrib.transformers.data_processor import DataProcessor

if __name__ == "__main__":

    loc = r'C:\Users\Jeremy\PycharmProjects\transformers\sktime\datasets\data\GunPoint\GunPoint_TEST.ts'
    df = DataProcessor().process(loc, starting_line=5)
    print(df.shape)

    dts = tr.DiscreteFourierTransformer()
    xt1 = dts.transform(df)

    ats = tr.AutoCorrelationFunctionTransformer()
    xt2 = ats.transform(df)

    pts = tr.PowerSpectrumTransformer()
    xt3 = pts.transform(df)

    cts = tr.CosineTransformer()
    xt4 = cts.transform(df)
