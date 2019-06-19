import sktime.contrib.transformers.mats_transformer as tr
from sktime.contrib.transformers.data_processor import DataProcessor

if __name__ == "__main__":

    loc = r'C:\Users\Jeremy\PycharmProjects\transformers\sktime\datasets\data\GunPoint\GunPoint_TEST.ts'
    df = DataProcessor().process(loc, starting_line=5)
    print(df.shape)

    dts = tr.DiscreteFourierTransformer()
    xt1 = dts.transform(df)
    print(xt1.shape)

    ats = tr.AutoCorrelationFunctionTransformer()
    xt2 = ats.transform(df)
    print(xt2.shape)

    pts = tr.PowerSpectrumTransformer()
    xt3 = pts.transform(df)
    print(xt3.shape)

    cts = tr.CosineTransformer()
    xt4 = cts.transform(df)
    print(xt4.shape)
