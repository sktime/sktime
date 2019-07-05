import sktime.contrib.transformers.spectral_based as tr
import sktime.contrib.transformers.reshape_transformer  as resize
import sktime.contrib.transformers.imputation_transformer  as imputer

from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts

def testReshapers(data_set):
    r=resize.Resizer()
    d2=r.transform(data_set)

def testImputation(data_set):
    r=imputer.Resizer()
    d2=r.transform(data_set)



def testSpectral(data_set):
    dts = tr.DiscreteFourierTransformer()
    xt1 = dts.transform(data_set)
    print(xt1.shape)

    ats = tr.AutoCorrelationFunctionTransformer()
    xt2 = ats.transform(data_set)
    print(xt2.shape)

    pts = tr.PowerSpectrumTransformer()
    xt3 = pts.transform(data_set)
    print(xt3.shape)

    cts = tr.CosineTransformer()
    xt4 = cts.transform(data_set)
    print(xt4.shape)


if __name__ == "__main__":
    print("Basic correctness checks for transformations")
    print("input data 1 to test resizing univariate")

    loc = 'C:\Users\Jeremy\PycharmProjects\transformers\sktime\datasets\data\GunPoint\GunPoint_TEST.ts'
    train,test = load_ts(loc)

