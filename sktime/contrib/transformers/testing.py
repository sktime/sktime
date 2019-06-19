import sktime.contrib.transformers.mats_transformer as tr
from sktime.contrib.transformers.data_processor import toDF

if __name__ == "__main__":

    loc = r'C:\Users\Jeremy\PycharmProjects\transformers\sktime\datasets\data\GunPoint\GunPoint_TEST.ts'
    df = toDF(loc, starting_line=5)

    ts = tr.AutoCorrelationFunctionTransformer()
    xt = ts.transform(df)
