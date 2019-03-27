import numpy as np
from sktime.transformers.Transformer import Transformer
import LoadData as ld
#from sklearn.base import BaseEstimator
#import pandas as pd


class PowerSpectrumTransformer(Transformer):

    def __init__(self, maxLag=100):
        self._maxLag = maxLag

    def transform(self, X):
        n_samps, self._num_atts = X.shape
        transformedX = np.empty(shape=(n_samps,int(self._num_atts/2)))
        for i in range(0,n_samps):
            transformedX[i] = self.ps(X[i])
        return transformedX

    def ps(self,x):
        fft=np.fft.fft(x)
        fft=fft.real*fft.real+fft.imag*fft.imag
        fft=fft[:int(self._num_atts/2)]
        return np.array(fft)

if __name__ == "__main__":
    print("Test for PS, to be verified vs Java version")
    d=[1,1,3,4,5,6,7,8,9,10]

    y=np.fft.fft(d)
    print(y)
    print(" FIRST TERM")
    print(y[0])
    print(" SHAPE")
    print(y.shape)
    y2=y.real*y.real+y.imag+y.imag
    print(y2)
    print(" Y[1] real")
    print(y[1].real)
    print(" Y[1] imag")
    print(y[1].imag)

    problem_path = "E:/TSCProblems/"
    results_path="E:/Temp/"
    dataset_name="ItalyPowerDemand"
    suffix = "_TRAIN.arff"
    train_x, train_y = ld.load_csv(problem_path + "/"+dataset_name + "/"+dataset_name+ suffix)
    ps=PowerSpectrumTransformer()
    trans_x=ps.transform(train_x)
    with open(results_path + dataset_name+"PS_Python.csv", "w") as f:
        f.write(dataset_name)
#        f.write(",maxLag,")
#        f.write(str(acf._lag))
        f.write("\n")
        for i in range(0,trans_x.shape[0]):
            for j in range(0, trans_x.shape[1]):
                f.write(str(trans_x[i][j]))
                f.write(",")
            f.write("\n")
