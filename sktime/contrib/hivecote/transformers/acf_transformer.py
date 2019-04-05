import numpy as np
import LoadData as ld
#import pandas as pd
#import transformers.Transformer

#https://www.statsmodels.org/0.9.0/_modules/statsmodels/tsa/stattools.html#acf

class ACFTransformer():
#    global MAX_LAG=100
#    global END_IGNORE=5
    def __init__(self, lag=100,end_terms=4):
        self._lag=lag
        self._end_terms=end_terms

    def transform(self, X):
        n_samps, self._num_atts = X.shape
        if self._lag>self._num_atts-self._end_terms:
            self._lag=self._num_atts-self._end_terms
        if self._lag < 0:
            self._lag=self._num_atts
        transformedX = np.empty(shape=(n_samps,self._lag))
        for i in range(0,n_samps):
            transformedX[i] = self.acf(X[i])
        return transformedX

    def acf(self,x):
        y = np.zeros(self._lag)
        for lag in range(1, self._lag+1):
            y[lag - 1] = np.corrcoef(x[lag:], x[:-lag])[0][1]
        return np.array(y)

if __name__ == "__main__":
    problem_path = "E:/TSCProblems/"
    results_path="E:/Temp/"
    dataset_name="ItalyPowerDemand"
    suffix = "_TRAIN.arff"
    train_x, train_y = ld.load_csv(problem_path + "/"+dataset_name + "/"+dataset_name+ suffix)
    acf=ACFTransformer()
    trans_x=acf.transform(train_x)
    with open(results_path + dataset_name+"ACF_Python.csv", "w") as f:
        f.write(dataset_name)
        f.write(",maxLag,")
        f.write(str(acf._lag))
        f.write("\n")
        for i in range(0,trans_x.shape[0]):
            for j in range(0, trans_x.shape[1]):
                f.write(str(trans_x[i][j]))
                f.write(",")
            f.write("\n")
    # print("Test for ACF, Verified vs Java version")
    # d=[1,1,3,4,5,6,7,8,9,10]
    # x=d[1:]
    # print(str(x))
    # x = d[2:]
    # print(str(x))
    # x=d[:-1]
    # print(str(x))
    # x = d[:-2]
    # print(str(x))
    # y=np.zeros(3)
    # for lag in range(1,4):
    #     temp=np.corrcoef(d[lag:],d[:-lag])[0][1]
    #     y[lag-1]=temp
    #     print(str(temp))
