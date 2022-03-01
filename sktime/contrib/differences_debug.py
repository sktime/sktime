from datasets import load_from_tsfile
from distances import lcss_distance, ddtw_distance, wddtw_distance, dtw_distance
from clustering import TimeSeriesKMeans

import numpy as np
trainX, trainY = load_from_tsfile(
    "C:/Code/sktime/sktime/datasets/data/BasicMotions/BasicMotions_TRAIN.ts",
                                  return_data_type="numpy3d")

first = np.array([[1.0,2.0,3.0,4.0,5.0,6.0],[11.0,12.0,13.0,14.0,15.0,16.0]])
second = np.array([[11.0,12.0,13.0,14.0,15.0,16.0],[1.0,2.0,3.0,4.0,5.0,6.0]])
#first = trainX[0]
#second = trainX[4]
#first = np.transpose(first)
#second = np.transpose(second)
print(first.shape)
dist = lcss_distance(first, second, window=3.0/150, epsilon=1.0)
dist2 = ddtw_distance(first, second)
dist3 = wddtw_distance(first, second)
dist4 = dtw_distance(first, second)
print("LCSS DIST =", dist)
print("DDTW DIST =", dist2)
print("WDDTW DIST =", dist3)
print("DTW DIST =", dist4)

#clst = TimeSeriesKMeans(metric="ddtw")
#clst.fit(trainX)

#LCSS DIST = 1.0
#DDTW DIST = 171.64158093694977
#WDDTW DIST = 85.82079046847488
#DTW DIST = 243.26895358032104
