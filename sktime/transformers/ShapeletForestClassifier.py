#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:01:59 2019

@author: david
"""

import time
from transformers.ensemble import ShapeletForestClassifier
from load_data import load_from_tsfile_to_dataframe

if __name__ == "__main__":
    
    dataset = "GunPoint"
    train_x, train_y = load_from_tsfile_to_dataframe("/home/david/sktime-datasets/" + dataset + "/" + dataset + "_TRAIN.ts")
    test_x, test_y = load_from_tsfile_to_dataframe("/home/david/sktime-datasets/" + dataset + "/" + dataset + "_TEST.ts")

    f = ShapeletForestClassifier(n_shapelets=1, metric="scaled_dtw", metric_params={"r": 0.1})
    c = time.time()
    f.fit(train_x, train_y)
    print(f.classes_)
    print("acc:", f.score(test_x, test_y))
    print(round(time.time() - c))