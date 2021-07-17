# -*- coding: utf-8 -*-
__all__ = [
    "classifier_list",
]
__author__ = ["bilal-196"]

from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.dictionary_based import WEASEL
from sktime.classification.dictionary_based import MUSE
from sktime.classification.distance_based import ElasticEnsemble
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.distance_based import ShapeDTW
from sktime.classification.distance_based import ProximityForest
from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.classification.interval_based import DrCIF
from sktime.classification.interval_based import RandomIntervalSpectralForest
from sktime.classification.interval_based import SupervisedTimeSeriesForest
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.shapelet_based import MrSEQLClassifier
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.classification.kernel_based import  Arsenal
from sktime.classification.kernel_based import ROCKETClassifier
from sktime.classification.hybrid import Catch22ForestClassifier
from sktime.classification.hybrid import HIVECOTEV1
import csv

"""
Main list of classifiers included to show their capabilities tags. 
For clarity, some utility classifiers, such as ColumnEnsembleClassifier, 
ComposableTimeSeriesForestClassifier, IndividualBOSS, IndividualTDE,
ProximityStump, ProximityTree, are not included as they miss such tags.
"""

classifier_list = [BOSSEnsemble, 
                   ContractableBOSS,
                   TemporalDictionaryEnsemble, 
                   WEASEL, 
                   MUSE, 
                   ElasticEnsemble, 
                   KNeighborsTimeSeriesClassifier, 
                   ShapeDTW, 
                   ProximityForest, 
                   CanonicalIntervalForest, 
                   DrCIF, 
                   RandomIntervalSpectralForest, 
                   SupervisedTimeSeriesForest, 
                   TimeSeriesForestClassifier, 
                   MrSEQLClassifier, 
                   ShapeletTransformClassifier,
                   Arsenal,
                   ROCKETClassifier, 
                   Catch22ForestClassifier, 
                   HIVECOTEV1]

with open("calssifier_tags.csv", "w+", newline='') as csvfile: 
      write = csv.writer(csvfile, delimiter=",") 
      tags = list(classifier_list[0].capabilities.keys())
      write.writerow(["Classifier Type", "Classifier", tags[0], tags[1], tags[2], tags[3], tags[4]])  
      for i, item in enumerate(classifier_list):
          attributes = list(item.capabilities.values())
          write = csv.writer(csvfile, delimiter = ",") 
          if i <= 4:
              write.writerow(["Dictionary Based", item.__name__, attributes[0], attributes[1], attributes[2], 
                              attributes[3], attributes[4]]) 
          elif i > 4 and i <= 8:
              write.writerow(["Distance Based", item.__name__, attributes[0], attributes[1], attributes[2], 
                              attributes[3], attributes[4]]) 
          elif i > 8 and i <= 13:
              write.writerow(["Interval Based", item.__name__, attributes[0], attributes[1], attributes[2], 
                              attributes[3], attributes[4]]) 
          elif i > 13 and i <= 15:
              write.writerow(["Shapelet Based", item.__name__, attributes[0], attributes[1], attributes[2], 
                              attributes[3], attributes[4]]) 
          elif i > 15 and i <= 17:
              write.writerow(["Kernel Based", item.__name__, attributes[0], attributes[1], attributes[2], 
                              attributes[3], attributes[4]])    
          elif i > 17 and i <= 19:
              write.writerow(["Hybrid Based", item.__name__, attributes[0], attributes[1], attributes[2], 
                              attributes[3], attributes[4]]) 