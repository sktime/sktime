""" Shapelet Transform Classifier
Hardcoded implementation of of a shapelet transform classifier pipeline that simply performs a (configurable) shapelet transform
then builds (by default) a random forest
"""

__author__ = "Tony Bagnall"
__all__ = ["ShapeletTransformClassifier"]

import numpy as np
import random
import sys
import pandas as pd
import time
import math

from itertools import compress
from sklearn.transformers.shapelets import *


class ShapeletTransformClassifier(BaseEstimator):

    """ Shapelet Transform Classifier
    Basic implementation along the lines of
@article{hills14shapelet,
  title={Classification of time series by shapelet transformation},
  author={J. Hills  and  J. Lines and E. Baranauskas and J. Mapp and A. Bagnall},
  journal={Data Mining and Knowledge Discovery},
  volume={28},
  number={4},
  pages={851--881},
  year={2014}
}
but with some of the refinements presented in
@article{bostrom17binary,
  author={A. Bostrom and A. Bagnall},
  title={Binary Shapelet Transform for Multiclass Time Series Classification},
  journal={Transactions on Large-Scale Data and Knowledge Centered Systems},
  volume={32},
  year={2017},
  pages={24--46}
}


    """

    def __init__(self)
        transform=shapelet
