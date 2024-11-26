"""Hidden Markov Model based annotation from hmmlearn.

This code provides a base interface template for models
from hmmlearn for using that library for annotation of time series.

Please see the original library
(https://github.com/hmmlearn/hmmlearn/blob/main/lib/hmmlearn/hmm.py)
"""

from sktime.detection.hmm_learn.base import BaseHMMLearn

__author__ = ["miraep8"]
__all__ = ["BaseHMMLearn"]
