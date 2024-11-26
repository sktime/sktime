"""Information Gain-based Temporal Segmentation.

Information Gain Temporal Segmentation (IGTS) is a method for segmenting
multivariate time series based off reducing the entropy in each segment [1]_.

The amount of entropy lost by the segmentations made is called the Information
Gain (IG). The aim is to find the segmentations that have the maximum information
gain for any number of segmentations.

References
----------
.. [1] Sadri, Amin, Yongli Ren, and Flora D. Salim.
    "Information gain-based metric for recognizing transitions in human activities.",
    Pervasive and Mobile Computing, 38, 92-109, (2017).
    https://www.sciencedirect.com/science/article/abs/pii/S1574119217300081
"""

from sktime.detection.igts import InformationGainSegmentation

__all__ = ["InformationGainSegmentation"]
__author__ = ["lmmentel"]
