"""Greedy Gaussian Segmentation (GGS).

The method approximates solutions for the problem of breaking a
multivariate time series into segments, where the data in each segment
could be modeled as independent samples from a multivariate Gaussian
distribution. It uses a dynamic programming search algorithm with
a heuristic that allows finding approximate solution in linear time with
respect to the data length and always yields locally optimal choice.

This module is structured with the ``GGS`` that implements the actual
segmentation algorithm and a ``GreedyGaussianSegmentation`` that
interfaces the algorithm with the sklearn/sktime api. The benefit
behind that design is looser coupling between the logic and the
interface introduced to allow for easier changes of either part
since segmentation still has an experimental nature. When making
algorithm changes you probably want to look into ``GGS`` when
evolving the sktime/sklearn interface look into ``GreedyGaussianSegmentation``.
This design also allows adapting ``GGS`` to other interfaces.

Notes
-----
Based on the work from [1]_.

- source code adapted based on: https://github.com/cvxgrp/GGS
- paper available at: https://stanford.edu/~boyd/papers/pdf/ggs.pdf

References
----------
.. [1] Hallac, D., Nystrup, P. & Boyd, S.
   "Greedy Gaussian segmentation of multivariate time series.",
    Adv Data Anal Classif 13, 727-751 (2019).
    https://doi.org/10.1007/s11634-018-0335-0
"""

from sktime.detection.ggs import GreedyGaussianSegmentation

__all__ = ["GreedyGaussianSegmentation"]
