"""ClaSP (Classification Score Profile) Segmentation.

Notes
-----
As described in
@inproceedings{clasp2021,
  title={ClaSP - Time Series Segmentation},
  author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
  booktitle={CIKM},
  year={2021}
}
"""

from sktime.detection.clasp import ClaSPSegmentation, find_dominant_window_sizes

__author__ = ["ermshaua", "patrickzib"]
__all__ = ["ClaSPSegmentation", "find_dominant_window_sizes"]
