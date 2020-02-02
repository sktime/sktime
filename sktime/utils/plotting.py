"""
Plotting utilities.
"""


__author__ = ["big-o@github"]
__all__ = ["composite_alpha"]


def composite_alpha(underlay_alpha, overlay_alpha):
    """
    Get the alpha value of two overlaid transparencies.

    References
    ----------
    * https://en.wikipedia.org/wiki/Alpha_compositing
    """
    return overlay_alpha + underlay_alpha * (1 - overlay_alpha)
