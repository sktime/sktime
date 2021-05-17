from sktime.clustering.base.base import AverageMixin


class BarycenterAveraging(AverageMixin):
    """
    Implementations based off:
    https://blog.acolyer.org/2016/05/13/
    dynamic-time-warping-averaging-of-time-series-allows-faster
    -and-more-accurate-classification/
    """

    @staticmethod
    def average(series, iterations=100):
        pass




"""
We try to synthesise a guess

We start with a guess. This can be a medoid or a random element

We then take this guess and run DTW to all the elements in the set

Each point will then be tied to at least one of these points 

For each point compute barycenter c of the set of points

"""