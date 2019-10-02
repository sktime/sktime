import numpy as np
from sktime.distances.stomp_ab import stomp_ab


def mpdist(ts1, ts2, m):
    """
        MPDist implementation.

        Parameters
        ----------
            ts1: numpy.array
                First time series.
            ts2: numpy.array
                Second time series.
            m: int
                Length of the subsequences.

        Output
        ------
            mpdist: float
                Distance between the two time series.
    """

    len1 = ts1.size
    len2 = ts2.size

    ts1 = ts1.flatten()
    ts2 = ts2.flatten()

    threshold = 0.05
    mp_ab, ip_ab = stomp_ab(ts1, ts2, m) # compute the AB matrix profile
    mp_ba, ip_ba = stomp_ab(ts2, ts1, m) # compute the BA matrix profile

    join_mp = np.concatenate([mp_ab, mp_ba])

    k = int(np.ceil(threshold * (len1 + len2)))

    sorted_mp = np.sort(join_mp) # sort the join matrix profile in ascending order

    if len(sorted_mp) > k:
        mpdist = sorted_mp[k]
    else:
        mpdist = sorted_mp[len(sorted_mp)-1]

    return mpdist
