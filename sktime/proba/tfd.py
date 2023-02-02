# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base classes for probability distribution objects."""

__author__ = ["fkiraly"]

import pandas as pd

from sktime.proba.base import _BaseTFProba


class Normal(_BaseTFProba):

    def __init__(self, mean, sd, index=None, columns=None):

        self.mean = mean
        self.sd = sd

        import tensorflow_probability as tfp
        tfd = tfp.distributions

        distr = tfd.Normal(loc=mean, scale=sd)

        if index is None:
            index = pd.RangeIndex(distr.batch_shape[0])

        if columns is None:
            columns = pd.RangeIndex(distr.batch_shape[1])

        super(Normal, self).__init__(index=index, columns=columns, distr=distr)
