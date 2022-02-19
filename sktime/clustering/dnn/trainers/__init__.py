# -*- coding: utf-8 -*-
__all__ = ["Trainer", "IDEC", "_TRAINERS"]
from sktime.clustering.dnn.trainers._base import Trainer
from sktime.clustering.dnn.trainers._IDEC import IDEC

_TRAINERS = {"IDEC": IDEC}
