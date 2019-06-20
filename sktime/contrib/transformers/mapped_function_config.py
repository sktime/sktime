import numpy as np

from scipy.signal import periodogram
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from enum import Enum

""" Config class for the mapped function
(comment to be added)

-- store the lambdas and some other infos in a static member dictionary
-- should work as a 'singleton' (unless I am missing something with Python - to be confirmed)
"""


class FunctionConfigs:

    class FuncType(Enum):
        discreteFT = 1
        discreteRealFT = 2,
        discreteHermiteFT = 3,
        powerSpectrum = 4,
        stdACF = 5,
        pACF = 6

    lambdaContainer = {
        FuncType.discreteFT: lambda p: np.fft.fftn(**p),
        FuncType.discreteRealFT: lambda p: np.fft.rfftn(**p),
        FuncType.discreteHermiteFT: lambda p: np.fft.hfft(**p),
        FuncType.powerSpectrum: lambda p: periodogram(**p),
        FuncType.stdACF: lambda p: acf(**p),
        FuncType.pACF: lambda p: pacf(**p)
    }

    constraintTo1D = {
        FuncType.powerSpectrum: True,
        FuncType.stdACF: True,
        FuncType.pACF: True
    }

    indexReturn = {
        FuncType.powerSpectrum: 0
    }

    def __init__(self):
        pass

