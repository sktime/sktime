#import numpy as np
#
#from sktime.libs.fracdiff import FracdiffStat
#
#
#def test_32():
#    np.random.seed(42)
#    X = np.random.randn(100, 20).cumsum(0)
#    f = FracdiffStat(mode="valid")
#    f = f.fit(X)
#    X = f.transform(X)
#    X = f.fit_transform(X)
