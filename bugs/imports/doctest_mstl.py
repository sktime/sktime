from sktime.datasets import load_airline
from sktime.transformations.mstl import MSTL

# bugs/imports/doctest_mstl.py
X = load_airline()
X.index = X.index.to_timestamp()
mstl = MSTL(return_components=True)
