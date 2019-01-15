'''
validation utilities for sktime
'''
# build on top of sklearn
from sklearn.utils.validation import *

def check_ts_X_y(X, y):
    '''
    use preexisting ones with bypass (temporarily)
    '''
    # TODO: add proper checks (e.g. check if input stuff is xpandas)
    return check_X_y(X, y, dtype=None, ensure_2d=False)
