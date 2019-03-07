'''
validation utilities for sktime
'''
# build on top of sklearn

def check_ts_X_y(X, y):
    '''
    use preexisting ones with bypass (temporarily)
    '''
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_X_y(X, y, dtype=None, ensure_2d=False)
    return X, y

def check_ts_array(X):
    '''
    use preexisting ones with bypass (temporarily)
    '''
    # TODO: add proper checks (e.g. check if input stuff is pandas full of objects)
    # currently it checks neither the data nor the datatype
    # return check_array(X, dtype=None, ensure_2d=False)
    return X
