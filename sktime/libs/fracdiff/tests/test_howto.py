class TestHowto:
    """
    Test if `How to use` in README runs without error

    fdiff
    >>> import numpy as np
    >>> from fracdiff import fdiff
    >>>
    >>> a = np.array([1, 2, 4, 7, 0])
    >>> fdiff(a, n=0.5)
    array([ 1.       ,  1.5      ,  2.875    ,  4.6875   , -4.1640625])
    >>> np.array_equal(fdiff(a, n=1), np.diff(a, n=1))
    True

    >>> a = np.array([[1, 3, 6, 10], [0, 5, 6, 8]])
    >>> fdiff(a, n=0.5, axis=0)
    array([[ 1. ,  3. ,  6. , 10. ],
           [-0.5,  3.5,  3. ,  3. ]])
    >>> fdiff(a, n=0.5, axis=-1)
    array([[1.    , 2.5   , 4.375 , 6.5625],
           [0.    , 5.    , 3.5   , 4.375 ]])

    Fracdiff
    >>> from fracdiff.sklearn import Fracdiff
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 1)
    >>> diff = Fracdiff(0.5).fit_transform(X)

    Pipeline
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import Pipeline
    >>>
    >>> X, y = np.random.randn(100, 2), np.random.randn(100)
    >>> pipeline = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('fracdiff', Fracdiff(0.5)),
    ...     ('regressor', LinearRegression()),
    ... ])
    >>> pipeline = pipeline.fit(X, y)

    FracdiffStat
    >>> from fracdiff.sklearn import FracdiffStat
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.randn(100, 3).cumsum(0)
    >>> f = FracdiffStat().fit(X)
    >>> f.d_
    array([0.71875 , 0.609375, 0.515625])
    """
