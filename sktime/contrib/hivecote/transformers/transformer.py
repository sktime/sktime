#Base class transformer, should be abstract

class Transformer:

    def __init__(self, maxLag=100):
        self._maxLag=maxLag

    def transform(self, X):
        transformedX = np.copy(X)
        return transformedX
