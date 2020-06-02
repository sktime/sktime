import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def get_augmentation_pipeline(aug_list):
    """Buids an sklearn pipeline of augmentations from a list of strings.

    Parameters
    ----------
    aug_list: list of strings
              A list of strings that determine the augmentations to apply, and in which order to apply them (the first
              string will be applied first). Possible augmentation strings are ['leadlag', 'ir', 'addtime', 'cumsum',
              'basepoint']

    Returns
    -------
    sklearn.Pipeline
        The transforms, in order, as an sklearn pipeline.

    Examples
    --------
    >>> get_augmentation_pipeline(['leadlag', 'ir', 'addtime'])
    Pipeline([
        ('leadlag', LeadLag()),
        ('ir', InvisibilityReset()),
        ('addtime', AddTime())
    ])
    """
    # Dictionary of augmentations
    AUGMENTATIONS = {
        'leadlag': LeadLag(),
        'ir': InvisibilityReset(),
        'addtime': AddTime(),
        'cumsum': CumulativeSum(),
        'basepoint': Basepoint()
    }

    pipeline = Pipeline([
        (tfm_str, AUGMENTATIONS[tfm_str]) for tfm_str in aug_list
    ])

    return pipeline


class AddTime(BaseEstimator, TransformerMixin):
    """Add time component to each path.

    For a path of shape [N, L, C] this adds a time channel of length L running in [0, 1]. The output shape is thus
    [N, L, C + 1] with time being the first channel.
    """
    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        # Batch and length dim
        B, L = data.shape[0], data.shape[1]

        # Time scaled to 0, 1
        time_scaled = np.linspace(0, 1, L).repeat(B, 1).view(B, L, 1)

        return np.concatenate((time_scaled, data), 2)


class InvisibilityReset(TransformerMixin):
    """Adds an 'invisibility-reset' dimension to the path. This adds sensitivity to translation.

    Introduced by Yang et al.: https://arxiv.org/pdf/1707.03993.pdf
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Batch, length, channels
        B, L, C = X.shape[0], X.shape[1], X.shape[2]

        # Add in a dimension of ones
        X_pendim = np.concatenate((np.ones(B, L, 1), X), 2)

        # Add pen down to 0
        pen_down = X_pendim[:, [-1], :]
        pen_down[:, :, 0] = 0
        X_pendown = np.concatenate((X_pendim, pen_down), 1)

        # Add home
        home = np.concatenate(B, 1, C + 1)
        X_penoff = np.concatenate((X_pendown, home), 1)

        return X_penoff


class LeadLag(TransformerMixin):
    """Applies the lead-lag transformation to each path.

    We take the lead of an input stream, and augment it with the lag of the input stream. This enables us to capture the
    quadratic variation of the stream and is particularly useful for applications in finance.

    Used widely in signature literature, see for example:
        - https://arxiv.org/pdf/1603.03788.pdf
        - https://arxiv.org/pdf/1310.4054.pdf
        - https://arxiv.org/pdf/1307.7244.pdf
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Interleave
        # Change to X.repeat(2, dim=1)
        X_repeat = X.repeat_interleave(2, dim=1)

        # Split out lead and lag
        lead = X_repeat[:, 1:, :]
        lag = X_repeat[:, :-1, :]

        # Combine
        X_leadlag = np.concatenate((lead, lag), 2)

        return X_leadlag


class CumulativeSum(TransformerMixin):
    """Cumulatively sums the values in the stream.

    Introduced in: https://arxiv.org/pdf/1603.03788.pdf
    """
    def __init__(self, append_zero=False):
        self.append_zero = append_zero

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.append_zero:
            X = Basepoint().transform(X)
        return np.cumsum(X, 1)


class Basepoint(TransformerMixin):
    """Appends a zero starting vector to every path.

    Introduced in: https://arxiv.org/pdf/2001.00706.pdf
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        zero_vec = np.zeros(size=(X.size(0), 1, X.size(2)))
        return np.concatenate((zero_vec, X), dim=1)



if __name__ == '__main__':
    a = np.random.randn(10, 5, 7)
    a = LeadLag()