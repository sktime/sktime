import torch
from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.base import \
    BaseSeriesAsFeaturesTransformer


def make_augmentation_pipeline(aug_list):
    """Buids an sklearn pipeline of augmentations from a list of strings.

    Parameters
    ----------
    aug_list: list of strings, A list of strings that determine the
        augmentations to apply, and in which order to apply them (the first
        string will be applied first). Possible augmentation strings are
        ['leadlag', 'ir', 'addtime', 'cumsum', 'basepoint']

    Returns
    -------
    sklearn.Pipeline
        The transforms, in order, as an sklearn pipeline.

    Examples
    --------
    >>> make_augmentation_pipeline(['leadlag', 'ir', 'addtime'])
    Pipeline([
        ('leadlag', LeadLag()),
        ('ir', InvisibilityReset()),
        ('addtime', AddTime())
    ])
    """
    # Assertions
    types = [tuple, list, None, str]
    assert any([type(aug_list) == t for t in types]), (
        "`aug_list` must be one of {}. Got {}.".format(types, type(aug_list))
    )
    aug_list = [aug_list] if isinstance(aug_list, str) else aug_list

    # Dictionary of augmentations
    AUGMENTATIONS = {
        'leadlag': _LeadLag(),
        'ir': _InvisibilityReset(),
        'addtime': _AddTime(),
        'cumsum': _CumulativeSum(),
        'basepoint': _BasePoint()
    }

    pipeline = Pipeline([
        (tfm_str, AUGMENTATIONS[tfm_str]) for tfm_str in aug_list
    ])

    return pipeline


class _AddTime(BaseSeriesAsFeaturesTransformer):
    """Add time component to each path.

    For a path of shape [B, L, C] this adds a time channel to be placed at the
    first index. The time channel will be of length L and scaled to exist in
    [0, 1].
    """
    def transform(self, data):
        # Batch and length dim
        B, L = data.shape[0], data.shape[1]

        # Time scaled to 0, 1
        time_scaled = torch.linspace(0, 1, L).repeat(B, 1).view(B, L, 1)

        return torch.cat((time_scaled, data), 2)


class _InvisibilityReset(BaseSeriesAsFeaturesTransformer):
    """Adds an 'invisibility-reset' dimension to the path. This adds
    sensitivity to translation.

    Introduced by Yang et al.: https://arxiv.org/pdf/1707.03993.pdf
    """
    def transform(self, X):
        # Batch, length, channels
        B, L, C = X.shape[0], X.shape[1], X.shape[2]

        # Add in a dimension of ones
        X_pendim = torch.cat((torch.ones(B, L, 1), X), 2)

        # Add pen down to 0
        pen_down = X_pendim[:, [-1], :]
        pen_down[:, :, 0] = 0
        X_pendown = torch.cat((X_pendim, pen_down), 1)

        # Add home
        home = torch.zeros(B, 1, C + 1)
        X_penoff = torch.cat((X_pendown, home), 1)

        return X_penoff


class _LeadLag(BaseSeriesAsFeaturesTransformer):
    """Applies the lead-lag transformation to each path.

    We take the lead of an input stream, and augment it with the lag of the
    input stream. This enables us to capture the quadratic variation of the
    stream and is particularly useful for applications in finance.

    Used widely in signature literature, see for example:
        - https://arxiv.org/pdf/1603.03788.pdf
        - https://arxiv.org/pdf/1310.4054.pdf
        - https://arxiv.org/pdf/1307.7244.pdf
    """
    def transform(self, X):
        # Interleave
        X_repeat = X.repeat_interleave(2, dim=1)

        # Split out lead and lag
        lead = X_repeat[:, 1:, :]
        lag = X_repeat[:, :-1, :]

        # Combine
        X_leadlag = torch.cat((lead, lag), 2)

        return X_leadlag


class _CumulativeSum(BaseSeriesAsFeaturesTransformer):
    """Cumulatively sums the values in the stream.

    Introduced in: https://arxiv.org/pdf/1603.03788.pdf

    Parameters
    ----------
    append_zero: bool
        Set True to append zero to the path before taking the cumulative sum.
    """
    def __init__(self, append_zero=False):
        self.append_zero = append_zero

    def transform(self, X):
        if self.append_zero:
            X = _BasePoint().fit_transform(X)
        return torch.cumsum(X, 1)


class _BasePoint(BaseSeriesAsFeaturesTransformer):
    """Appends a zero starting vector to every path.

    Introduced in: https://arxiv.org/pdf/2001.00706.pdf
    """
    def transform(self, X):
        zero_vec = torch.zeros(size=(X.size(0), 1, X.size(2)))
        return torch.cat((zero_vec, X), dim=1)
