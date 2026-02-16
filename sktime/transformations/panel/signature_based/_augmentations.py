import numpy as np
from sklearn.pipeline import Pipeline

from sktime.transformations.base import BaseTransformer


def _make_augmentation_pipeline(augmentation_list):
    """Build an sklearn pipeline of augmentations from a tuple of strings.

    Parameters
    ----------
    augmentation_list: list of strings, A list of strings that determine the
        augmentations to apply, and in which order to apply them (the first
        string will be applied first). Possible augmentation strings are
        ['leadlag', 'ir', 'addtime', 'cumsum', 'basepoint']

    Returns
    -------
    sklearn.Pipeline
        The transforms, in order, as an sklearn pipeline.

    Examples
    --------
        augementations = ('leadlag', 'ir', 'addtime')
        _make_augmentation_pipeline(augmentations)
        # Will return
        Pipeline([
            ('leadlag', LeadLag()),
            ('ir', InvisibilityReset()),
            ('addtime', AddTime())
        ])
    """
    # Dictionary of augmentations
    AUGMENTATIONS = {
        "leadlag": _LeadLag(),
        "ir": _InvisibilityReset(),
        "addtime": _AddTime(),
        "cumsum": _CumulativeSum(),
        "basepoint": _BasePoint(),
    }

    # Assertions, check we have a tuple/list
    if augmentation_list is not None:
        if isinstance(augmentation_list, str):
            augmentation_list = (augmentation_list,)
        if not [x in list(AUGMENTATIONS.keys()) for x in augmentation_list]:
            raise ValueError(
                "augmentation_list must only contain string elements from "
                f" {list(AUGMENTATIONS.keys())}. Found: {augmentation_list}"
            )

    # Setup pipeline
    if augmentation_list is not None:
        pipeline = Pipeline(
            [(tfm_str, AUGMENTATIONS[tfm_str]) for tfm_str in augmentation_list]
        )
    else:
        pipeline = None

    return pipeline


class _AddTime(BaseTransformer):
    """Add time component to each path.

    For a path of shape [B, L, C] this adds a time channel to be placed at the first
    index. The time channel will be of length L and scaled to exist in [0, 1].
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "jambo6",
        "maintainers": "jambo6",
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
    }

    def _transform(self, X, y=None):
        data = np.swapaxes(X, 1, 2)
        # Batch and length dim
        B, L = data.shape[0], data.shape[1]

        # Time scaled to 0, 1
        time_scaled = np.linspace(0, 1, L).reshape(1, L).repeat(B, 0).reshape(B, L, 1)

        Xt = np.concatenate((time_scaled, data), 2)
        return np.swapaxes(Xt, 1, 2)


class _InvisibilityReset(BaseTransformer):
    """Add 'invisibility-reset' dimension to the path.

    This adds sensitivity to translation.

    Introduced by Yang et al.:
    https://arxiv.org/pdf/1707.03993.pdf
    : https: //arxiv.org/pdf/1707.03993.pdf
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
    }

    def _transform(self, X, y=None):
        X = np.swapaxes(X, 1, 2)

        # Batch, length, channels
        B, L, C = X.shape[0], X.shape[1], X.shape[2]

        # Add in a dimension of ones
        X_pendim = np.concatenate((np.ones(shape=(B, L, 1)), X), 2)

        # Add pen down to 0
        pen_down = X_pendim[:, [-1], :]
        pen_down[:, :, 0] = 0
        X_pendown = np.concatenate((X_pendim, pen_down), 1)

        # Add home
        home = np.zeros(shape=(B, 1, C + 1))
        X_penoff = np.concatenate((X_pendown, home), 1)

        Xt = np.swapaxes(X_penoff, 1, 2)
        return Xt


class _LeadLag(BaseTransformer):
    """Applies the lead-lag transformation to each path.

    We take the lead of an input stream, and augment it with the lag of the
    input stream. This enables us to capture the quadratic variation of the
    stream and is particularly useful for applications in finance.

    Used widely in signature literature, see for example:
        - https://arxiv.org/pdf/1603.03788.pdf
        - https://arxiv.org/pdf/1310.4054.pdf
        - https://arxiv.org/pdf/1307.7244.pdf
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
    }

    def _transform(self, X, y=None):
        X = np.swapaxes(X, 1, 2)

        # Interleave
        X_repeat = X.repeat(2, axis=1)

        # Split out lead and lag
        lead = X_repeat[:, 1:, :]
        lag = X_repeat[:, :-1, :]

        # Combine
        X_leadlag = np.concatenate((lead, lag), 2)

        Xt = np.swapaxes(X_leadlag, 1, 2)
        return Xt


class _CumulativeSum(BaseTransformer):
    """Cumulatively sums the values in the stream.

    Introduced in: https://arxiv.org/pdf/1603.03788.pdf

    Parameters
    ----------
    append_zero: bool
        Set True to append zero to the path before taking the cumulative sum.
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
    }

    def __init__(self, append_zero=False):
        self.append_zero = append_zero
        super().__init__()

    def _transform(self, X, y=None):
        if self.append_zero:
            X = _BasePoint().fit_transform(X)
        Xt = np.cumsum(X, 2)
        return Xt


class _BasePoint(BaseTransformer):
    """Appends a zero starting vector to every path.

    Introduced in: https://arxiv.org/pdf/2001.00706.pdf
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
    }

    def _transform(self, X, y=None):
        X = np.swapaxes(X, 1, 2)
        zero_vec = np.zeros(shape=(X.shape[0], 1, X.shape[2]))
        Xt = np.concatenate((zero_vec, X), axis=1)
        return np.swapaxes(Xt, 1, 2)
