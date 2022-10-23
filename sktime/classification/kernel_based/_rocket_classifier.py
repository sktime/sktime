# -*- coding: utf-8 -*-
"""RandOm Convolutional KErnel Transform (Rocket).

Pipeline classifier using the ROCKET transformer and RidgeClassifierCV estimator.
"""

__author__ = ["MatthewMiddlehurst", "victordremov", "fkiraly"]
__all__ = ["RocketClassifier"]

from deprecated.sphinx import deprecated

from sktime.classification.convolution_based import RocketClassifier as new_rc


# TODO: remove message in v0.15.0 and change base class
@deprecated(
    version="0.13.4",
    reason="RocketClassifier has moved to the classification.convolution_based package. This version will be removed in v0.15.0.",  # noqa: E501
    category=FutureWarning,
)
class RocketClassifier(new_rc):
    """Classifier wrapped for the Rocket transformer using RidgeClassifierCV.

    This classifier simply transforms the input data using the Rocket [1]_
    transformer and builds a RidgeClassifierCV estimator using the transformed data.

    Shorthand for the pipeline
    `rocket * StandardScaler(with_mean=False) * RidgeClassifierCV(alphas)`
    where `alphas = np.logspace(-3, 3, 10)`, and
    where `rocket` depends on params `rocket_transform`, `use_multivariate` as follows:

        | rocket_transform | `use_multivariate` | rocket (class)          |
        |------------------|--------------------|-------------------------|
        | "rocket"         | any                | Rocket                  |
        | "minirocket"     | "yes               | MiniRocketMultivariate  |
        | "minirocket"     | "no"               | MiniRocket              |
        | "multirocket"    | "yes"              | MultiRocketMultivariate |
        | "multirocket"    | "no"               | MultiRocket             |

    classes are sktime classes, other parameters are passed on to the rocket class.

    To build other classifiers with rocket transformers, use `make_pipeline` or the
    pipeline dunder `*`, and different transformers/classifiers in combination.

    Parameters
    ----------
    num_kernels : int, optional, default=10,000
        The number of kernels the for Rocket transform.
    rocket_transform : str, optional, default="rocket"
        The type of Rocket transformer to use.
        Valid inputs = ["rocket", "minirocket", "multirocket"]
    max_dilations_per_kernel : int, optional, default=32
        MiniRocket and MultiRocket only. The maximum number of dilations per kernel.
    n_features_per_kernel : int, optional, default=4
        MultiRocket only. The number of features per kernel.
    use_multivariate : str, ["auto", "yes", "no"], optional, default="auto"
        whether to use multivariate rocket transforms or univariate ones
        "auto" = multivariate iff data seen in fit is multivariate, otherwise univariate
        "yes" = always uses multivariate transformers, native multi/univariate
        "no" = always univariate transformers, multivariate by framework vectorization
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.
    estimator_ : ClassifierPipeline
        RocketClassifier as a ClassifierPipeline, fitted to data internally

    See Also
    --------
    Rocket

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/shapelet_based/ROCKETClassifier.java>`_.

    References
    ----------
    .. [1] Dempster, Angus, François Petitjean, and Geoffrey I. Webb. "Rocket:
       exceptionally fast and accurate time series classification using random
       convolutional kernels." Data Mining and Knowledge Discovery 34.5 (2020)

    Examples
    --------
    >>> from sktime.classification.kernel_based import RocketClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = RocketClassifier(num_kernels=500)
    >>> clf.fit(X_train, y_train)
    RocketClassifier(...)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        num_kernels=10000,
        rocket_transform="rocket",
        max_dilations_per_kernel=32,
        n_features_per_kernel=4,
        use_multivariate="auto",
        n_jobs=1,
        random_state=None,
    ):
        super(RocketClassifier, self).__init__(
            num_kernels=num_kernels,
            rocket_transform=rocket_transform,
            max_dilations_per_kernel=max_dilations_per_kernel,
            n_features_per_kernel=n_features_per_kernel,
            use_multivariate=use_multivariate,
            n_jobs=n_jobs,
            random_state=random_state,
        )
