"""Temporary fork of sklearn metrics utilities to ensure compatibility with sklearn.

sklearn 1.7 changes metrics signatures, which breaks compatibility
of sktime across sklearn versions. This fork ensures upwards compatibility.
"""
# copyright/attribution scikit-learn developers
# _check_reg_targets_post16 is copy of the function of the same name from sklearn 1.6.X
# _check_reg_targets_pre16 is copy of the function of the same name from sklearn 1.5.X

__all__ = ["_check_reg_targets"]


from sktime.utils.dependencies import _check_soft_dependencies


def _check_reg_targets(y_true, y_pred, multioutput, dtype="numeric", xp=None):
    """Check that y_true and y_pred belong to the same regression task.

    To reduce redundancy when calling `_find_matching_floating_dtype`,
    please use `_check_reg_targets_with_floating_dtype` instead.

    Extracted from v1.6.X sklearn/metrics/_regression.py#L61
    Parameter `sample_weight` introduced in 1.7 removed
    Original Authors: The scikit-learn developers

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.

    xp : module, default=None
        Precomputed array namespace module. When passed, typically from a caller
        that has already performed inspection of its own inputs, skips array
        namespace inspection.

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.

    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    """
    kwargs = {
        "y_true": y_true,
        "y_pred": y_pred,
        "multioutput": multioutput,
        "dtype": dtype,
    }
    if _check_soft_dependencies("scikit-learn>=1.6", severity="none"):
        return _check_reg_targets_post_16(xp=xp, **kwargs)
    else:
        return _check_reg_targets_pre16(**kwargs)


def _check_reg_targets_post_16(y_true, y_pred, multioutput, dtype="numeric", xp=None):
    """Check that y_true and y_pred belong to the same regression task.

    To reduce redundancy when calling `_find_matching_floating_dtype`,
    please use `_check_reg_targets_with_floating_dtype` instead.

    Extracted from v1.6.X sklearn/metrics/_regression.py#L61
    Parameter `sample_weight` introduced in 1.7 removed
    Original Authors: The scikit-learn developers

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.

    xp : module, default=None
        Precomputed array namespace module. When passed, typically from a caller
        that has already performed inspection of its own inputs, skips array
        namespace inspection.

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.

    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    """
    from sklearn.utils._array_api import get_namespace
    from sklearn.utils.validation import check_array, check_consistent_length

    xp, _ = get_namespace(y_true, y_pred, multioutput, xp=xp)

    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = xp.reshape(y_true, (-1, 1))

    if y_pred.ndim == 1:
        y_pred = xp.reshape(y_pred, (-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            f"y_true and y_pred have different number of output"
            f" ({y_true.shape[1]}!={y_pred.shape[1]})"
        )

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")
    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                f"Allowed 'multioutput' string values are {allowed_multioutput_str}. "
                f"You provided multioutput={multioutput!r}"
            )
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        elif n_outputs != multioutput.shape[0]:
            raise ValueError(
                "There must be equally many custom weights "
                f"({multioutput.shape[0]}) as outputs ({n_outputs})."
            )
    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    return y_type, y_true, y_pred, multioutput


def _check_reg_targets_pre16(y_true, y_pred, multioutput, dtype="numeric"):
    """Check that y_true and y_pred belong to the same regression task.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    multioutput : array-like or string in ['raw_values', uniform_average',
        'variance_weighted'] or None
        None is accepted due to backward compatibility of r2_score().

    dtype : str or list, default="numeric"
        the dtype argument passed to check_array.

    Returns
    -------
    type_true : one of {'continuous', continuous-multioutput'}
        The type of the true target data, as output by
        'utils.multiclass.type_of_target'.

    y_true : array-like of shape (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples, n_outputs)
        Estimated target values.

    multioutput : array-like of shape (n_outputs) or string in ['raw_values',
        uniform_average', 'variance_weighted'] or None
        Custom output weights if ``multioutput`` is array-like or
        just the corresponding argument if ``multioutput`` is a
        correct keyword.
    """
    from sklearn.utils.validation import check_array, check_consistent_length

    check_consistent_length(y_true, y_pred)
    y_true = check_array(y_true, ensure_2d=False, dtype=dtype)
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            f"y_true and y_pred have different number of output "
            f"({y_true.shape[1]}!={y_pred.shape[1]})"
        )

    n_outputs = y_true.shape[1]
    allowed_multioutput_str = ("raw_values", "uniform_average", "variance_weighted")

    if isinstance(multioutput, str):
        if multioutput not in allowed_multioutput_str:
            raise ValueError(
                f"Allowed 'multioutput' string values are {allowed_multioutput_str}. "
                f"You provided multioutput={multioutput!r}"
            )
    elif multioutput is not None:
        multioutput = check_array(multioutput, ensure_2d=False)
        if n_outputs == 1:
            raise ValueError("Custom weights are useful only in multi-output cases.")
        elif n_outputs != len(multioutput):
            raise ValueError(
                "There must be equally many custom weights (%d) as outputs (%d)."
                % (len(multioutput), n_outputs)
            )

    y_type = "continuous" if n_outputs == 1 else "continuous-multioutput"

    return y_type, y_true, y_pred, multioutput
