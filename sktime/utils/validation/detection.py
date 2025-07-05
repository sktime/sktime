#!/usr/bin/env python3 -u
"""Series of checks for annotation classes."""

__author__ = ["mloning"]
__all__ = ["check_fmt", "check_labels"]


def check_fmt(fmt):
    """Check annotation format.

    Parameters
    ----------
    fmt : str {"sparse", "dense"}
        Annotation format

    Returns
    -------
    fmt : str
        Checked annotation format.
    """
    valid_fmts = ["sparse", "dense"]
    if fmt not in valid_fmts:
        raise ValueError(f"`fmt` must be in: {valid_fmts}, but found: {fmt}.")


def check_labels(labels):
    """Check annotation label.

    Parameters
    ----------
    labels : str {"indicator", "score", "int_label"}
        Annotation labels

    Returns
    -------
    label : str
        Checked annotation label.
    """
    valid_labels = ["indicator", "score", "int_label"]
    if labels not in valid_labels:
        raise ValueError(f"`labels` must be in: {valid_labels}, but found: {labels}.")


def check_task(task):
    """Check annotation task.

    Parameters
    ----------
    task : str {"segmentation", "change_point_detection", "anomaly_detection"}
        Annotation task.
    """
    valid_tasks = ["segmentation", "change_point_detection", "anomaly_detection"]
    if task != "None" and task not in valid_tasks:
        raise ValueError(f"`task` must be in: {valid_tasks}, but found: {task}.")


def check_learning_type(learning_type):
    """Check learning type.

    Parameters
    ----------
    learning_type : str {"supervised", "unsupervised"}
        Annotation Learning type.
    """
    valid_learning_types = ["supervised", "unsupervised"]
    if learning_type not in valid_learning_types:
        raise ValueError(
            f"`learning_type` must be in: {valid_learning_types}, "
            f"but found: {learning_type}."
        )
