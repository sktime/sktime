#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

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
    labels : str {"indicator", "score"}
        Annotation labels

    Returns
    -------
    label : str
        Checked annotation label.
    """
    valid_labels = ["indicator", "score"]
    if labels not in valid_labels:
        raise ValueError(f"`labels` must be in: {valid_labels}, but found: {labels}.")
