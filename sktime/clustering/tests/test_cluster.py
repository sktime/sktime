# -*- coding: utf-8 -*-
import pytest

from sktime.clustering._cluster import Cluster


def test_cluster():
    try:
        Cluster(metric="dtw")
    except Exception:
        pytest.fail("Failed to construct base Cluster with dtw")

    try:
        Cluster(metric="ddtw")
    except Exception:
        pytest.fail("Failed to construct base Cluster with ddtw")

    try:
        Cluster(metric="wdtw")
    except Exception:
        pytest.fail("Failed to construct base Cluster with wdtw")

    try:
        Cluster(metric="wddtw")
    except Exception:
        pytest.fail("Failed to construct base Cluster with wddtw")

    try:
        Cluster(metric="lcss")
    except Exception:
        pytest.fail("Failed to construct base Cluster with lcss")

    try:
        Cluster(metric="erp")
    except Exception:
        pytest.fail("Failed to construct base Cluster with erp")

    try:
        Cluster(metric="msm")
    except Exception:
        pytest.fail("Failed to construct base Cluster with msm")

    try:
        Cluster(metric="twe")
    except Exception:
        pytest.fail("Failed to construct base Cluster with twe")

    try:
        Cluster(metric="mpdist")
    except Exception:
        pytest.fail("Failed to construct base Cluster with mpdist")
