# -*- coding: utf-8 -*-
"""ShapeletTransform test code."""
import numpy as np
from numpy import testing

from sktime.datasets import load_basic_motions, load_unit_test
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform


def test_st_on_unit_test():
    """Test of ShapeletTransform on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train")
    indices = np.random.RandomState(0).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    st = RandomShapeletTransform(
        max_shapelets=10, n_shapelet_samples=500, random_state=0
    )
    st.fit(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train.iloc[indices])
    testing.assert_array_almost_equal(data, shapelet_transform_unit_test_data)


def test_st_on_basic_motions():
    """Test of ShapeletTransform on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 5, replace=False)

    # fit the shapelet transform
    st = RandomShapeletTransform(
        max_shapelets=10, n_shapelet_samples=500, random_state=0
    )
    st.fit(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train.iloc[indices])
    testing.assert_array_almost_equal(data, shapelet_transform_basic_motions_data)


shapelet_transform_unit_test_data = np.array(
    [
        [
            0.0844948936238554,
            0.15355295821521212,
            0.181181653406588,
        ],
        [
            0.1005739706371287,
            0.13728264572793214,
            0.14232864157873387,
        ],
        [
            0.15574477738923456,
            0.24572051577166446,
            0.28114769465706674,
        ],
        [
            0.14368755013218146,
            0.11173267717634065,
            0.0832337710268795,
        ],
        [
            0.05901578181579171,
            0.11702300016584308,
            0.14612594209368096,
        ],
    ]
)
shapelet_transform_basic_motions_data = np.array(
    [
        [
            1.0891026731161,
            0.9869567751155376,
            1.500478686384502,
            1.9604066805556999,
            1.9300459325831565,
            1.6290470525017764,
            1.2492444105003735,
            1.0060446077996184,
        ],
        [
            1.1700758360488173,
            1.0555356143008514,
            0.6147335984845409,
            0.9762741759423724,
            0.5589265732729417,
            1.032742062232156,
            1.083409283764176,
            1.111697276658204,
        ],
        [
            1.6798705292742746,
            1.9684063044201972,
            2.453685502926318,
            1.9677105642494732,
            2.029428399113479,
            1.3483536658952058,
            1.9877554408696116,
            0.5488432707540976,
        ],
        [
            1.1079276425471314,
            1.0065997349055864,
            1.0618258792202282,
            0.32297427738972406,
            1.1450380706584913,
            1.0387357068111138,
            0.676941814847556,
            1.0156811721014811,
        ],
        [
            0.33414369747055067,
            0.2870956468054047,
            1.734401894586686,
            1.9064659364611127,
            1.7299782521480092,
            1.6297951854173116,
            1.5659881867675363,
            1.1203189560668823,
        ],
    ]
)
