# -*- coding: utf-8 -*-
"""ShapeletTransform test code."""
import numpy as np
from numpy import testing

from sktime.datasets import load_basic_motions, load_unit_test
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform


def test_st_on_unit_test():
    """Test of ShapeletTransform on unit test data."""
    # load unit test data
    X_train, y_train = load_unit_test(split="train", return_X_y=True)
    indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)

    # fit the shapelet transform
    st = RandomShapeletTransform(
        max_shapelets=10, n_shapelet_samples=500, batch_size=100, random_state=0
    )
    st.fit(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train.iloc[indices])
    testing.assert_array_almost_equal(data, shapelet_transform_unit_test_data)


def test_st_on_basic_motions():
    """Test of ShapeletTransform on basic motions data."""
    # load basic motions data
    X_train, y_train = load_basic_motions(split="train", return_X_y=True)
    indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)

    # fit the shapelet transform
    st = RandomShapeletTransform(
        max_shapelets=10, n_shapelet_samples=500, batch_size=100, random_state=0
    )
    st.fit(X_train.iloc[indices], y_train[indices])

    # assert transformed data is the same
    data = st.transform(X_train.iloc[indices])
    testing.assert_array_almost_equal(data, shapelet_transform_basic_motions_data)


shapelet_transform_unit_test_data = np.array(
    [
        [
            0.2016835914684362,
            0.2832255263004523,
            0.6600061145300044,
            0.19651105974493702,
            0.08189106653880576,
            0.02312927236278174,
            0.9951100053746514,
            0.29624757127319756,
        ],
        [
            0.10727766079454963,
            0.1825919200757111,
            0.35804924658625903,
            0.12299909200727441,
            0.16222713514620674,
            0.04704951170694063,
            0.22375417590660943,
            0.19200264930549701,
        ],
        [
            0.27736714817510305,
            0.3790367337585163,
            0.6618200580865141,
            0.2770517785660788,
            0.04604914008560876,
            0.026772176113201634,
            0.6526699074372295,
            0.2822402838348733,
        ],
        [
            0.07926685557939991,
            0.13577243996174876,
            0.2988108151883935,
            0.13548096555892514,
            0.1309678838777755,
            0.03923772503690868,
            0.3353372067604588,
            0.149513800152528,
        ],
        [
            0.14993904932316407,
            0.21157295884219354,
            0.6317209423143071,
            0.1386514210448127,
            0.04550177709613305,
            0.023145760517636903,
            0.8810230656359461,
            0.27157781389204255,
        ],
        [
            0.2007777643266854,
            0.2879662865379504,
            0.6366319019882033,
            0.16860100053733224,
            0.11339490758683682,
            0.02919054693596177,
            0.9919492539012587,
            0.25322366219032405,
        ],
        [
            0.07789879263223631,
            0.13859466122680128,
            0.31325465710713146,
            0.1132383755884878,
            0.1393452750170486,
            0.039288972161079946,
            0.3075650854168276,
            0.18474786977969887,
        ],
        [
            0.29627238241229037,
            0.4022872464113452,
            0.6427505682492338,
            0.21930422221051868,
            0.029190546935961764,
            0.026795309581298315,
            0.9975708816151168,
            0.27925069717845535,
        ],
        [
            0.11856838520625607,
            0.18727218521003008,
            0.2783077311933276,
            0.09929286684171075,
            0.1816235076210207,
            0.05814089646523,
            0.1519352018890305,
            0.17206734438440757,
        ],
        [
            0.11945033003389503,
            0.19647528401651682,
            0.3302812204338322,
            0.12035302745105327,
            0.14355187796979302,
            0.04460922432927829,
            0.26384798060198206,
            0.19673482571934198,
        ],
    ]
)
shapelet_transform_basic_motions_data = np.array(
    [
        [
            1.2492444105003735,
            0.996733360858608,
            0.3124966240071118,
            0.24549475659014006,
            2.0040350133871616,
            0.8992485913138568,
            1.1729283579207437,
        ],
        [
            1.083409283764176,
            1.0265363137424077,
            1.1486653850703599,
            1.1080193754789156,
            1.1859720571515342,
            1.0897378736098289,
            1.026489359437069,
        ],
        [
            1.9877554408696116,
            1.7569428181466589,
            1.225605942220214,
            1.2306104201043717,
            1.7960519587847787,
            0.4149296106616864,
            0.8438485623029778,
        ],
        [
            0.676941814847556,
            0.8502774578656134,
            0.9817573460247753,
            0.9990562012080618,
            1.2159234473675102,
            1.0402246567741018,
            1.0069865489524839,
        ],
        [
            1.5659881867675363,
            1.129337217113165,
            0.9779777228031981,
            0.9429942571033179,
            1.9643574030940585,
            1.1476955825349098,
            1.2383719640463084,
        ],
        [
            1.2634586531894108,
            1.0301998861101407,
            0.3239878181194262,
            0.2611837994102352,
            1.7500769918980303,
            0.8870629611416393,
            1.123296642512825,
        ],
        [
            1.1526819072859862,
            1.0450765544820333,
            1.042969270519787,
            0.9962887485820753,
            0.41297349277643053,
            1.0750599091794886,
            1.0513092084526798,
        ],
        [
            2.055006627077328,
            1.8400295646412996,
            1.452977627160922,
            1.4393540329445718,
            2.5949136726298665,
            0.6800186218469984,
            0.004834922389716515,
        ],
        [
            0.830755528637595,
            0.905011775876799,
            0.9841811829253059,
            0.9810064099863535,
            1.3370612907695631,
            1.005227621107627,
            0.9971441195913887,
        ],
        [
            0.9617984853145142,
            0.9745666599684619,
            1.0438148624254169,
            1.0192054300201443,
            1.2479822584006093,
            1.0025191227808834,
            1.043729559186247,
        ],
    ]
)


# def print_array(array):
#     print("[")
#     for sub_array in array:
#         print("[")
#         for value in sub_array:
#             print(value.astype(str), end="")
#             print(", ")
#         print("],")
#     print("]")
#
#
# if __name__ == "__main__":
#     X_train, y_train = load_unit_test(split="train", return_X_y=True)
#     indices = np.random.RandomState(0).choice(len(y_train), 10, replace=False)
#
#     st_u = RandomShapeletTransform(
#         max_shapelets=10,
#         n_shapelet_samples=500,
#         batch_size=100,
#         random_state=0,
#     )
#     st_u.fit(X_train.iloc[indices], y_train[indices])
#
#     data = st_u.transform(X_train.iloc[indices])
#     print_array(data.to_numpy())
#
#     X_train, y_train = load_basic_motions(split="train", return_X_y=True)
#     indices = np.random.RandomState(4).choice(len(y_train), 10, replace=False)
#
#     st_m = RandomShapeletTransform(
#         max_shapelets=10,
#         n_shapelet_samples=500,
#         batch_size=100,
#         random_state=0,
#     )
#     st_m.fit(X_train.iloc[indices], y_train[indices])
#
#     data = st_m.transform(X_train.iloc[indices])
#     print_array(data.to_numpy())
