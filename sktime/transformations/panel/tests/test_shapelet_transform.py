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
            0.1757719884657562,
            0.15992162325775366,
            0.2832255263004523,
            0.0710103864788105,
            0.02472501952125449,
            0.03880304537321462,
            0.023384970777461846,
        ],
        [
            0.1148001273948484,
            0.10898669626593543,
            0.1825919200757111,
            0.16953505643766167,
            0.06156909578959216,
            0.07844222785518115,
            0.04629631296857717,
        ],
        [
            0.25277167087402796,
            0.23352125976208876,
            0.3790367337585163,
            0.05990298575911813,
            0.028919805844854513,
            0.03093325167650216,
            0.011676844170868343,
        ],
        [
            0.0975857350906744,
            0.07789370649741871,
            0.13577243996174876,
            0.12486541778253603,
            0.07546156244868375,
            0.06061592700562912,
            0.04083725843750778,
        ],
        [
            0.1294396672527971,
            0.1282575955154975,
            0.21157295884219354,
            0.09620043749030398,
            0.02446743415925637,
            0.04479864382210876,
            0.027855044425825797,
        ],
        [
            0.1976485548434086,
            0.1722132884297256,
            0.2879662865379504,
            0.06565983381204364,
            0.028837217149966846,
            0.0460491400856088,
            0.024868126525037428,
        ],
        [
            0.08830679640583053,
            0.072072260760344,
            0.13859466122680128,
            0.12851278948997544,
            0.07200833538720175,
            0.06379541863917046,
            0.04721017418031554,
        ],
        [
            0.2634863310538662,
            0.24389448064315408,
            0.4022872464113452,
            0.05337978669509336,
            0.028770358738606245,
            0.03331524963729007,
            0.003721072293094656,
        ],
        [
            0.11555561268097786,
            0.123394136414884,
            0.18727218521003008,
            0.1673101472734542,
            0.10290729335731612,
            0.0758898777410288,
            0.04823249145627325,
        ],
        [
            0.1296621386612994,
            0.1264594087030189,
            0.19647528401651682,
            0.15959375829536182,
            0.07129480587615834,
            0.07352009422155487,
            0.040777413564837287,
        ],
    ]
)
shapelet_transform_basic_motions_data = np.array(
    [
        [
            1.345304008891698,
            0.5651218104290232,
            0.3102570336173056,
            1.92145983751738,
            1.4411168012703106,
            1.212176828971964,
            1.1922650509959434,
        ],
        [
            1.105519916764494,
            1.2152251845670572,
            1.1492289144740129,
            1.1837601838569116,
            1.0367038985817003,
            1.0614497285153601,
            0.9970284443290787,
        ],
        [
            1.5726145462951937,
            1.446908060211927,
            1.2137378608948617,
            1.7704417793189327,
            1.6504744825964877,
            1.0125655489981515,
            0.8805631229455629,
        ],
        [
            0.8266223592567571,
            1.0096919814562915,
            0.9812823283169091,
            1.2126939252712559,
            1.0479933531859196,
            1.0849385648906396,
            1.0231333175224768,
        ],
        [
            1.5017404728104367,
            0.906140048841181,
            0.9776170784480609,
            1.953546387790228,
            1.1240767574598622,
            1.768396416772073,
            1.2363486741216145,
        ],
        [
            1.2793718534982117,
            0.8154850464304445,
            0.3269953964084556,
            1.7394302557408154,
            1.1362320761286104,
            1.4113623431926274,
            1.1811571324766674,
        ],
        [
            1.2076728541350499,
            1.1646040489688703,
            1.0438913682060793,
            0.4175309917812291,
            0.5019214730610423,
            1.0384407692349018,
            1.0367507772184055,
        ],
        [
            1.703555710666114,
            1.820468147672736,
            1.4591416455209858,
            2.1626562222185974,
            1.7469053773609182,
            0.16954293345291546,
            0.006355687709297252,
        ],
        [
            0.6854050578831657,
            1.1608190267080896,
            0.9813124909480698,
            1.3408031804866116,
            1.0870576270525134,
            1.04492271592571,
            1.0106589477935877,
        ],
        [
            1.041065667342548,
            1.3269673407111702,
            1.0470706635636349,
            1.2415896261884927,
            1.0685842330229485,
            1.106250418676523,
            1.037273757209312,
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
