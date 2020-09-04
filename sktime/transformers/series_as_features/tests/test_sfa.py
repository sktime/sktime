import numpy as np
from sktime.transformers.series_as_features.dictionary_based._sfa import SFA
from sktime.datasets import load_gunpoint
from sktime.utils.data_container import tabularize


# Check the transformer has changed the data correctly.
def test_transformer():
    # load training data
    X, Y = load_gunpoint(split="train", return_X_y=True)

    word_length = 6
    alphabet_size = 4

    p = SFA(word_length=word_length, alphabet_size=alphabet_size,
            binning_method="equi-depth").fit(X, Y)
    print("Equi Depth")
    print(p.breakpoints)
    assert p.breakpoints.shape == (word_length, alphabet_size)
    assert np.equal(0, p.breakpoints[1, :-1]).all()  # imag component is 0

    p = SFA(word_length=word_length, alphabet_size=alphabet_size,
            binning_method="equi-width").fit(X, Y)
    print("Equi Width")
    print(p.breakpoints)
    assert p.breakpoints.shape == (word_length, alphabet_size)
    assert np.equal(0, p.breakpoints[1, :-1]).all()  # imag component is 0

    p = SFA(word_length=word_length, alphabet_size=alphabet_size,
            binning_method="information-gain").fit(X, Y)
    print("Information Gain")
    print(p.breakpoints)
    assert p.breakpoints.shape == (word_length, alphabet_size)

    print(p.breakpoints[1, :-1])
    assert np.equal(0, p.breakpoints[1, :-1]).all()  # imaginary component is 0


def test_dft_mft():
    # load training data
    X, Y = load_gunpoint(split="train", return_X_y=True)
    X_tab = tabularize(X, return_array=True)

    word_length = 6
    alphabet_size = 4

    print("Single DFT transformation")
    window_size = np.shape(X_tab)[1]
    p = SFA(word_length=word_length, alphabet_size=alphabet_size,
            window_size=window_size, binning_method="equi-depth").fit(X, Y)
    dft = p._discrete_fourier_transform(X_tab[0])
    mft = p._mft(X_tab[0])

    assert ((mft-dft < 0.0001).all())

    print("Windowed DFT transformation")

    for norm in [True, False]:
        for window_size in [140]:
            p = SFA(word_length=word_length, norm=norm,
                    alphabet_size=alphabet_size, window_size=window_size,
                    binning_method="equi-depth").fit(X, Y)
            mft = p._mft(X_tab[0])
            for i in range(len(X_tab[0]) - window_size + 1):
                dft_transformed = p._discrete_fourier_transform(
                                        X_tab[0, i:window_size+i])
                assert(mft[i] - dft_transformed < 0.001).all()

            assert(len(mft) == len(X_tab[0]) - window_size + 1)
            assert(len(mft[0]) == word_length)


def test_sfa_anova():
    # load training data
    X, Y = load_gunpoint(split="train", return_X_y=True)

    word_length = 6
    alphabet_size = 4

    for binning in ["information-gain", "equi-depth"]:
        print("SFA with ANOVA one-sided test")
        window_size = 32
        p = SFA(word_length=word_length, anova=True,
                alphabet_size=alphabet_size, window_size=window_size,
                binning_method=binning).fit(X, Y)

        print(p.breakpoints)
        print(p.support)
        print(p.dft_length)
        assert p.breakpoints.shape == (word_length, alphabet_size)

        print("SFA with first feq coefficients")
        p2 = SFA(word_length=word_length, anova=False,
                 alphabet_size=alphabet_size, window_size=window_size,
                 binning_method=binning).fit(X, Y)

        print(p2.breakpoints)
        print(p2.support)
        print(p2.dft_length)

        assert(p.dft_length != p2.dft_length)
        assert(p.breakpoints != p2.breakpoints).any()

    # p.transform(X)


# test word lengths larger than the window-length
def test_word_length():
    # load training data
    X, Y = load_gunpoint(split="train", return_X_y=True)

    word_lengths = [6, 7, 8, 11]
    alphabet_size = 4
    window_sizes = [4, 5, 6, 16]

    for binning in ["equi-depth", "information-gain"]:
        for word_length in word_lengths:
            for norm in [True, False]:
                for anova in [True, False]:
                    for window_size in window_sizes:
                        p = SFA(word_length=word_length, anova=anova,
                                alphabet_size=alphabet_size,
                                window_size=window_size, norm=norm,
                                binning_method=binning).fit(X, Y)

                        print("Norm", norm, "Anova", anova)
                        print(np.shape(p.breakpoints), word_length,
                              window_size)
                        print("dft_length", p.dft_length, "word_length",
                              p.word_length)
                        # assert(np.shape(p.breakpoints)==
                        #     min(word_lengths, window_sizes))
