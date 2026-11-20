import pytest
import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies
from sktime.tests.test_switch import run_test_module_changed
from sktime.libs.tbats.tbats import HarmonicsChoosingStrategy, Context, Components, ModelParams

@pytest.mark.skipif(
    not _check_soft_dependencies("pmdarima", severity="none")
    or not run_test_module_changed("sktime.libs.tbats"),
    reason="Execute tests iff pmdarima is available and anything in the TBATS module has changed",
)
class TestTBATSHarmonicsChoosingStrategy(object):
    class ModelMock:
        def __init__(self, y, params, aic_score):
            self.params = params
            self.y = y
            self.aic = aic_score
            self.is_fitted = True

        def calculate_aic(self):
            return self.aic

    class CaseMock:
        def __init__(self, components, aic_score):
            self.components = components
            self.aic_score = aic_score

        def fit(self, y):
            params = ModelParams(self.components, alpha=0)
            return TestTBATSHarmonicsChoosingStrategy.ModelMock(y, params, self.aic_score)

    class ContextMock(Context):
        def __init__(self, aic_score_map):
            super().__init__(n_jobs=1)
            self.aic_score_map = aic_score_map

        def create_case(self, components):
            for (harmonics, aic_score) in self.aic_score_map:
                if np.array_equal(components.seasonal_harmonics, harmonics):
                    return TestTBATSHarmonicsChoosingStrategy.CaseMock(components, aic_score)
            raise Exception('Unknown score for harmonics ' + str(components.seasonal_harmonics))

    @pytest.mark.parametrize(
        "components, aic_score_map, expected_harmonics",
        [
            [  # no periods means no harmonics
                dict(),
                [
                    # strategy shouldn't even attempt to build a model
                ],
                [],
            ],
            [  # when score for 5 is better than for 6, strategy will be checking less complex models
                dict(seasonal_periods=[21]),
                [  # AIC score map for chosen harmonics
                    ([1], 10.),
                    ([6], 8.),
                    ([7], 9.),
                    ([5], 7.),
                    ([4], 6.),  # best model
                    ([3], 7.),
                ],
                [4],
            ],
            [  # when score for 7 is better than one for 6, strategy will be checking more complex models
                dict(seasonal_periods=[21]),
                [  # AIC score map for chosen harmonics
                    ([1], 20.),
                    ([5], 9.),
                    ([6], 8.),
                    ([7], 7.),
                    ([8], 6.),
                    ([9], 5.),  # best model
                    ([10], 6.),
                ],
                [9],
            ],
            [  # if initial model is the best one, it should be returned
                dict(seasonal_periods=[30]),
                [  # AIC score map for chosen harmonics
                    ([1], 1.),  # best model
                    ([5], 8.),
                    ([6], 6.),
                    ([7], 9.),
                ],
                [1],
            ],
            [  # a model with small amount of harmonics
                dict(seasonal_periods=[4]),
                [  # AIC score map for chosen harmonics
                    ([1], 1.),  # this is the only possible model
                ],
                [1],
            ],
            [  # two periods
                dict(seasonal_periods=[7, 365]),
                [  # AIC score map for chosen harmonics
                    ([1, 1], 20.),
                    ([2, 1], 18.),  # best model for 1st season
                    ([3, 1], 19.),
                    ([2, 5], 16.),
                    ([2, 6], 15.),
                    ([2, 7], 14.),
                    ([2, 8], 13.),  # best model
                    ([2, 9], 16.),
                ],
                [2, 8],
            ],
            [  # three periods
                dict(seasonal_periods=[7, 30, 365]),
                [  # AIC score map for chosen harmonics
                    ([1, 1, 1], 20.),
                    ([2, 1, 1], 18.),
                    ([3, 1, 1], 17.),  # best model for 1st season
                    ([4, 1, 1], 19.),
                    ([3, 5, 1], 16.),
                    ([3, 6, 1], 15.),
                    ([3, 7, 1], 14.),
                    ([3, 8, 1], 13.),  # best model for 2nd season
                    ([3, 9, 1], 16.),
                    ([3, 8, 7], 14.),
                    ([3, 8, 6], 12.),
                    ([3, 8, 5], 11.),
                    ([3, 8, 4], 10.),  # best model
                    ([3, 8, 3], 11.),
                ],
                [3, 8, 4],
            ],
        ]
    )
    def test_choose(self, components, aic_score_map, expected_harmonics):
        context = self.ContextMock(aic_score_map)
        strategy = HarmonicsChoosingStrategy(context, checking_range=1)
        harmonics = strategy.choose([1, 2, 3], Components(**components))
        assert np.array_equal(expected_harmonics, harmonics)

    @pytest.mark.parametrize(
        "seasonal_periods, expected_max_harmonics",
        [
            [  # no periods means no harmonics
                [], [],
            ],
            [  # period of length 2 is limited to 1
                [2], [1],
            ],
            [  # floor((4 - 1) / 2) = 1
                [4], [1],
            ],
            [
                [5], [2],
            ],
            [
                [9], [4],
            ],
            [  # floor((28 - 1) / 2) = 13
                [28], [13],
            ],
            [  # 2nd seasonal harmonic for 16 is equal to 1st seasonal harmonic for 8
                [8, 16], [3, 1]
            ],
            [  # 10th seasonal harmonic for 100 is equal to 1st seasonal harmonic for 10
                [10, 100], [4, 9]
            ],
            [  # 2nd seasonal harmonic for 100 is equal to 1st seasonal harmonic for 50
                # 5th seasonal harmonic for 50 is equal to 1st seasonal harmonic for 10
                [10, 50, 100], [4, 4, 1]
            ],
            [  # 5th seasonal harmonic for 50 is equal to 2nd seasonal harmonic for 20
                [20, 50], [9, 4]
            ],
            [  # This method does not work with floats, see _better implementation
                [25.5, 51], [12, 25]
            ],
        ]
    )
    def test_calculate_max(self, seasonal_periods, expected_max_harmonics):
        strategy = HarmonicsChoosingStrategy(Context(), checking_range=1)
        harmonics = strategy.calculate_max(np.array(seasonal_periods))
        assert np.array_equal(expected_max_harmonics, harmonics)

    @pytest.mark.parametrize(
        "seasonal_periods, expected_max_harmonics",
        [
            [  # no periods means no harmonics
                [], [],
            ],
            [  # period of length 2 is limited to 1
                [2], [1],
            ],
            [  # floor((4 - 1) / 2) = 1
                [4], [1],
            ],
            [
                [5], [2],
            ],
            [
                [9], [4],
            ],
            [  # floor((28 - 1) / 2) = 13
                [28], [13],
            ],
            [  # 2nd seasonal harmonic for 16 is equal to 1st seasonal harmonic for 8
                [8, 16], [3, 1]
            ],
            [  # 10th seasonal harmonic for 100 is equal to 1st seasonal harmonic for 10
                [10, 100], [4, 9]
            ],
            [  # 2nd seasonal harmonic for 100 is equal to 1st seasonal harmonic for 50
                # 5th seasonal harmonic for 50 is equal to 1st seasonal harmonic for 10
                [10, 50, 100], [4, 4, 1]
            ],
            [  # 5th seasonal harmonic for 50 is equal to 2nd seasonal harmonic for 20
                [20, 50], [9, 4]
            ],
            [  # The better method also works with floats
                [25.5, 51], [12, 1]
            ],
        ]
    )
    def test_calculate_max_better(self, seasonal_periods, expected_max_harmonics):
        strategy = HarmonicsChoosingStrategy(Context(), checking_range=1)
        harmonics = strategy.calculate_max(
            np.asarray(seasonal_periods),
            HarmonicsChoosingStrategy.max_harmonic_dependency_reduction_better
        )
        assert np.array_equal(expected_max_harmonics, harmonics)

    @pytest.mark.parametrize(
        "n_jobs, max_harmonic, expected_range",
        [
            [  # no harmonics to check, return empty array
                12, 1, [],
            ],
            [  # only 1 harmonic to check
                12, 2, [2],
            ],
            [  # 5 harmonics to check and 5 cores, should contain all harmonics
                5, 6, range(2, 7),
            ],
            [  # 32 harmonics to check and 32 cores, should contain all harmonics
                32, 33, range(2, 34),
            ],
            [  # only 1 core and 1 harmonic to check
                1, 2, range(2, 3)
            ],
            [  # only 1 core but needs to check those 3 models anyway
                1, 12, range(5, 8)
            ],
            [  # only 1 core, will check 3 most complex models
                1, 5, range(3, 6)
            ],
            [  # 5 cores, should check models around 6
                5, 16, range(4, 9)
            ],
            [  # 6 cores, should check models around 6
                6, 16, range(3, 9)
            ],
            [  # 8 cores, should check all models around 6
                8, 11, range(2, 10)
            ],
            [  # 4 cores, range should cover all of the most complex cases
                4, 7, range(4, 8)
            ],
        ]
    )
    def test_initial_harmonics_to_check(self, n_jobs, max_harmonic, expected_range):
        strategy = HarmonicsChoosingStrategy(Context(), checking_range=n_jobs)
        obtained_range = strategy.initial_harmonics_to_check(max_harmonic)
        assert np.array_equal(expected_range, obtained_range)

    @pytest.mark.parametrize(
        "n_jobs, max_harmonic, chosen_harmonic, previous_range, expected_range",
        [
            [  # no harmonics to check, return empty array
                1, 1, 1, [], [],
            ],
            [  # previously checked are 4 and 3, we should check 2 now
                1, 4, 3, [3, 4], [2],
            ],
            [  # we should check higher orders of harmonics
                2, 10, 7, [5, 6, 7], [8, 9],
            ],
            [  # we should check all lower orders of harmonics
                8, 10, 5, [5, 6, 7], [2, 3, 4],
            ],
            [  # nothing to check, we already checked lower and higher order models
                8, 10, 6, [5, 6, 7], [],
            ],
            [  # we have already chosen the simplest model
                8, 10, 2, [2, 3, 4, 5, 6, 7], [],
            ],
            [  # we have already chosen the most complex model
                8, 4, 4, [2, 3, 4], [],
            ],
            [  # we are still choosing the simplest model, check lower level harmonics
                2, 12, 1, [5, 6, 7], [3, 4],
            ],
        ]
    )
    def test_next_harmonics_to_check(self, n_jobs, max_harmonic, chosen_harmonic, previous_range, expected_range):
        strategy = HarmonicsChoosingStrategy(Context(), checking_range=n_jobs)
        obtained_range = strategy.next_harmonics_to_check(
            max_harmonic=max_harmonic,
            previously_checked=previous_range,
            chosen_harmonic=chosen_harmonic
        )
        assert np.array_equal(expected_range, obtained_range)
