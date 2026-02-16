# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper functions for in-context covariates and regression."""

import itertools
import math

from sktime.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("jax", severity="none"):
    import jax
    import jax.numpy as jnp

import numpy as np
from sklearn import preprocessing

_TOL = 1e-6


def _unnest(nested):
    return np.array(list(itertools.chain.from_iterable(nested)))


def _repeat(elements, counts):
    return np.array(
        list(itertools.chain.from_iterable(map(itertools.repeat, elements, counts)))
    )


def _to_padded_jax_array(x):
    if x.ndim == 1:
        (i,) = x.shape
        di = 2 ** math.ceil(math.log2(i)) - i
        return jnp.pad(x, ((0, di),), mode="constant", constant_values=0.0)
    elif x.ndim == 2:
        i, j = x.shape
        di = 2 ** math.ceil(math.log2(i)) - i
        dj = 2 ** math.ceil(math.log2(j)) - j
        return jnp.pad(x, ((0, di), (0, dj)), mode="constant", constant_values=0.0)
    else:
        raise ValueError(f"Unsupported array shape: {x.shape}")


class BatchedInContextXRegBase:
    """BatchedInContextXRegBase."""

    def __init__(
        self,
        targets,
        train_lens,
        test_lens,
        train_dynamic_numerical_covariates=None,
        train_dynamic_categorical_covariates=None,
        test_dynamic_numerical_covariates=None,
        test_dynamic_categorical_covariates=None,
        static_numerical_covariates=None,
        static_categorical_covariates=None,
    ):
        self.targets = targets
        self.train_lens = train_lens
        self.test_lens = test_lens
        self.train_dynamic_numerical_covariates = (
            train_dynamic_numerical_covariates or {}
        )
        self.train_dynamic_categorical_covariates = (
            train_dynamic_categorical_covariates or {}
        )
        self.test_dynamic_numerical_covariates = test_dynamic_numerical_covariates or {}
        self.test_dynamic_categorical_covariates = (
            test_dynamic_categorical_covariates or {}
        )
        self.static_numerical_covariates = static_numerical_covariates or {}
        self.static_categorical_covariates = static_categorical_covariates or {}

    def _assert_covariates(self, assert_covariate_shapes=False):
        # Check presence.
        if (
            self.train_dynamic_numerical_covariates
            and not self.test_dynamic_numerical_covariates
        ) or (
            not self.train_dynamic_numerical_covariates
            and self.test_dynamic_numerical_covariates
        ):
            raise ValueError(
                "train_dynamic_numerical_covariates and"
                " test_dynamic_numerical_covariates must be both present or both"
                " absent."
            )

        if (
            self.train_dynamic_categorical_covariates
            and not self.test_dynamic_categorical_covariates
        ) or (
            not self.train_dynamic_categorical_covariates
            and self.test_dynamic_categorical_covariates
        ):
            raise ValueError(
                "train_dynamic_categorical_covariates and"
                " test_dynamic_categorical_covariates must be both present or both"
                " absent."
            )

        # Check keys.
        for dict_a, dict_b, dict_a_name, dict_b_name in (
            (
                self.train_dynamic_numerical_covariates,
                self.test_dynamic_numerical_covariates,
                "train_dynamic_numerical_covariates",
                "test_dynamic_numerical_covariates",
            ),
            (
                self.train_dynamic_categorical_covariates,
                self.test_dynamic_categorical_covariates,
                "train_dynamic_categorical_covariates",
                "test_dynamic_categorical_covariates",
            ),
        ):
            if w := set(dict_a.keys()) - set(dict_b.keys()):
                raise ValueError(
                    f"{dict_a_name} has keys not present in {dict_b_name}: {w}"
                )
            if w := set(dict_b.keys()) - set(dict_a.keys()):
                raise ValueError(
                    f"{dict_b_name} has keys not present in {dict_a_name}: {w}"
                )

        # Check shapes.
        if assert_covariate_shapes:
            if len(self.targets) != len(self.train_lens):
                raise ValueError(
                    "targets and train_lens must have the same number of elements."
                )

            if len(self.train_lens) != len(self.test_lens):
                raise ValueError(
                    "train_lens and test_lens must have the same number of elements."
                )

            for i, (target, train_len) in enumerate(zip(self.targets, self.train_lens)):
                if len(target) != train_len:
                    raise ValueError(
                        f"targets[{i}] has length "
                        f"{len(target)} != expected {train_len}."
                    )

            for key, values in self.static_numerical_covariates.items():
                if len(values) != len(self.train_lens):
                    raise ValueError(
                        f"static_numerical_covariates has key {key} with number of"
                        f" examples {len(values)} != expected {len(self.train_lens)}."
                    )

            for key, values in self.static_categorical_covariates.items():
                if len(values) != len(self.train_lens):
                    raise ValueError(
                        f"static_categorical_covariates has key {key} with number of"
                        f" examples {len(values)} != expected {len(self.train_lens)}."
                    )

            for lens, dict_cov, dict_cov_name in (
                (
                    self.train_lens,
                    self.train_dynamic_numerical_covariates,
                    "train_dynamic_numerical_covariates",
                ),
                (
                    self.train_lens,
                    self.train_dynamic_categorical_covariates,
                    "train_dynamic_categorical_covariates",
                ),
                (
                    self.test_lens,
                    self.test_dynamic_numerical_covariates,
                    "test_dynamic_numerical_covariates",
                ),
                (
                    self.test_lens,
                    self.test_dynamic_categorical_covariates,
                    "test_dynamic_categorical_covariates",
                ),
            ):
                for key, cov_values in dict_cov.items():
                    if len(cov_values) != len(lens):
                        raise ValueError(
                            f"{dict_cov_name} has key {key} with number of examples"
                            f" {len(cov_values)} != expected {len(lens)}."
                        )
                    for i, cov_value in enumerate(cov_values):
                        if len(cov_value) != lens[i]:
                            raise ValueError(
                                f"{dict_cov_name} has key {key} with its {i}-th example"
                                f" length {len(cov_value)} != expected {lens[i]}."
                            )

    def create_covariate_matrix(
        self,
        one_hot_encoder_drop="first",
        use_intercept=True,
        assert_covariates=False,
        assert_covariate_shapes=False,
    ):
        """create_covariate_matrix."""
        if assert_covariates:
            self._assert_covariates(assert_covariate_shapes)

        x_train, x_test = [], []

        # Numerical features.
        for name in sorted(self.train_dynamic_numerical_covariates):
            x_train.append(
                _unnest(self.train_dynamic_numerical_covariates[name])[:, np.newaxis]
            )
            x_test.append(
                _unnest(self.test_dynamic_numerical_covariates[name])[:, np.newaxis]
            )

        for covs in self.static_numerical_covariates.values():
            x_train.append(_repeat(covs, self.train_lens)[:, np.newaxis])
            x_test.append(_repeat(covs, self.test_lens)[:, np.newaxis])

        if x_train:
            x_train = np.concatenate(x_train, axis=1)
            x_test = np.concatenate(x_test, axis=1)

            # Normalize for robustness.
            x_mean = np.mean(x_train, axis=0, keepdims=True)
            x_std = np.where(
                (w := np.std(x_train, axis=0, keepdims=True)) > _TOL, w, 1.0
            )
            x_train = [(x_train - x_mean) / x_std]
            x_test = [(x_test - x_mean) / x_std]

        # Categorical features. Encode one by one.
        one_hot_encoder = preprocessing.OneHotEncoder(
            drop=one_hot_encoder_drop,
            sparse_output=False,
            handle_unknown="ignore",
        )
        for name in sorted(self.train_dynamic_categorical_covariates.keys()):
            ohe_train = _unnest(self.train_dynamic_categorical_covariates[name])[
                :, np.newaxis
            ]
            ohe_test = _unnest(self.test_dynamic_categorical_covariates[name])[
                :, np.newaxis
            ]
            x_train.append(np.array(one_hot_encoder.fit_transform(ohe_train)))
            x_test.append(np.array(one_hot_encoder.transform(ohe_test)))

        for covs in self.static_categorical_covariates.values():
            ohe = one_hot_encoder.fit_transform(np.array(covs)[:, np.newaxis])
            x_train.append(_repeat(ohe, self.train_lens))
            x_test.append(_repeat(ohe, self.test_lens))

        x_train = np.concatenate(x_train, axis=1)
        x_test = np.concatenate(x_test, axis=1)

        if use_intercept:
            x_train = np.pad(x_train, ((0, 0), (1, 0)), constant_values=1.0)
            x_test = np.pad(x_test, ((0, 0), (1, 0)), constant_values=1.0)

        return _unnest(self.targets), x_train, x_test

    def fit(self):
        """Fit."""
        raise NotImplementedError("Fit is not implemented.")


class BatchedInContextXRegLinear(BatchedInContextXRegBase):
    """Linear in-context regression model."""

    def fit(
        self,
        ridge=0.0,
        one_hot_encoder_drop="first",
        use_intercept=True,
        force_on_cpu=False,
        max_rows_per_col=0,
        max_rows_per_col_sample_seed=42,
        debug_info=False,
        assert_covariates=False,
        assert_covariate_shapes=False,
    ):
        """Fit."""
        flat_targets, x_train_raw, x_test = self.create_covariate_matrix(
            one_hot_encoder_drop=one_hot_encoder_drop,
            use_intercept=use_intercept,
            assert_covariates=assert_covariates,
            assert_covariate_shapes=assert_covariate_shapes,
        )

        x_train = x_train_raw.copy()
        if max_rows_per_col:
            nrows, ncols = x_train.shape
            if nrows > (w := ncols * max_rows_per_col):
                subsample = jax.random.choice(
                    jax.random.PRNGKey(max_rows_per_col_sample_seed),
                    nrows,
                    (w,),
                    replace=False,
                )
                x_train = x_train[subsample]
                flat_targets = flat_targets[subsample]

        device = jax.devices("cpu")[0] if force_on_cpu else None
        # Runs jitted version of the solvers which are quicker at the cost of
        # running jitting during the first time calling. Re-jitting happens whenever
        # new (padded) shapes are encountered.
        # Occasionally it helps with the speed and the accuracy if we force single
        # thread execution on cpu for accelerator machines:
        # 1. Avoid moving data to accelerator memory.
        # 2. Avoid precision loss if any.
        with jax.default_device(device):
            x_train_raw = _to_padded_jax_array(x_train_raw)
            x_train = _to_padded_jax_array(x_train)
            flat_targets = _to_padded_jax_array(flat_targets)
            x_test = _to_padded_jax_array(x_test)
            beta_hat = (
                jnp.linalg.pinv(
                    x_train.T @ x_train + ridge * jnp.eye(x_train.shape[1]),
                    hermitian=True,
                )
                @ x_train.T
                @ flat_targets
            )
            y_hat = x_test @ beta_hat
            y_hat_context = x_train_raw @ beta_hat if debug_info else None

        outputs = []
        outputs_context = []

        # Reconstruct the ragged 2-dim batched forecasts from flattened linear fits.
        train_index, test_index = 0, 0
        for train_index_delta, test_index_delta in zip(self.train_lens, self.test_lens):
            outputs.append(
                np.array(y_hat[test_index : (test_index + test_index_delta)])
            )
            if debug_info:
                outputs_context.append(
                    np.array(
                        y_hat_context[train_index : (train_index + train_index_delta)]
                    )
                )
            train_index += train_index_delta
            test_index += test_index_delta

        if debug_info:
            return outputs, outputs_context, flat_targets, x_train, x_test
        else:
            return outputs
