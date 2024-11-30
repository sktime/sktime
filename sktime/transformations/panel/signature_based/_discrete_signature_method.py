import numpy as np
import pandas as pd
from itertools import combinations
from sktime.transformations.base import BaseTransformer

class DiscreteSignatureTransformer(BaseTransformer):
    def __init__(self, degree=2, use_index=False, use_smaller_equal=False):
        self.degree = degree
        self.use_index = use_index
        self.use_smaller_equal = use_smaller_equal

    def transform(self, X, y=None):
        data = X.to_numpy()

        if self.use_index:
            time_index = np.arange(data.shape[0]).reshape(-1, 1)
            data = np.hstack([time_index, data])

        n_dims = data.shape[1]
        combs = [combinations(range(n_dims), d) for d in range(1, self.degree + 1)]

        # Dictionary to store the transformed features
        features = {}

        # Calculate signature features for each combination
        for degree, comb_set in enumerate(combs, start=1):
            for comb in comb_set:
                column_name = "".join(map(str, comb))
                features[column_name] = self._compute_signature(data, comb)

        # Convert dictionary of features to a pandas DataFrame
        return pd.DataFrame([features])

    def _compute_signature(self, data, indices):
        selected_data = data[:, indices]
        n = selected_data.shape[0]
        product_sum = 0
        count = 0

        # Compute sum of products for pairs (i < j or i <= j depending on use_smaller_equal)
        for i in range(n):
            for j in range(i + int(not self.use_smaller_equal), n):
                product_sum += np.prod(selected_data[[i, j], :], axis=0)
                count += 1

        # Return the sum of products
        return product_sum / count if count > 0 else 0
