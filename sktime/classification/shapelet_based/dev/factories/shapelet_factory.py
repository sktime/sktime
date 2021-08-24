from sktime.classification.shapelet_based.dev.shapelets.shapelet_base import ShapeletBase, ShapeletDependent, \
    ShapeleIndependent
import numpy as np
import math
from numba import njit


class ShapeletFactory():
    def get_random_shapelet(self, shapelet_size, instance_index, x, y) -> ShapeletBase:
        pass

    def get_distance(self, shapelet: ShapeletBase, xi) -> float:
        pass


class ShapeletFactoryDependent(ShapeletFactory):
    def get_random_shapelet(self, shapelet_size, instance_index, x) -> ShapeletDependent:
        return None

    def get_distance(self, shapelet: ShapeletDependent, xi) -> float:
        return 0


class ShapeletFactoryIndependent(ShapeletFactory):
    def get_random_shapelet(self, shapelet_size, instance_index, x, y) -> ShapeleIndependent:
        dimension_index = np.random.randint(0, len(x))
        start_position = np.random.randint(0, len(x[dimension_index]) - shapelet_size)
        end_position = start_position + shapelet_size

        return ShapeleIndependent(instance_index, start_position, shapelet_size, y, dimension_index,
                                  x[dimension_index][start_position:end_position].reset_index(drop=True))

    def get_distance(self, shapelet: ShapeleIndependent, xi) -> float:
        @njit(fastmath=True)
        def distance(a,b):
            d = np.linalg.norm(a - b)
            d = d * d
            d = 1.0 / len(a) * d
            return d

        @njit(fastmath=True)
        def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
            nrows = ((a.size - L) // S) + 1
            n = a.strides[0]
            return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

        xdim = xi[shapelet.dimension_id]

        subseqs = strided_app(xdim.to_numpy(), shapelet.length, 1)

        dists = np.apply_along_axis(distance, 1, subseqs)

        return min(dists)

    def get_distance_it(self, shapelet: ShapeleIndependent, xi) -> float:

        def s_dist(start, sh, xi):
            s_sum, temp = 0, 0

            for j in range(len(sh)):
                temp = sh[j] - xi[start + j]
                s_sum = s_sum + (temp * temp);

            return math.sqrt(s_sum)

        best_sum = 999999

        shapelet_data = shapelet.data.to_numpy()
        instance_data = xi[shapelet.dimension_id].to_numpy()
        shapelet_length = shapelet.length
        min_length = len(instance_data)

        for i in range(min_length - shapelet_length + 1):
            sum = s_dist(i, shapelet_data, instance_data)
            if sum < best_sum:
                best_sum = sum

        dist = 0.0 if best_sum == 0.0 else (1.0 / shapelet_length * best_sum)

        return dist
