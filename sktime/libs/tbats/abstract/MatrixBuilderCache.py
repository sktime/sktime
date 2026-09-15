from . import MatrixBuilderInterface


class MatrixBuilderCache(MatrixBuilderInterface):

    def __init__(self, matrix_builder):
        """
        Wraps matrix_builder so that all matrices are calculated once only

        :param MatrixBuilderInterface matrix_builder: The builder that provides matrices
        """
        self.builder = matrix_builder
        self.w = None
        self.g = None
        self.F = None
        self.D = None

    def make_w_vector(self):
        if self.w is None:
            self.w = self.builder.make_w_vector()
        return self.w

    def make_g_vector(self):
        if self.g is None:
            self.g = self.builder.make_g_vector()
        return self.g

    def make_F_matrix(self):
        if self.F is None:
            self.F = self.builder.make_F_matrix()
        return self.F

    def calculate_D_matrix(self):
        if self.D is None:
            self.D = self.builder.calculate_D_matrix()
        return self.D
