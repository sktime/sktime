class MatrixBuilderInterface(object):

    def make_w_vector(self):
        raise NotImplementedError()

    def make_g_vector(self):
        raise NotImplementedError()

    def make_F_matrix(self):
        raise NotImplementedError()

    def calculate_D_matrix(self):
        raise NotImplementedError()
