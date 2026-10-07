from ..abstract import ComponentMatrix, SeedFinder as AbstractSeedFinder
from . import Components


class SeedFinder(AbstractSeedFinder):

    def __init__(self, components):
        """
        :param Components components:
        :return:
        """
        super().__init__(components)
        self.mask = None

    def to_matrix_for_linear_regression(self, w_tilda):
        w_tilda_obj = ComponentMatrix(w_tilda, self.components).without_arma()
        return w_tilda_obj.as_matrix()

    def from_linear_regression_coefs_to_x0(self, linear_regression_coefs):
        lr_coefs = ComponentMatrix(
            linear_regression_coefs,
            model_components=self.components,
            append_arma_coefs=True  # ARMA is not part of linear_regression so we need to re-add it
        )
        return lr_coefs.as_vector()
