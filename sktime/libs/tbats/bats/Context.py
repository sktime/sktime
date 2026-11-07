from ..abstract import MatrixBuilderCache, Context as AbstractContext
from . import *


class Context(AbstractContext):

    def create_constant_model(self, constant_value):
        return self.create_model(
            ModelParams(components=Components.create_constant_components(), alpha=0, x0=[constant_value])
        )

    def create_model(self, params, validate_input=True):
        return Model(
            params,
            validate_input=validate_input,
            context=self,
        )

    def create_seed_finder(self, components):
        return SeedFinder(components)

    def create_matrix_builder(self, params):
        base_builder = MatrixBuilder(params)
        return MatrixBuilderCache(base_builder)

    def create_default_starting_params(self, y, components):
        return ModelParams.with_default_starting_params(y, components=components)

    def create_params_optimizer(self):
        return ParamsOptimizer(self)

    def create_case_from_dictionary(self, **components_dictionary):
        return self.create_case(components=self.create_components(**components_dictionary))

    def create_case(self, components):
        return Case(components=components, context=self)

    def create_components(self, **components):
        return Components(**components)
