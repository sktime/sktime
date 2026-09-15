class ContextInterface(object):

    def get_exception_handler(self):
        # abstract method
        raise NotImplementedError()

    def create_constant_model(self, constant_value):
        # abstract method
        raise NotImplementedError()

    def create_model(self, params, validate_input=True):
        # abstract method
        raise NotImplementedError()

    def create_seed_finder(self, components):
        # abstract method
        raise NotImplementedError()

    def create_matrix_builder(self, params):
        # abstract method
        raise NotImplementedError()

    def create_default_starting_params(self, y, components):
        # abstract method
        raise NotImplementedError()

    def create_params_optimizer(self):
        # abstract method
        raise NotImplementedError()

    def create_case(self, components):
        # abstract method
        raise NotImplementedError()

    def create_case_from_dictionary(self, **components_dictionary):
        # abstract method
        raise NotImplementedError()

    def create_components(self, **components):
        # abstract method
        raise NotImplementedError()

    def create_harmonics_choosing_strategy(self):
        # abstract method
        raise NotImplementedError()

    def multiprocessing(self):
        # abstract method
        raise NotImplementedError()