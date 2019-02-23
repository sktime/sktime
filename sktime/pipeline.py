from sklearn.pipeline import Pipeline


class TSPipeline(Pipeline):
    def __init__(self, steps, memory=None, random_state=None, check_input=True):
        super(TSPipeline, self).__init__(steps, memory=memory)
        self.random_state = random_state
        self.check_input = check_input

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state

        # If random state is set for entire pipeline, set random state for all random components
        if random_state is not None:
            for step in self.steps:
                if hasattr(step[1], 'random_state'):
                    step[1].set_params(**{'random_state': self.random_state})

    @property
    def check_input(self):
        return self._check_input

    @check_input.setter
    def check_input(self, check_input):
        self._check_input = check_input

        # If check_input is set for entire pipeline, set check input for all components
        if not check_input:
            for step in self.steps:
                if hasattr(step[1], 'check_input'):
                    step[1].set_params(**{'check_input': self.check_input})

