# -*- coding: utf-8 -*-
class _NonSequentialPipelineStepResultsMixin:
    """Mixin class for pipelines in which the steps are not executed sequentially"""

    def __init__(self):
        self._step_result = None

    @property
    def step_result(self):
        return self._step_result

    @step_result.setter
    def step_result(self, value):
        self._step_result = value
