from . import ContextInterface
from .. import error
import multiprocessing as actual_processing
import multiprocessing.dummy as dummy_processing

class Context(ContextInterface):

    def __init__(self, show_warnings=True, n_jobs=None, multiprocessing_start_method=None):
        self.exception_handler = error.ExceptionHandler(show_warnings)
        self.n_jobs = n_jobs
        self.multiprocessing_start_method = multiprocessing_start_method

    def get_exception_handler(self):
        return self.exception_handler

    def multiprocessing(self):
        if self.n_jobs == 1:
            return dummy_processing
        if self.multiprocessing_start_method is None:
            return actual_processing
        return actual_processing.get_context(self.multiprocessing_start_method)
