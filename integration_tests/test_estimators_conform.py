from sktime.utils.estimator_checks import check_ts_estimator
import inspect
import pyclbr
import os


directory_list =['../sktime/classifiers/','../sktime/regressors/']

def test_estimators_conform():
    for directory in directory_list:
        for module in os.listdir(os.path.dirname(directory)):
            if module == '__init__.py' or module[-3:] != '.py':
                continue
            
            module_name = 'sktime.classifiers.'+module[:-3]
            module_info = pyclbr.readmodule(module_name)

            return check_ts_estimator(module_info.values)
