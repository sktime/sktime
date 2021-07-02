.. _developer_guide_forecasting:

Adding Datasets to sktime
===========
Follow these steps to add a new dataset to sktime:

*  Include CSV file or supported other format under :code:`sktime/datasets/data/<dataset-name>`
*  Add :code:`load_<dataset-name>(...)` function in file :code:`sktime/datasets/base.py`
*  Add :code:`<dataset-name>` to the list :code:`__all__ = [...` in file :code:`sktime/datasets/__init__.py`
*  Add :code:`<dataset-name>` as argument to method :code:`included_datasets = (...` in file :code:`sktime/sktime/datasets/setup.py`
*  Add import statement :code:`from sktime.datasets.base import load_<dataset-name>(...)` in file :code:`sktime/sktime/datasets/setup.py`

Thank you for your contribution!
