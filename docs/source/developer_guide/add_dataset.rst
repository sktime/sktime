.. _developer_guide_add_datset:

====================
Adding a New Dataset
====================

Follow these steps to add a new dataset to sktime:

*  Include CSV file or other supported format under :code:`sktime/datasets/data/<dataset-name>`
*  Add :code:`load_<dataset-name>(...)` function in file :code:`sktime/datasets/base.py`
*  Add :code:`<dataset-name>` to the list :code:`__all__ = [...]` in file :code:`sktime/datasets/__init__.py`
*  Add :code:`<dataset-name>` as argument to method :code:`included_datasets = (...` in file :code:`sktime/sktime/datasets/setup.py`
*  Add :code:`<dataset-name>` to the list of included problems in file :code:`sktime/sktime/datasets/setup.py`
