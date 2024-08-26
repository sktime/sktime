====================
Adding a New Dataset
====================

To add a new dataset into :code:`sktime` internal dataset repository, please proceed with the following steps:

1. From the root of your :code:`sktime` local repository, create a :code:`<dataset-name>` folder:

   .. code-block:: shell

      mkdir ./datasets/data/<dataset-name>

2. In the above directory, add your dataset file :code:`<dataset-name>.<EXT>`, where :code:`<EXT>` is the file extension:

   * The list of supported file formats is available in the :code:`sktime/MANIFEST.in` file (*e.g.*, :code:`.csv`, :code:`.txt`).
   * If your file format ``<EXT>`` does not figure in the list, simply add it in the :code:`sktime/MANIFEST.in` file:
   ::

      "sktime/MANIFEST.in"
      ...
      recursive-include sktime/datasets *.csv ... *.<EXT>
      ...

3. In ``sktime/datasets/_single_problem_loaders.py``, declare a :code:`load_<dataset-name>(...)` function. Feel free to use any other declared functions as templates for either classification or regression datasets.

4. In ``sktime/datasets/__init__.py``, append :code:`"load_<dataset-name>"` to the list :code:`__all__`.

5. In ``sktime/datasets/setup.py``, append :code:`"<dataset-name>"` to the tuple :code:`included_datasets`.
