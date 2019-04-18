from Cython.Build import cythonize

cythonize(
          ["sktime/distances/elastic_cython.pyx"],
          annotate=True)