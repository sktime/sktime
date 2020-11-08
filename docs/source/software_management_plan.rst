.. _software_management_plan:

Software management plan
========================

Welcome to sktime's software management plan.

Contents
--------

.. contents:: :local:


Mission statement
^^^^^^^^^^^^^^^^^

sktime enables understandable and composable machine learning with time series. It provides scikit-learn compatible algorithms and model composition tools, supported by a clear taxonomy of learning tasks, with instructive documentation and a friendly community.


Goals
^^^^^

* Make the time series analysis ecosystem more understandable, usable and inter-operable;
* Ensure compatibility with foundational machine learning libraries such as scikit-learn;
* Provide state-of-the-art time series analysis capabilities;
* Build a more connected time series analysis community by connecting methodology experts with domain experts who work with time series data.

Project output
^^^^^^^^^^^^^^

* A unified Python framework toolbox for machine learning with time series;
* Documentation, tutorials and other educational materials for users and developers;
* Research publications on algorithm development, applied use cases, and machine learning software design;
* Workshops, sprints and talks for users and developers.

License
^^^^^^^

sktime is available under an open-source, permissive `BSD-3-clause license <https://github.com/alan-turing-institute/sktime/blob/master/LICENSE>`_.

Development roadmap
^^^^^^^^^^^^^^^^^^^

See our `development roadmap <https://www.sktime.org/en/latest/roadmap.html>`_ for an overview of ongoing and planned work.


Governance
^^^^^^^^^^

sktime is a consensus-based community project. Anyone with an interest in the project can join the community and participate in the project. How that participation takes place is described in our `governance document <https://www.sktime.org/en/latest/governance.html>`_.

Dependencies
^^^^^^^^^^^^

sktime is closely integrated with the scientific computing ecosystem in Python and depends on packages such as Numpy, pandas, scikit-learn, Cython and numba.

sktime also interfaces a number of packages from the time series analysis ecosystem in Python, including statsmodels, tsfresh and pmdarima among others.

Development operations
^^^^^^^^^^^^^^^^^^^^^^

sktime follows best practices for developing open-source software:

* We have implemented an extensive unit testing framework;
* We use GitHub and continuous integration services to automatically test new contributions before integrating them into our source code, including unit tests and code quality tests;
* We use GitHub for collaborative and open development practices, including a `process for enhancement proposals <https://github.com/sktime/enhancement-proposals>`_;
* We distribute pre-compiled files for different operation systems for user friendly installation
* We have implemented an automatic release pipeline to streamline the release process, including compilation, packaging and uploading of compiled files to PyPI.

Scientific references
^^^^^^^^^^^^^^^^^^^^^

* [paper] `Markus Löning, Anthony Bagnall, Sajaysurya Ganesh, Viktor Kazakov, Jason Lines, Franz Király (2019): “sktime: A Unified Interface for Machine Learning with Time Series” <http://learningsys.org/neurips19/assets/papers/sktime_ml_systems_neurips2019.pdf>`_
* [software] `Markus Löning, Tony Bagnall, Sajaysurya Ganesh, George Oastler, Jason Lines, ViktorKaz, …, Aadesh Deshmukh (2020). alan-turing-institute/sktime. Zenodo. http://doi.org/10.5281/zenodo.3749000 <http://doi.org/10.5281/zenodo.3749000>`_

Ackwnoledgements
^^^^^^^^^^^^^^^^

Our software management plan is inspired by the best practices and
recommendations found in:

* `Open Life Science mentorship program <https://openlifesci.org>`_
* `Software Sustainability Institute blog post <https://www.software.ac.uk/resources/guides/software-management-plans>`_
* `Elixir webinar <https://elixir-europe.org/events/webinar-software-management-plans>`_
