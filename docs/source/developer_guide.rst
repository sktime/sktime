
.. _developer_guide:

===============
Developer Guide
===============

Welcome to sktime's developer guide!

.. note::

    The developer guide is under development. We welcome contributions! For
    more details, please go to :issue:`464`.

New developers should:

* sign up to the developer Discord (see link in README) and say hello in the ``#contributors`` channel
* install a development version of ``sktime``, see :ref:`installation`
* set up CI tests locally and ensure they know how to check them remotely, see :ref:`continuous_integration`
* get familiar with the git workflow (:ref:`git_workflow`) and coding standards (:ref:`coding_standards`)
* feel free, at any point in time, to post questions on Discord, or ask core developers for help (see here for a `list of core developers <https://www.sktime.org/en/stable/about/team.html>`_)

* feel free to join the collaborative coding sessions for pair programming or getting help on developer set-up

Further special topics are listed below.

sktime follows `scikit-learn <https://scikit-learn.org/stable/>`_\ ’s API whenever possible.
If you’re new to scikit-learn, take a look at their `getting-started guide <https://scikit-learn.org/stable/getting_started.html>`_.
If you’re already familiar with scikit-learn, you may still learn something new from their `developers’ guide <https://scikit-learn.org/stable/developers/index.html>`_.

.. toctree::
   :maxdepth: 1

   installation
   developer_guide/git_workflow
   developer_guide/continuous_integration
   developer_guide/coding_standards
   developer_guide/add_estimators
   developer_guide/add_dataset
   developer_guide/deprecation
   developer_guide/dependencies
   developer_guide/documentation
   developer_guide/testing_framework
   developer_guide/release
