.. _sciref_series:

Scientific reference: time series
=================================

``sktime``'s conceptual model is based on a number of "time series" related mathematical
objects:

* "time series", observations of a single process made over time.
Example: average temperature in London, on a given day.
* "time series panels", collections of multiple time series, independently observed.
Example: fever curves of 10 specific hospital patients, for the time in the hospital.
* "hierarchical time series", collections of multiple time series where individual
instances are observed at instances defined by variables defining the hierarchy.
Example: gummy bear sales at 100 specific stores, in 10 federal states.

This document provides a formal reference for the above.
The objects are defined in terms of:

* abstract data types (ADT) - defines domain, type and common operations
* scientific types (scitype) - ADT plus statistical assumptions 


Primitive values - the `Primitives` scitype
-------------------------------------------

The following sets/domains are called basic primitive domains:

* the real numbers, i.e., the set :math:`\mathbb{R}`
* the set of boolean truth value, i.e., the two-element set containing `True` or `False`
* the set of all finite length strings of ASCII characters
* any other finite set :math:`\mathcal{C}`, called "categories" in this context

A set/domain is called primitive domain if it is a basic primitive domain, or 
finite Cartesian products of basic primitive domains.

An object is said to be of primitive type if it is an element of a primitive domain.

We use the scitype `Primitives` to type objects of primitive type.


Time Series - the `Series` scitype
----------------------------------

An indexed series, with value domain :math:`\mathcal{X}`, index domain :math:`\mathcal{T}`,
of length :math:`T`, is a collection of index points :math:`t_1, \dots, t_T` and observations
:math:`x_1, \dots x_T`. We say (and interpret) that :math:`x_i` is the value observed at index
:math:`t_i`.

For an indexed series :math:`s`, with value domain :math:`\mathcal{X}`, index domain :math:`\mathcal{T}`,
with time points :math:`x_1, \dots x_T` and index points :math:`t_1, \dots, t_T`, we:

* write :math:`\ell(s) := T`, and call `\ell(s)` the "length of :math:`s`"
* write :math:`s(\tau) := x_i` if :math:`\tau = t_i` and if 
there is a unique :math:`i` such that :math:`\tau = t_i`
* write :math:`\mbox{index}(s) := (t_1, \dots, t_T)`
* write :math:`\mbox{values}(s) := (x_1, \dots, x_T)`
* may choose to say that :math:`s` is a time series if :math:`\mathcal{T}` 
is totally ordered and (order)-isomorphic to a sub-set of :math:`\mathbb{R}`,
in a context where elements of :math:`\mathcal{T}` are interpreted as time stamps
* write :math:`\mbox{series}(\mathcal{X}, \mathcal{T})` for the set of all index series
with value domain :math:`\mathcal{X}`, index domain :math:`\mathcal{T}`


For frequent issues with installation, consult the `Release versions - troubleshooting`_ section.

Installing sktime from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime`` releases are available via PyPI and can be installed via ``pip`` using:

.. code-block:: bash

    pip install sktime

This will install ``sktime`` with core dependencies, excluding soft dependencies.

To install ``sktime`` with maximum dependencies, including soft dependencies, install with the ``all_extras`` modifier:

.. code-block:: bash

    pip install sktime[all_extras]


Installing sktime from conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``sktime`` releases are available via ``conda`` from ``conda-forge``.
They can be installed via ``conda`` using:

.. code-block:: bash

    conda install -c conda-forge sktime

This will install ``sktime`` with core dependencies, excluding soft dependencies.

To install ``sktime`` with maximum dependencies, including soft dependencies, install with the ``all-extras`` recipe:

.. code-block:: bash

    conda install -c conda-forge sktime-all-extras

Note: currently this does not include dependencies ``catch-22``, ``pmdarima``, and ``tbats``.
As these packages are not available on ``conda-forge``, they must be installed via ``pip`` if desired.
Contributions to remedy this situation are appreciated.

