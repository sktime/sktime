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
* any other finite set :math:`\mathcal{C}`, called "categoriy set" in this context

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
* write :math:`\mbox{values}(s) := (x_1, \dots, x_T)`
* write :math:`\mbox{index}(s) := (t_1, \dots, t_T)`
* write :math:`\tau \in \mbox{index}(s)` if :math:`\tau = t_i` for some :math:`i`
* write :math:`s(\tau) := x_i` if :math:`\tau = t_i` and if 
there is a unique :math:`i` such that :math:`\tau = t_i`
* may choose to say that :math:`s` is a time series if :math:`\mathcal{T}` 
is totally ordered and (order)-isomorphic to a sub-set of :math:`\mathbb{R}`,
in a context where elements of :math:`\mathcal{T}` are interpreted as time stamps
* write :math:`\mbox{series}(\mathcal{X}, \mathcal{T})` for the set of all index series
with value domain :math:`\mathcal{X}`, index domain :math:`\mathcal{T}`

Key operations for the above indexed series :math:`s` are:

* inspecting :math:`\ell(s)`
* inspecting :math:`\mbox{index}(s)` or :math:`\mbox{values}(s)`
* for a :math:`\tau \in \mbox{index}(s)`, retrieve :math:`s(\tau)`
* inspect the value domain :math:`\mathcal{X}` or index domain :math:`\mathcal{T}` of :math:`s`
* sub-setting and selection operations

For an indexed series :math:`s\in \mbox{series}(\mathcal{X}, \mathcal{T})`, we say that
:math:`s` is:

* univariate, if :math:`\mathcal{X}` is a basic primitive domain
* multivariate, if :math:`s` is a primitive domain but not a basic primitive domain
* of non-primitive value domain, if :math:`s` is not a primitive domain

For a time series :math:`s\in \mbox{series}(\mathcal{X}, \mathcal{T})`, we say that
:math:`s` is:

* equally spaced, if :math:`t_{i+1} - t_i = c` for some :math:`c\in\mathbb{R}` and all
    :math:`i\in 1,\dots, T-1`
* unequally spaced, if :math:`s` is not equally spaced.

We use the scitype `Series` to type univariate and multivariate time series.

For `Series` typed,
:math:`\mbox{series}(\mathcal{X}, \mathcal{T})`-valued random variables,
we make no additional assumptions about the law (e.g., stationarity).


Notes on the `Series` scitype
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``sktime`` conceptual model differs in some points from
occasionally found conceptualization approaches in mathematical textbooks:

* defining time series only over a finite index set :math:`t_1,\dots, t_T`, 
instead of, say, defining :math:`s(t)` for all :math:`t` in an infinite or even
uncountably infinite set (e.g., the reals). This is because, in a computer,
time series observations are always finite. 
Some books do not make this distinction and conflate "generative" knowledge
in mathematical assumptions with observed information.
We made this decisions since real observations, including those that ``python`` can handle, are always finite.
* considering the index set as part of the time series and crucial, inspectable information.
Some sources conceptualize the index set as "external" to the time series, or irrelevant.
We think that considering the index set as part of the time series is conceptually natural for any user of ``pandas``.
* allowing the value domain to range over primitives, not over :math:`\mathbb{R}` or a subset thereof.
Again, having "data frame row" typed observations, and not just float arrays, is conceptually natural for any user of ``pandas``.


Time Series panels - the `Panel` scitype
----------------------------------------

An indexed series, with value domain :math:`\mathcal{X}`, index domain :math:`\mathcal{T}`,
of length :math:`T`, is a collection of index points :math:`t_1, \dots, t_T` and observations
:math:`x_1, \dots x_T`. We say (and interpret) that :math:`x_i` is the value observed at index
:math:`t_i`.

For an indexed series :math:`s`, with value domain :math:`\mathcal{X}`, index domain :math:`\mathcal{T}`,
with time points :math:`x_1, \dots x_T` and index points :math:`t_1, \dots, t_T`, we:

* write :math:`\ell(s) := T`, and call `\ell(s)` the "length of :math:`s`"
* write :math:`\mbox{values}(s) := (x_1, \dots, x_T)`
* write :math:`\mbox{index}(s) := (t_1, \dots, t_T)`
* write :math:`\tau \in \mbox{index}(s)` if :math:`\tau = t_i` for some :math:`i`
* write :math:`s(\tau) := x_i` if :math:`\tau = t_i` and if 
there is a unique :math:`i` such that :math:`\tau = t_i`
* may choose to say that :math:`s` is a time series if :math:`\mathcal{T}` 
is totally ordered and (order)-isomorphic to a sub-set of :math:`\mathbb{R}`,
in a context where elements of :math:`\mathcal{T}` are interpreted as time stamps
* write :math:`\mbox{series}(\mathcal{X}, \mathcal{T})` for the set of all index series
with value domain :math:`\mathcal{X}`, index domain :math:`\mathcal{T}`

Key operations for the above indexed series :math:`s` are:

* inspecting :math:`\ell(s)`
* inspecting :math:`\mbox{index}(s)` or :math:`\mbox{values}(s)`
* for a :math:`\tau \in \mbox{index}(s)`, retrieve :math:`s(\tau)`
* inspect the value domain :math:`\mathcal{X}` or index domain :math:`\mathcal{T}` of :math:`s`
* sub-setting and selection operations

For an indexed series :math:`s\in \mbox{series}(\mathcal{X}, \mathcal{T})`, we say that
:math:`s` is:

* univariate, if :math:`\mathcal{X}` is a basic primitive domain
* multivariate, if :math:`s` is a primitive domain but not a basic primitive domain
* of non-primitive value domain, if :math:`s` is not a primitive domain

For a time series :math:`s\in \mbox{series}(\mathcal{X}, \mathcal{T})`, we say that
:math:`s` is:

* equally spaced, if :math:`t_{i+1} - t_i = c` for some :math:`c\in\mathbb{R}` and all
    :math:`i\in 1,\dots, T-1`
* unequally spaced, if :math:`s` is not equally spaced.

We use the scitype `Series` to type univariate and multivariate time series.

For `Series` typed,
:math:`\mbox{series}(\mathcal{X}, \mathcal{T})`-valued random variables,
we make no additional assumptions about the law (e.g., stationarity).
