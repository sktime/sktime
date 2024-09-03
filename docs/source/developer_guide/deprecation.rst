.. _developer_guide_deprecation:

===========
Deprecation
===========

``sktime`` aims to be stable and reliable towards its users.
Our high-level policy to ensure this is:

"``sktime`` should never break user code without a clear and actionable warning
given at least one (MINOR) release cycle in advance."

Here, "break" expressly includes a change to abstract logic, such as the algorithm
being used, not just changes that lead to exceptions or performance degradation.

For instance, if a user has code

.. code:: python

    from sktime.forecasting.foo import BarForecaster

    bar = BarForecaster(42, x=43)
    bar.fit(y_train, fh=[1, 2, 3])
    y_pred = bar.predict()

then no release of ``sktime`` should change, without warning:

* import location of ``BarForecaster``
* argument signature of ``BarForecaster``, including name, order, and defaults of arguments
* the abstract algorithm that ``BarForecaster`` carries out for the given arguments

Changes that can be carried out without warning:

* adding more arguments at the end of the argument list, with a default value that retains prior behaviour,
  as long as the new arguments are well-documented
* pure refactoring of internal code, as long as the public API remains the same
* changing the implementation without changing the abstract algorithm, e.g., for performance reasons

The deprecation policy outlined in this document provides details on how to carry out
changes that need change or deprecation handling, in a user-friendly and reliable way.

It is accompanied by formulaic patterns for developers, with examples,
and a process for release managers, to make the policy easy to follow.


Deprecation policy
==================

sktime `releases <https://github.com/sktime/sktime/releases>`_ follow `semantic versioning <https://semver.org>`_.
A release number denotes <major>.<minor>.<patch> versions.

Our current deprecation policy is as follows:

* all interface breaking (not downwards compatible) changes to public interfaces must be accompanied by deprecation.
  Examples: changes to defaults of existing parameters, removal of parameters.
  Non-examples: new parameters with a default value that leads to prior behaviour.
* such changes or removals happen only at MINOR or MAJOR versions, not at PATCH versions.
* deprecation warnings must be included for at least one full MINOR version cycle before change or removal.
  Therefore, typically, the change or removal happens at the *second* next MINOR release.

Example timeline:

1. developer A resolves, at current state v0.9.3, to remove functionality X
at some point in the near future.

2. therefore, by the above, we should introduce a deprecation message, visible from next release (e.g., v0.9.4),
which says that functionality will be removed at v0.11.0

3. developer A makes a pull request to remove functionality X which includes that deprecation warning.
The pull request is reviewed by core developers, with the suggestion by developer A accepted or rejected.

4. If accepted and merged before v0.10.0 release, the PR goes in the next release, with a deprecation note in the release notes.
If PR acceptance takes until after v0.10.0 but before v0.11.0, the planned removal moves to v0.12.0 and the warning needs to be updated.

5. an additional PR to remove deprecation warning and functionality X is prepared by
developer A, for v0.12.0 but not merged

6. a release manager merges the PR in part 5 as part of the release v0.12.0, effecting the removal.
Release notes of v0.12.0 includes a removal note.

Deprecation and change process
==============================

The general deprecation/change process consists of two parts:

* scheduling of a deprecation/change by a developer
* deprecation/change actions carried out by a release manager

The developer sided process takes place in PR made by the developer proposing the deprecation, and is as follows:

* **Raise a warning.** For all deprecated functionality, we raise a :code:`DeprecationWarning` if the change is scheduled within the next two MINOR version cycles.
  Otherwise a :code:`FutureWarning` is also acceptable.
* **The warning should be instructive to the user.**
  The warning message should give the version number when the functionality will be changed, describe the new usage
  and any transitional actions in downstream code, with clearly stated timelines (specified versions) of expected changes.
* **Docstrings should be updated to reflect the deprecation.** Docstrings should be updated to reflect the deprecation/change.
  This typically includes deprecation timelines, pre/post deprecation functionality.
* **Add a TODO comment in the code for the release manager.**
  Add a TODO comment to all pieces of code that should be removed or changed, e.g.,: :code:`TODO: remove in v0.11.0`.
  The TODO comment should describe all actions in explicit detail (e.g. removal of arguments, removal of functions or blocks of code).
  If changes need to be applied across multiple places, place multiple TODO comments.
  Ensure the result of the TODO actions is tested and does not lead to test breakage when actioned by the release manager.
  This is best accompanied by a prepared PR that the release manager only needs to merge.
* as all tech decisions, deprecations/changes are first proposed in a PR and need to be reviewed by other developers.

The release manager process happens at every release and is as follows:

* **Summarize any scheduled deprecations and changes in the changelog.**: As soon as a deprecation/change is scheduled,
  it should be announced in the "deprecations and changes" section of the changelog, with exact version timelines,
  and any actions to be carried out by users or maintainers of third party extensions (usage and extension contracts).
* **Carry out deprecation and change actions.** As part of every release process at a MINOR or MAJOR version,
  the release manager searches all deprecated functionality that is due to be removed will be removed by searching for the TODO comments.
  These will be carried out as described.
  If the action results in CI failure, the release manager should open an issue and contact the developer for swift resolution,
  and possibly move the action to the next release cycle if this would unduly delay the release process.
* **Summarize any actioned deprecations and changes in the changelog.**: All deprecations and changes that have been
  carried out should be summarized in the "deprecations and changes" section of the changelog.

Special deprecations
====================

This section outlines the deprecation process for some advanced cases.

Deprecating and change of parameters
------------------------------------

The following are common cases of deprecation or change around parameters
of functions or classes (e.g., estimators):

* changing the default value of a parameter
* renaming a parameter
* adding a parameter with a default value that changes prior behaviour
* changing the sequence of parameters
* removing a parameter

In all cases, it needs to be ensured that:

* warnings are raised in cases where user logic would change
* the warning message includes a complete recipe for how to change the code,
  to retain current behaviour, or change to alternative behaviour
* sufficient notice is given, i.e., the warning message is present for at least
  one MINOR version cycle before the change is carried out
* "todo" comments are left for the release managers to carry out the change,
  and optimally a merge-ready change branch/PR is provided, to be merged at the
  scheduled version of change

No such warning is necessary if no working user logic would change, this is the case if:

* a parameter is added with a default value that retains prior behaviour,
  at the end of the parameter list
* a parameter is removed where non-defaults would always raise unexpected exceptions

Recipes for individual cases above follow.

Fully worked examples for some of these cases are given in the
last section of this document, "Examples to illustrate recipes".

Changing the default value of a parameter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To change the default value of a parameter, follow steps 1-3 in the pull request
implementing the change.

1. at current version, change the default value to ``"changing_value"``.
Internally, add logic that overrides the value of the parameter with the old default
value, if the parameter is set to ``"changing_value"``. If the parameter is an
``__init__`` parameter of an estimator class,
the value cannot be directly overridden, but this needs to be done in a private
parameter copy, since all ``__init__`` parameters must be written
to ``self`` unchanged. I.e., write the parameter to ``self._<param_name>`` unchanged,
and add logic that overrides the value of ``self._<param_name>`` with the old default,
and ensure to use ``self._<param_name>`` in the rest of the code instead of
``self.<param_name>``.

2. add a warning, using ``sktime.utils.warnings.warn``, if the parameter is called
with a non-default. This warning should always include the name of the estimator/function,
the version of change, and a clear instruction on how to change the code to retain
prior behaviour. E.g., ``"Parameter <param_name> of <estimator_name> will change
default value from <old_value> to <new_value> in sktime version <version_number>.
To retain prior behaviour, set <param_name> to <old_value> explicitly"``.

3. add a TODO comment to the code, to remove the warning and change the default value,
in the next MINOR version cycle. E.g., add the comment
``# TODO <version_number>: change default of <param_name> to <new_value>,
update docstring, and remove warning``,
at the top of the function or class where the parameter is defined.

4. the release manager will carry out the TODO action in the next MINOR version cycle,
and remove the TODO comment. Optimally, a change branch is provided that the
release manager can merge, and its PR ID is mentioned in the todo.

Renaming a parameter
~~~~~~~~~~~~~~~~~~~~

To rename a parameter, follow steps 1-6 in the pull request
implementing the change.

1. at current version, add a parameter with the new name at the end of the
list of parameters, with the same default value as the old parameter.
Do not remove the old parameter.

2. change the value of the old parameter to the string ``"deprecated"``.
Change all code in the function or class that uses the old parameter to use
the new parameter instead. This can be done by a bulk-replace.

3. at the start of the function or class init, add logic that overrides the value
of the new parameter with the value of the old parameter, if the old parameter
is not ``"deprecated"``. If the parameter is an ``__init__`` parameter
of an estimator class,
the value cannot be directly overridden, but this needs to be done in a private
parameter, since all ``__init__`` parameters must be written to ``self`` unchanged.

4. add a warning, using ``sktime.utils.warnings.warn``, if the old parameter is called
with a non-default. This warning should always include the name of the estimator/function,
the version of change, and a clear instruction on how to change the code to retain
prior behaviour. E.g., ``"Parameter <param_name> of <estimator_name> will be renamed
from <old_name> to <new_name> in sktime version <version_number>.
To retain prior behaviour, use a kwargs call of <new_name> instead of <old_name>"``.

5. update the docstring of the function or class to refer only to the new parameter.

6. add a TODO comment to the code, to remove the warning and change the default value,
in the next MINOR version cycle. E.g., add the comment
``# TODO <version_number>: change name of parameter <old_name> to <new_name>,
remove old parameter at the end, and remove warning``,
at the top of the function or class where the parameter is defined.

7. the release manager will carry out the TODO action in the next MINOR version cycle,
  and remove the TODO comment. Optimally, a change branch is provided that the
  release manager can merge, and its PR ID is mentioned in the todo.

Adding a parameter with a default value that changes prior behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This should be done in two steps:

* adding the parameter, but with a default value that retains prior behaviour.
  As this preserves prior behaviour, no deprecation or change mechanism is necessary.
* then, follow the steps for changing the default value of a parameter, above.

Changing the sequence of parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This type of change should be avoided, as it it is difficult to carry out.
If instead one of the above change patterns can be used, that is preferred.

To change the sequence of parameters, follow steps 1-6 in the pull request
implementing the change.

1. at current version, change the defaults of all parameters after and including
the first parameter to change position to ``"position_change"``.

2. Internally, add logic that overrides the value of the parameter with the old default
value, if the parameter is set to ``"position_change"``.
For ``__init__`` parameters of an estimator class,
the values cannot be directly overridden, but this needs to be done in a private
parameter copy, since all ``__init__`` parameters must be written
to ``self`` unchanged. I.e., write the parameter to ``self._<param_name>`` unchanged,
and add logic that overrides the value of ``self._<param_name>`` with the old default,
and ensure to use ``self._<param_name>`` in the rest of the code instead of
``self.<param_name>``.

3. add a warning, using ``sktime.utils.warnings.warn``, if any of the position changing
parameters are called with a non-default. This warning should always include
the name of the estimator/function, the version of change, and a clear instruction
on how to change the code to retain prior behaviour. The instruction
should direct the user to use ``kwargs`` calls instead of positional calls, for
all parameters that change position.

4. add a TODO comment to the code, to remove the warning and change the sequence,
as well as changing default values to the old defaults,
in the next MINOR version cycle.
The TODO comment should contain complete lines of code.
Optimally, a change branch is provided that the
release manager can merge, and its PR ID is mentioned in the todo.

Removing a parameter
~~~~~~~~~~~~~~~~~~~~

If the parameter is removed a position that is not at the end of the parameter list,
it should be first moved to the end o the parameter list.

For removal of a parameter, follow the steps of "changing the default value",
with a different warning message, namely that the parameter will be removed.

The error message should contain details on whether prior behaviour can be retained,
if yes in which cases, and if yes, how.


Deprecating tags
----------------

To deprecate tags, it needs to be ensured that warnings are raised when the tag is used.
There are two common scenarios: removing a tag, or renaming a tag.

For either scenario, the helper class ``TagAliaserMixin`` (in ``sktime.base``) can be used.

To deprecate tags, add the ``TagAliaserMixin`` to ``BaseEstimator``, or another ``BaseObject`` descendant.
It is advised to select the youngest descendant that fully covers use of the deprecated tag.
``TagAliaserMixin`` overrides the tag family of methods, and should hence be the first class to inherit from
(or in case of multiple mixins, earlier than ``BaseObject``).

``alias_dict`` in ``TagAliaserMixin`` contains a dictionary of deprecated tags:
For removal, add an entry ``"old_tag_name": ""``.
For renaming, add an entry ``"old_tag_name": "new_tag_name"``
``deprecate_dict`` contains the version number of renaming or removal, and should have the same keys as ``alias_dict``.

The ``TagAliaserMixin`` class will ensure that new tags alias old tags and vice versa, during
the deprecation period. Informative warnings will be raised whenever the deprecated tags are being accessed.

When removing/renaming tags after the deprecation period,
ensure to remove the removed tags from the dictionaries in ``TagAliaserMixin`` class.
If no tags are deprecated anymore (e.g., all deprecated tags are removed/renamed),
ensure to remove this class as a parent of ``BaseObject`` or ``BaseEstimator``.

Examples to illustrate recipes
==============================

Below are example templates for some of the cases above.
The examples are carried out for a class with ``fit`` / ``predict`` methods,,
but the same principles apply to functions, or classes with other APIs.

Changing the default value of a parameter
-----------------------------------------

Code before any change
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    class EstimatorName:
        """The old docstring.

        Parameters
        ----------
        parameter : str, default="old_default"
            The parameter description.
        """
        def __init__(self, parameter="old_default"):
            self.parameter = parameter

        def fit(self, X, y):
            parameter = self.parameter
            # Fit the model using parameter
            fitting_logic(parameter)
            return self

        def predict(self, X):
            parameter = self.parameter
            # Predict using the fitted model
            y_pred = prediction_logic(parameter)
            return y_pred

Step 1: during deprecation period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is done by the developer, in a PR.
Optionally, the developer can prepare a PR for step 2
that the release manager can merge.

.. code:: python

    from sktime.utils.warnings import warn

    # TODO (release <MAJOR>.<MINOR>.0)
    # change the default of 'parameter' to <new_value>
    # update the docstring for parameter
    class EstimatorName:
        """The old docstring with deprecation info.

        Parameters
        ----------
        parameter : str, default="old_default"
            The parameter description.
            Default value of parameter will change to <new_value>
            in version '<MAJOR>.<MINOR>.0'.
        """
        def __init__(self, parameter="changing_value"):
            self.parameter = parameter
            # TODO (release <MAJOR>.<MINOR>.0)
            # change the default of 'parameter' to <new_value>
            # remove the following 'if' check
            # de-indent the following 'else' check
            if parameter == "changing_value":
                warn(
                    "in `EstimatorName`, the default value of parameter 'parameter'"
                    " will change to <new_value> in version '<MAJOR>.<MINOR>.0'. "
                    "To keep current behaviour and to silence this warning, "
                    "set 'parameter' to 'old' explicitly.",
                    category=DeprecationWarning,
                    obj=self,
                )
                self._parameter = "old_default"
            else:
                self._parameter = parameter

        def fit(self, X, y):
            parameter = self._parameter
            # Fit the model using parameter
            fitting_logic(parameter)
            return self

        def predict(self, X):
            parameter = self._parameter
            # Predict using the fitted model
            y_pred = prediction_logic(parameter)
            return y_pred

Step 2: after deprecation period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is done by the release manager, either by merging a prepared PR,
or by carrying out the TODO action.

.. code:: python

    class EstimatorName:
        """The final docstring.

        Parameters
        ----------
        parameter : str, default="new_default"
            The parameter description.
        """
        def __init__(self, parameter="new_default"):
            self.parameter = parameter
            self._parameter = parameter

        def fit(self, X, y):
            parameter = self._parameter
            # Fit the model using parameter
            fitting_logic(parameter)
            return self

        def predict(self, X):
            parameter = self._parameter
            # Predict using the fitted model
            y_pred = prediction_logic(parameter)
            return y_pred

Optionally, use of the private parameter ``self._parameter`` can be removed,
and replaced by ``self.parameter``,
if it is not used elsewhere in the code.

Renaming a parameter
--------------------

Code before any change
~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    class EstimatorName:
        """The old docstring.

        Parameters
        ----------
        old_parameter : str, default="default"
            The parameter description.
        """

        def __init__(self, old_parameter="default"):
            self.old_parameter = old_parameter

        def fit(self, X, y):
            old_parameter = self.old_parameter
            # Fit the model using parameter
            fitting_logic(old_parameter)
            return self

        def predict(self, X):
            old_parameter = self.old_parameter
            # Predict using the fitted model
            y_pred = prediction_logic(old_parameter)
            return y_pred

Step 1: during deprecation period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is done by the developer, in a PR.
Optionally, the developer can prepare a PR for step 2
that the release manager can merge.

.. code:: python

   from sktime.utils.warnings import warn

    class EstimatorName:
        """The old docstring, but already points to the new name.

        The docstring should replace 'old_parameter' with 'new_parameter',
        and no longer mention 'old_parameter'.

        Parameters
        ----------
        new_parameter : str, default="default"
            The parameter description.
        """
        def __init__(self, old_parameter="deprecated", new_parameter="default"):
            # IMPORTANT: both params need to be written to self during change period
            self.new_parameter = new_parameter
            self.old_parameter = old_parameter
            # TODO (release <MAJOR>.<MINOR>.0)
            # remove the 'old_parameter' argument from '__init__' signature
            # move 'new_parameter' to the position of 'old_parameter'
            # remove the following 'if' check
            # de-indent the following 'else' check
            if old_parameter != "deprecated":
                warn(
                    "in `EstimatorName`, parameter 'old_parameter'"
                    " will be renamed to new_parameter in version '<MAJOR>.<MINOR>.0'. "
                    "To keep current behaviour and to silence this warning, "
                    "use 'new_parameter' instead of 'old_parameter', "
                    "set new_parameter explicitly via kwarg, and do not set"
                    " old_parameter.",
                    category=DeprecationWarning,
                    obj=self,
                )
                self._parameter = old_parameter
            else:
                self._parameter = new_parameter

       def fit(self, X, y):
            old_parameter = self._parameter
            # Fit the model using parameter
            fitting_logic(old_parameter)
            return self

       def predict(self, X):
            old_parameter = self._parameter
            # Predict using the fitted model
            y_pred = prediction_logic(old_parameter)
            return y_pred

Step 2: after deprecation period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This step is done by the release manager, either by merging a prepared PR,
or by carrying out the TODO action.

.. code:: python

    class EstimatorName:
        """Same as in step 2, no change necessary.

        Parameters
        ----------
        new_parameter : str, default="default"
            The parameter description.
        """
       def __init__(self, new_parameter="default"):
           self.new_parameter = new_parameter
           self._parameter = new_parameter

       def fit(self, X, y):
            old_parameter = self._parameter
            # Fit the model using parameter
            fitting_logic(old_parameter)
            return self

       def predict(self, X):
            old_parameter = self._parameter
            # Predict using the fitted model
            y_pred = prediction_logic(old_parameter)
            return y_pred

Optionally, use of the private parameter ``self._parameter`` can be removed,
and replaced by ``self.new_parameter``,
if it is not used elsewhere in the code.
