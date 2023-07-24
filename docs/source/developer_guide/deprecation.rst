.. _developer_guide_deprecation:

===========
Deprecation
===========

Before we can make changes to sktime's user interface, we need to make sure that users have time to make the necessary adjustments in their code.
For this reason, we first need to deprecate functionality and change it only in a next release.

.. note::

    For upcoming changes and next releases, see our `Milestones <https://github.com/sktime/sktime/milestones?direction=asc&sort=due_date&state=open>`_.
    For our long-term plan, see our :ref:`roadmap`.

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

1. developer A resolves, at current state v0.9.3, to remove functionality X at some point in the near future.
2. therefore, by the above, we should introduce a deprecation message, visible from next release (e.g., v0.9.4),
  which says that functionality will be removed at v0.11.0
3. developer A makes a pull request to remove functionality X which includes that deprecation warning.
  The pull request is reviewed by core developers, with the suggestion by developer A accepted or rejected.
4. If accepted and merged before v0.10.0 release, the PR goes in the next release, with a deprecation note in the release notes.
  If PR acceptance takes until after v0.10.0 but before v0.11.0, the planned removal moves to v0.12.0 and the warning needs to be updated.
5. an additional PR to remove deprecation warning and functionality X is prepared by developer A, for v0.12.0 but not merged
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

This section outlines the deprecation process for some advaned cases.

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
