.. _release:

Releases
========

This section is for core developers releasing a new version of ``sktime``.

Core developers making a release should be release managers (appointed by CC) and have write access to the repository.


Summary of release process
--------------------------

``sktime`` aims for a release every two weeks.

The release cycle process is as follows:

1. 1 week before release date, update the release board.
2. for major releases or substantial features, optionally extend the release cycle
3. feature freeze 1 day before release date. Only release managers should merge at this point.
  The feature freeze should be announced on the core dev channel (on slack) 1 week before, and 1 day before it comes into action.
  Any delays and extensions to the feature freeze should also be announced.
4. if "must have" are not merged by planned release date: either delay release date and extend freeze period, or deprioritize.


The release process is as follows, on high-level:

1. ensure deprecation actions are carried out.
  Deprecation actions for a version should be marked by "version number" annotated comments in the code.
  E.g., for the release 0.12.0, search for the string 0.12.0 in the code and carry out described deprecation actions.
  Collect list of deprecation actions in an issue, as they will have to go in the release notes.
  Deprecation actions should be merged only by release managers.

2. create a "release" pull request. This should make changes to the version numbers and have complete release notes.
  See below for version numbers and release notes.

3. The PR and release notes should optimally be reviewed by the other core developers, then merged once testst pass.

4. create a GitHub draft release with a new tag following the syntax v[MAJOR].[MINOR].[PATCH],
   e.g., the string ``v0.12.0`` for version 0.12.0.
   The GitHub release notes should contain only "new contributors" and "all contributors" section,
   and otherwise link to the release notes in the changelog,
   following the pattern of current GitHub release notes.

5. publish the GitHub draft release. Creation of the new tag will trigger the release CI/CD.

6. wait for the release CI/CD to finish. If tests fail due to sporadic unrelated failure, restart.
  If tests fail genuinely, something went wrong in the above steps, investigate, fix, and repeat.
  Common possibilities are core devs not respecting the feature freeze period,
  new releases of dependencies that happen in the ca one day period between release PR and release.

7. once release CI/CD has passed, check ``sktime`` version on ``pypi``, this should be the new version.
  It should be checked that all wheels have been uploaded, `here <https://pypi.org/simple/sktime/>`__.
  A validatory install of ``sktime`` in a new ``python`` environment should be carried out (one arbitrary version/OS),
  according to the install instructions.
  If the install does not succeed or wheels have not been uploaded, urgent action to diagnose and remedy must be taken.
  All core developers should be urgently informed of such a situation through mail-all in the core developer channel on slack.
  In the most common case, the install instructions need to be updated.

8. If the release on ``pypi`` has succeeded, the ``conda`` release should automatically trigger via
  https://github.com/conda-forge/sktime-feedstock
  If this has not happened, action to diagnose and remedy should be taken.
  Once the package is available on ``conda``, a test install should be carried out to validate the release.


Version number locations
------------------------

Version numbers need to be updated in:

* root ``__init__.py``
* ``README.md``
* ``pyproject.toml``


Release notes
-------------

Release notes can be generated using the ``build_tools.release`` script, and should be placed at the top of the ``changelog.rst``.
Generally, release notes should follow the general pattern of previous release notes, with sections:

* highlights
* dependency changes, if any
* deprecations and removals, if any.
  In PATCH versions, there are no deprecation actions, but there can be new deprecations.
  Deprecation action usually happen with the MINOR release cycle.
* core interface changes, if any. This means, changes to the base class interfaces.
  Only MINOR or MAJOR releases should have core interface changes that are not downwards compatible.
* enhancements, by module/area
* documentation
* maintenance
* bugfixes
* all contributor credits


Legacy build tools
------------------

We no longer use the legacy build tools using the make file.

To run the legacy release workflow, e.g., for development purposes, run

.. code:: bash

   make release

This calls
`build_tools/make_release.py <https://github.com/alan-turing-institute/sktime/blob/main/build_tools/make_release.py>`__
that will guide you through the release process.

IMPORTANT: when running the legacy build tools, ensure to not accidentally push a release tag to the ``sktime`` repo.
