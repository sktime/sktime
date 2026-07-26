.. _release:

Releases
========

This section is for core developers releasing a new version of ``sktime``.

Core developers making a release should be release managers (appointed by CC) and have write access to the repository.


Summary of release process
--------------------------

The release process includes, in sequence:

* release cycle management
* release version preparation
* the release on ``pypi``
* the release on ``conda``
* troubeleshooting and accident remedial, if applicable

Details follow below.

Release cycle process
^^^^^^^^^^^^^^^^^^^^^

``sktime`` aims for a release every two weeks.

The release cycle process is as follows:

1. 1 week before release date, update the release board.
2. for major releases or substantial features, optionally extend the release cycle
3. feature freeze 1 day before release date. Only release managers should merge at this point.
  The feature freeze should be announced on the core dev channel 1 week before, and 1 day before it comes into action.
  Any delays and extensions to the feature freeze should also be announced.
4. if "must have" are not merged by planned release date: either delay release date and extend freeze period, or deprioritize.

Preparing the release version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The release process is as follows, on high-level:

1. ensure deprecation actions are carried out.
  Deprecation actions for a version should be marked by "version number" annotated comments in the code.
  E.g., for the release 0.12.0, search for the string 0.12.0 in the code and carry out described deprecation actions.
  It is recommended to collect the list of deprecation actions in or multiple separate PR,
  as they will have to go in the release notes.
  Deprecation actions should be merged only by release managers.

2. create a "release" pull request (ideally from a branch following the naming pattern ``release/v0.x.y``). This should make changes to the version numbers and have complete release notes.
  See below for version numbers and release notes.

3. The PR and release notes should optimally be reviewed by the other core developers, then merged once tests pass.

4. create a GitHub draft release with a new tag following the syntax v[MAJOR].[MINOR].[PATCH],
   e.g., the string ``v0.12.0`` for version 0.12.0.
   The GitHub release notes should contain only "new contributors" and "all contributors" section,
   and otherwise link to the release notes in the changelog,
   following the pattern of current GitHub release notes.

``pypi`` release and release validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

5. publish the GitHub draft release. Creation of the new tag will trigger the ``pypi`` release CI/CD.

6. wait for the ``pypi`` release CI/CD to finish. If tests fail due to sporadic unrelated failure, restart.
  If tests fail genuinely, something went wrong in the above steps, investigate, fix, and repeat.
  Common possibilities are core devs not respecting the feature freeze period,
  new releases of dependencies that happen in the ca one day period between release PR and release.

7. once release CI/CD has passed, check ``sktime`` version on ``pypi``, this should be the new version.
  It should be checked that all wheels have been uploaded, `on pypi, here <https://pypi.org/simple/sktime/>`__.
  A validatory install of ``sktime`` in a new ``python`` environment should be carried out (one arbitrary version/OS),
  according to the install instructions.
  If the install does not succeed or wheels have not been uploaded, urgent action to diagnose and remedy must be taken.
  All core developers should be urgently informed of such a situation through mail-all in the core developer channel.
  In the most common case, the install instructions need to be updated.
  If wheel upload has failed, the tag in 5. needs to be deleted and recreated.
  The tag can be deleted using the ``git`` command ``git push --delete origin tagname`` from a local repo.

``conda`` release and release validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

8. If the release on ``pypi`` has succeeded, there should be an automated release PR created
against the sktime conda-forge repo: https://github.com/conda-forge/sktime-feedstock.

  .. note:: Manual creation of release pull request:

     In cases where the `conda-forge` release PR is not created automatically it can be created and submitted manually. For general
     guidelines related to maintaining conda feedstcok packages see `conda-forge package<https://conda-forge.org/docs/maintainer/updating_pkgs.html>`_.

     After forking and cloning the repo, edit the ``meta.yml`` file and

     - increment the version in the line that contains ``{% set version = "0.X.Y" %}``
     - paste the sha256 sum of the source archive from github in the ``source/sha256`` section
     - submit PR and ask for review

9. The conda release PR need to be reviewed and in dependencies should be checked against any changes in the main sktime repo.
  In case the dependencies (or python version support) have changes, the ``meta.yml`` file in the conda recipe need to updated to reflect those changes.

10. Once reviewed, the conda release PR should merged, and it will automatically trigger a release of the conda package.

11. After 1h, it should be checked whether the package has been released on conda.
  Once the package is available on ``conda``, a test install should be carried out to validate the release.
  Should either of these fail, the same urgent action as in 7. should be taken.


Version number locations
------------------------

Version numbers need to be updated in:

* root ``__init__.py``
* ``README.md``
* ``pyproject.toml``


Release notes
-------------

Release notes can be generated using the ``build_tools.changelog.py`` script, and should be placed at the top of the ``changelog.rst``.
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
`build_tools/make_release.py <https://github.com/sktime/sktime/blob/main/build_tools/make_release.py>`__
that will guide you through the release process.

IMPORTANT: when running the legacy build tools, ensure to not accidentally push a release tag to the ``sktime`` repo.
