.. _git_workflow:

Git and GitHub workflow
=======================

The preferred workflow for contributing to the ``sktime`` repository is to
fork the `main
repository <https://github.com/sktime/sktime/>`__ on
GitHub, clone, and develop on a new branch.

The workflow consists of two main parts:

* **First-time setup: Creating a fork and cloning the repository**: This section will help you set up your own forked copy of the ``sktime``
repository on GitHub and a local copy of the forked repository on your machine. This needs to be done only once, when you
start contributing to ``sktime``.

* **Every-time workflow: Developing a feature**: This is the process of developing a new feature, e.g., a bugfix or new estimator.
This is done every time you want to contribute a new feature.


.. note::

    GUI-based solutions to carry out the below workflow steps are also available.
    For example, to manage branches and commits, you can use:

    * `GitHub Desktop <https://desktop.github.com/>`_. This is the official GitHub GUI client and also integrates with your browser.
    * `Visual Studio Code <https://code.visualstudio.com/>`_, with suitable git extensions.
    * `pycharm <https://www.jetbrains.com/pycharm/>`_ (native installation).

    These solutions will carry out the same steps under the hood, but with a graphical interface.
    Even if you use a GUI, we recommended to understand the underlying commands, and try them out in the terminal at least once.


Creating a fork and cloning the repository - initial one time setup
-------------------------------------------------------------------

1.  Fork the `project
    repository <https://github.com/sktime/sktime>`__ by
    clicking on the 'Fork' button near the top right of the page. This
    creates a copy of the code under your GitHub user account. For more
    details on how to fork a repository see `this
    guide <https://help.github.com/articles/fork-a-repo/>`__.

2.  `Clone <https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository>`__
    your fork of the sktime repo from your GitHub account to your local
    disk:

    .. code:: bash

       git clone git@github.com:<username>/sktime.git
       cd sktime

    where :code:`<username>` is your GitHub username.

3.  Configure and link the remote for your fork to the upstream
    repository:

    .. code:: bash

       git remote -v
       git remote add upstream https://github.com/sktime/sktime.git

4.  Verify the new upstream repository you've specified for your fork:

    .. code:: bash

       git remote -v
       > origin    https://github.com/<username>/sktime.git (fetch)
       > origin    https://github.com/<username>/sktime.git (push)
       > upstream  https://github.com/sktime/sktime.git (fetch)
       > upstream  https://github.com/sktime/sktime.git (push)

.. note::

    Step 1 needs to be done once per GitHub account, and need to be repeated
    only if you are using a second GitHub account, or if you are intentionally
    resetting your fork.

    Steps 2-4 need to be done once per local machine, and need to be repeated
    only if you are working on a new machine, or after a reset of your local setup,
    e.g., after an operating system reinstall.


Developing a feature - repeat for every new feature
---------------------------------------------------

1.  `Sync <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_
    the ``main`` branch of your fork with the upstream repository:

    .. code:: bash

       git fetch upstream
       git checkout main
       git merge upstream/main

2.  Create a new feature branch, from the ``main`` branch, to hold
    your changes, with a descriptive name, replace ``<feature-branch>`` with that name:

    .. code:: bash

       git checkout main
       git checkout -b <feature-branch>

    Always use a ``feature`` branch. It's good practice to never work on
    the ``main`` branch! Name the ``feature`` branch after your
    contribution.

.. note::

    We recommend to never make changes in ``main`` branch of your fork, and always use a
    separate dedicated branch for a particular task.

3.  Develop your contribution on your feature branch. Add changed files
    using ``git add`` and then ``git commit`` files to record your
    changes in Git:

    .. code:: bash

       git add <modified_files>
       git commit

4.  When finished, push the changes to your GitHub account with:

    .. code:: bash

       git push --set-upstream origin my-feature-branch

5.  Follow `these
    instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`__
    to create a pull request from your fork. If your work is still work
    in progress, open a draft pull request.

.. note::

    We recommend to open a pull request early, so that other contributors become aware of
    your work and can give you feedback early on.

6.  To add more changes, simply repeat steps 3 - 4. Pull requests are
    updated automatically if you push new changes to the same branch.

.. _Discord: https://discord.com/invite/54ACzaFsn7

.. note::

   If any of the above seems like magic to you, look up the `Git documentation <https://git scm.com/documentation>`_.
   If you get stuck, chat with us on `Discord`_, or join one of the community sessions on `Discord`_.

7.  Between the time you created a pull request and when it is ready to merge into the
    ``main`` branch, the ``main`` branch of the sktime repo may have been updated with
    new changes by other contributors, and may cause merge conflicts. To keep your
    feature branch up-to-date with the ``main`` branch of the sktime repo, you can do
    the following:

    .. code:: bash

       git fetch upstream
       git checkout main
       git merge upstream/main
       git checkout <feature-branch>
       git merge main

    This will first update ``main`` branch of your fork with the latest changes from the
    ``main`` branch of the sktime repo, and then update your feature branch with those
    changes. If there are any merge conflicts, you will need to resolve them manually.

.. note::

    We strongly, emphatically, recommend to never use ``rebase`` for updating your
    feature branch when contributing to ``sktime``.
    ``rebase`` can lead to states that are very hard to recover from,
    because it rewrites history. **Always use ``merge`` to update your feature branch.**
    We squash all pull requests to a single commit on ``main``,
    so the history of your feature branch is not important.


Managing Branches - Advanced Guide
----------------------------------

This section provides some advanced tips on managing multiple branches.

Working on multiple features in parallel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are working on a different tasks in parallel without interdependency:
for each task, create a new feature branch from the ``main`` branch of your fork,
following the section "Contributing a feature - for every new feature", above.

We strongly recommend to not use the same branch for multiple tasks,
as it will make the history of the branch messy and harder to review,
and substantially increases the risk of bugs and conflicts.

Working on a chain of dependent tasks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more complex tasks, it may be useful to limit complexity by
chaining tasks after another.

For instance, working on an estimator that first requires a bugfix to be merged.

In this case, create a new branch from the branch of the previous task, and continue
your development from there. For such cases, please remember to specify in the PR
description that this PR depends on the previous PR.

Further, whenever making changes to the previous branch, ensure to update
the dependent branch with the latest changes from the previous branch.

The general workflow for ensuring that all branches in the chain are up-to-date,
is as follows. Assume we have branches A, B, C, etc, where A depends on ``main``,
B depends on A, C depends on B, etc.

After any change to any of the branches:

1. update your fork from the upstream repository
2. merge ``main`` into A, and resolve any conflicts
3. merge A into B, and resolve any conflicts
4. merge B into C, and resolve any conflicts
5. etc, until all branches in the chain have been merged and resolved


Cleaning up
~~~~~~~~~~~

Once your pull request is merged in the ``main`` branch of the sktime repo, you can
delete your feature branch:

.. code:: bash

    git checkout main
    git branch -D <feature-branch>

You can also delete the remote branch on your fork:

.. code:: bash

    git push origin --delete <feature-branch>
