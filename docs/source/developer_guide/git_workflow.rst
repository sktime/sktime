.. _git_workflow:

Git and GitHub workflow
=======================

The preferred workflow for contributing to sktime’s repository is to
fork the `main
repository <https://github.com/sktime/sktime/>`__ on
GitHub, clone, and develop on a new branch.

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

5.  `Sync <https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork>`_
    the ``main`` branch of your fork with the upstream repository:

    .. code:: bash

       git fetch upstream
       git checkout main
       git merge upstream/main

6.  Create a new ``feature`` branch from the ``main`` branch to hold
    your changes:

    .. code:: bash

       git checkout main
       git checkout -b <feature-branch>

    Always use a ``feature`` branch. It's good practice to never work on
    the ``main`` branch! Name the ``feature`` branch after your
    contribution.

7.  Develop your contribution on your feature branch. Add changed files
    using ``git add`` and then ``git commit`` files to record your
    changes in Git:

    .. code:: bash

       git add <modified_files>
       git commit

8.  When finished, push the changes to your GitHub account with:

    .. code:: bash

       git push --set-upstream origin my-feature-branch

9.  Follow `these
    instructions <https://help.github.com/articles/creating-a-pull-request-from-a-fork>`__
    to create a pull request from your fork. If your work is still work
    in progress, open a draft pull request.

.. note::

    We recommend to open a pull request early, so that other contributors become aware of
    your work and can give you feedback early on.

10. To add more changes, simply repeat steps 7 - 8. Pull requests are
    updated automatically if you push new changes to the same branch.

.. _Discord: https://discord.com/invite/gqSab2K

.. note::

   If any of the above seems like magic to you, look up the `Git documentation <https://git scm.com/documentation>`_.
   If you get stuck, chat with us on `Discord`_, or join one of the community sessions on `Discord`_.
