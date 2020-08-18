Governance
==========

Overview
--------

sktime is a meritocratic, consensus-based community project. Anyone with
an interest in the project can join the community, contribute to the
project, and participate in the governance process. This document
describes how that participation takes place and how to set about
earning merit within our community.

We are particularly motivated to support new and/or anxious
collaborators, people who are looking to learn and develop their skills,
and anyone who has experienced discrimination in the past.

.. list-table::
   :header-rows: 1

   * - Section
     - Purpose
   * - :ref:`code-of-conduct`
     - How we expect all members of the sktime community to interact
   * - :ref:`roles`
     - What roles we have in sktime's community and what rights and responsibilities they have
   * - :ref:`decision-making`
     - How and by whom decisions are made
   * - :ref:`acknowledging-contributions`
     - How we acknowledge contributions
   * - :ref:`outlook`
     - What we may change in the future

.. _code-of-conduct:

Code of Conduct
---------------

We value the participation of every member of our community and want to
ensure an that every contributor has an enjoyable and fulfilling
experience. Accordingly, everyone who participates in the sktime project
is expected to show respect and courtesy to other community members at
all times.

We ask all members of the community to conform to our `Code of
Conduct <https://github.com/alan-turing-institute/sktime/blob/master/CODE_OF_CONDUCT.rst>`__.

.. _roles:

Roles
-----

We distinguish between the following key roles that community members
may exercise. For each role, we describe their rights and
responsibilities, and appointment process in more detail below.

.. list-table::
   :header-rows: 1

   * - Role
     - Rights/responsibilities
     - Appointment
   * - :ref:`contributors`
     - \-
     - Concrete contribution
   * - :ref:`algorithm-maintainers`
     - Algorithm maintenance, voting and veto right for changes to their algorithm
     - Algorithm contribution or appointment by current maintainer
   * - :ref:`core-developers`
     - Direct write access, issue/PR management, veto right, voting, nomination
     - Nomination by core developers, vote by core developers, 2/3 majority
   * - :ref:`coc-committee-members`
     - CoC maintenance, investigation and enforcement
     - Nomination by core developers, vote by core developers, 2/3 majority and simple CoC majority
   * - :ref:`cc-members`
     - Conflict resolution, technical leadership, project management
     - Nomination by core developers, vote by core developers, 2/3 majority and simple CC majority

.. _contributors:

Contributors
~~~~~~~~~~~~

Contributors are community members who have contributed in concrete ways
to the project. Anyone can become a contributor, and contributions can
take many forms – not only code – as detailed in the `contributing
guide <https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTING.rst>`__.

For more details on how we acknowledge contributions, see :ref:`acknowledging-contributions` below.

All contributors are listed in `CONTRIBUTORS.md <https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTORS.md>`__.

.. _algorithm-maintainers:

Algorithm maintainers
~~~~~~~~~~~~~~~~~~~~~

Algorithm maintainers are contributors who have contributed an
algorithm. They have the same voting rights as core developers with
regard to their algorithm.

In sktime, algorithms are encapsulated in classes with specific
interface requirements and are called estimators. To faciliate
maintainership questions, we try to write algorithms in separate files
when possible.

Rights and responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
  :header-rows: 1

  * - Right/responsibility
    - Description
  * - Decision making with respect to their algorithm
    - Algorithm maintainers can partipate in the decision making process by vetoing changes and casting votes with regard to proposed changes to their algorithm. This does not extend to proposed changes to the common framework and API.
  * - Maintenance
    - They are responsible for maintaining the code and documentation for their algorithm, including bug fixes, unit testing, coding style, compliance with the common API, docstrings, documentation and tutorials notebooks.
  * - Support
    - They are the first point of contact for users and other contributors for all questions, issues and proposals regarding their algorithm.

Appointment
^^^^^^^^^^^

The contributor who contributes an algorithm is automatically appointed
as its first maintainer. If they can no longer fulfil their maintenance
responsibilities, maintainers are expected to resign.

When the maintainer resigns, they can appoint another contributor as the
new maintainer. No vote is required.

Maintainers are listed in the `CODEOWNERS <https://github
.com/alan-turing-institute/sktime/blob/master/CODEOWNERS>`__ file.

.. _core-developers:

Core developers
~~~~~~~~~~~~~~~

Core developers are contributors who have shown that they are dedicated
to the continued development of the project through ongoing engagement
with the community.

Current core developers are listed in the `core-developers
team <https://github.com/orgs/sktime/teams/core-developers/members>`__
within the sktime organisation on GitHub.

.. _rights-and-responsibilities-1:

Rights and responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Right/responsibility
     - Description
   * - Direct access
     - Being a core developer allows contributors to more easily carry on with their project related activities by giving them direct access to the project’s repository.
   * - Issue/PR management
     - Core developers are responsible for reviewing and managing issues and pull requests. This includes commenting on issues, reviewing code contributions, merging approved pull requests, and closing issues once resolved.
   * - Decision making
     - They can partipate in the decision making process by vetoing changes and casting votes.
   * - Nomination
     - They can nominate new core developers, CoC committee members and CC members.


.. _appointment-1:

Appointment
^^^^^^^^^^^

New core developers can be nominated by any current core developer. Once
they have been nominated, there will be a vote by the current core
developers.

Voting on new core developers is one of the few activities that takes
place on the project’s private chat. While it is expected that most
votes will be unanimous, a 2/3 majority of the cast votes is enough. The
vote needs to be open for at least five working days.

Core developers that have not contributed to the project (commits or
GitHub comments) in the past 12 months will be asked if they want to
become *emeritus core developers* and give up their rights and
responsibilities. When they become active again, emeritus developers can
retake their role without having to be appointed.

.. _coc-committee-members:

CoC committee members
~~~~~~~~~~~~~~~~~~~~~

The current members of the CoC committee are listed in the
`CoC <https://github.com/alan-turing-institute/sktime/blob/master/CODE_OF_CONDUCT.rst>`__.

.. _rights-and-responsibilities-2:

Rights and responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

CoC committee members are responsible for investigating potential CoC
incidents and enforcing the CoC.
They are the point of contact for reporting potential CoC incidents.

In addition, they are responsible for maintaining and improving the CoC.

.. _appointment-2:

Appointment
^^^^^^^^^^^

Membership of the CC is by nomination by a core developer and a vote by
all core developers. A nomination will result in discussion which cannot
take more than a week and then a vote by the core developers which will
stay open for a week. CoC membership votes are subject to:

* a 2/3 majority of all cast votes, and
* a simple majority approval of all the current CoC members.

To avoid deadlocks if there is an even number of CoC members, one of
them will have a tie breaking privilege.

.. _cc-members:

CC members
~~~~~~~~~~

CC members are core developers with additional rights and
responsibilities to avoid deadlocks and ensure a smooth progress of the
project.

Current CC members are listed in the `community-council
team <https://github.com/orgs/sktime/teams/community-council/members>`__
within the sktime organisation on GitHub.

.. _rights-and-responsibilities-3:

Rights and responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Right/responsibility
     - Description
   * - Decision making: conflict resolution
     - see :ref:`stage-3` below
   * - Technical direction
     - Strategic planning, development roadmap
   * - Project management
     - Funding, collaborations with external organisations, community infrastructure (chat server, GitHub repositories, continuous integration accounts, social media accounts)


.. _appointment-3:

Appointment
^^^^^^^^^^^

Membership of the CC is by nomination by a core developer and a vote by
all core developers. A nomination will result in discussion which cannot
take more than a week and then a vote by the core developers which will
stay open for a week. CC membership votes are subject to:

* a 2/3 majority of all cast votes, and
* a simple majority approval of all the current CC members.

To avoid deadlocks if there is an even number of CC members, one of them
will have a tie breaking privilege.

CC members who do not actively engage with the CC responsibilities are
expected to resign.

Communications
^^^^^^^^^^^^^^

The CC has regular public meetings that the full community is welcome to
attend.

-  The agenda, available at
   `community-council/AGENDA.md <https://github.com/sktime/community-council/blob/master/AGENDA.md>`__,
   will be discussed in each meeting.
-  If you want to add an agenda item to the meeting, simply `update the
   agenda <https://github.com/sktime/community-council/edit/master/AGENDA.md>`__
   and make sure you attend the meeting.
-  All meetings are logged and available at
   `community-council/previous_meetings/ <https://github.com/sktime/community-council/tree/master/previous_meetings>`__.
-  To contact the CC directly, please send an email to info@sktime.org.

.. _decision-making:

Decision making
---------------

The purpose of this section is to formalize the decision-making process
used by the sktime project. We clarify \* what types of changes we make
decision on, \* how decisions are made, and \* who participates in the
decision making.

sktime’s decision-making process is designed to take into account
feedback from all community members and strives to find consensus, while
avoiding deadlocks when no consensus can be found.

All discussion and votes takes place on the project’s `issue
tracker <https://github.com/alan-turing-institute/sktime/issues>`__,
`pull requests <https://github.com/alan-turing-institute/sktime/pulls>`__ or an :ref:`steps`. Some
sensitive discussions and appointment votes occur on private chats.

The CC reserves the right to overrule decisions.

We distinguish between the following types of proposed changes. The
corresponding decision making process is described in more detail below.

.. list-table::
   :header-rows: 1

   * - Type of change
     - Decision making process
   * - Code additions, such as new algorithms
     - Lazy consensus, supported by the :ref:`algorithm-inclusion-guidelines`
   * - Minor documentation changes, such as typo fixes, or addition/correction of a sentence
     - Lazy consensus
   * - Code changes and major documentation changes
     - Lazy consensus
   * - Changes to the API design, hard dependencies, or supported versions
     - Lazy consensus based on an :ref:`steps`
   * - Changes to sktime's governance (this document and the CoC)
     - Lazy consensus based on an :ref:`steps`
   * - Appointment
     - Voting

.. _stage-1:

Stage 1: lazy consensus with veto right
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sktime uses a “consensus seeking” process for making decisions. The
community tries to find a resolution that has no open objections among
core developers.

-  To accept proposed changes, we require approval by one core developer
   (lazy consensus) and no rejection by a core developer (veto right).
-  Approvals and rejections can be expressed as +1 and -1 comments,
   respectively.
-  Core developers are expected to give reasonable time to others to
   give their opinion on the pull request if they’re not confident
   others would agree.
-  More important changes that impact the full project require a more
   detailed analysis and a consensus that is both explicit and informed.
   These changes require an :ref:`steps`.

When no consensus can be found, the decision is escaled to :ref:`stage-2`.

.. _stage-2:

Stage 2: voting
~~~~~~~~~~~~~~~

When no consensus can be found, any core developer can call for a vote
at any point during the discussion.

-  The vote will conclude 10 working days from the call for the vote.
-  Votes are public and voluntary. Abstentions are allowed. You can
   abstain by simply not casting a vote.
-  All votes are a binary vote: for or against accepting the proposed
   changes.
-  Votes are casts as comments: +1 (approval) or -1 (rejection).

For all types of changes, except appointments, the winning condition is
a 2/3 majority of the votes casts by core developers including CC
members. If the proposed change cannot gather a 2/3 majority of the
votes cast by core developers, the decision is escalated to the :ref:`stage-3`.

For appointments, winning conditions vary depending on the role as
described in :ref:`roles` above. Appointment
decisions are not escalated to the CC. If a nomination cannot gather
sufficient support, the nomination is rejected.

.. _stage-3:

Stage 3: conflict resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the proposed change cannot gather a 2/3 majority of the votes cast,
the CC tries to resolve the deadlock.

-  The CC will use consensus seeking.
-  If no consensus can be found within a month, the decision is made
   through a simple majority vote (with tie breaking) among the CC
   members.
-  Any CC decision must be supported by an :ref:`steps`, which has been made public and discussed before the vote.

.. _steps:

sktime enhancement proposal
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For all decision of the CC, an sktime enhancement proposal (STEP) must have been made
public and discussed before the vote.

An sktime enhancement proposal is a consolidated document, with a concise
problem statement, a clear description of the proposed solution and a
comparison with alternative solutions, as outlined in our
`template <https://github.com/sktime/enhancement-proposals/blob/master/TEMPLATE.md>`__.

We collect and discuss proposals in sktime’s `repository for
enhancement-proposals <https://github.com/sktime/enhancement-proposals>`__.

.. _algorithm-inclusion-guidelines:

Algorithm inclusion guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Curation is about how we select contributions, which criteria we use in
order to decide which contributions to include, and in which cases we
deprecate and remove contributions.

We have the following guidelines:

-  We only consider published algorithms which have been shown to be
   competitive in comparative benchmarking studies or practically useful
   in applied projects. A technique that provides a clear-cut
   improvement (e.g. an enhanced data structure or a more efficient
   approximation technique) on a widely-used method will also be
   considered for inclusion.
-  From the algorithms or techniques that meet the above criteria, only
   those which fit well within the current API of sktime are accepted.
   For algorithms that do not fit well into the current API, the API
   will have to be extended first. For extending current API, see the
   `decision making process <#Decision-making>`__ for major changes.
-  The contributor should support the importance of the proposed
   addition with research papers and/or implementations in other similar
   packages, demonstrate its usefulness via common
   use-cases/applications and corroborate performance improvements, if
   any, with benchmarks and/or plots. It is expected that the proposed
   algorithm should outperform the methods that are already implemented
   in sktime in at least some areas.
-  We strive to consolidate existing functionality if helps to improve
   the usability and maintainability of the project. For example, when
   there are multiple techniques for the same purpose, we prefer to
   combine them into a single class and make case distinctions based on
   hyper-parameters.

Note that your implementation need not be in sktime to be used together
with sktime tools. You can implement your favorite algorithm in a sktime
compatible way in one of `our companion
repositories <https://github.com/sktime>`__ on GitHub. We will be happy
to list it under `related
software <https://github.com/alan-turing-institute/sktime/wiki/related-software>`__.

If algorithms require major dependencies, we encourage to create a
separate companion repository. For example, for deep learning techniques
based on TensorFlow and Keras, we have
`sktime-dl <https://github.com/sktime/sktime-dl>`__. For smaller
dependencies which are limited to a few files, we encourage to use soft
dependencies, which are only required for particular modules, but not
for most of sktime’s functionality and not for installing sktime.

.. _acknowledging-contributions:

Acknowledging contributions
---------------------------

sktime is collaboratively developed by its diverse community of
developers, users, educators, and other stakeholders. We value all kinds
of contributions and are committed to recognising each of them fairly.

We follow the `all-contributors <https://allcontributors.org>`__
specification to recognise all contributors, including those that don’t
contribute code. Please see `our list of all
contributors <https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTORS.md>`__.

If you think, we’ve missed anything, please let us know or open a PR
with the appropriate changes to
`sktime/.all-contributorsrc <https://github.com/alan-turing-institute/sktime/blob/master/.all-contributorsrc>`__.

Note that contributors do not own their contributions. sktime is an
open-source project, and all code is contributed under `our open-source
license <https://github.com/alan-turing-institute/sktime/blob/master/LICENSE>`__.
All contributors ackwoledge that they have all the rights to the code
they contribute to make it available under this license.

The project belongs to the sktime community, and all parts of it are
always considered “work in progress” so that they can evolve over time
with newer contributions.

.. _outlook:

Outlook
-------

We are open to improvement suggestions for our governance model. Once
the community grows more and sktime’s code base becomes more
consolidated, we will consider the following changes:

-  Allow for more time to discuss changes, and more time to cast vote
   when no consensus can be found,
-  Require more positive votes (less lazy consensus) to accept changes
   during consensus seeking stage,
-  Reduce time for maintainers to reply to issues

In addition, we plan to add more roles for managing/coordinating
specific project:

* Community manager (mentorship, outreach, social media, etc),
* Sub-councils for project-specific technical leadership (e.g.  for documentation, learning tasks, continuous integration)

.. _references:

References
----------

Our governance model is inspired by various existing governance
structures. In particular, we’d like to acknowledge:

* scikit-learn’s `governance model <https://sktime.org/stable/governance.html>`__
* `The Turing Way <https://github.com/alan-turing-institute/the-turing-way>`__ project
* `The Art of Community <https://www.jonobacon.com/books/artofcommunity/>`__ by Jono Bacon
* The `astropy <https://www.astropy.org>`__ project
* The `nipy <https://nipy.org>`__ project
* The `scikit-hep <https://scikit-hep.org>`__ project
