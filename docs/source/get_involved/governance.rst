.. _governance:

Governance
==========

Overview
--------

sktime is a consensus-based community project. Anyone with an interest in the project can join the community, contribute to the project, and participate in the governance process. This document describes how that participation takes place, which roles we have in our community, how we make decisions, and how we acknowledge contributions.

We are particularly motivated to support new and/or anxious contributors, people who are looking to learn and develop their skills, and members of groups underrepresented in the tech sector. Go to our `contributing guide <https://github.com/sktime/sktime/blob/main/CONTRIBUTING.rst>`__ for more details.

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

We ask all members of the community to conform to our :ref:`code_of_conduct`.

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
   * - :ref:`cc-observers`
     - Full view of CC communication, direct input on CC decisions
     - Nomination by core developers, vote by CC members, simple CC majority

.. _contributors:

Contributors
~~~~~~~~~~~~

Contributors are community members who have contributed in concrete ways
to the project. Anyone can become a contributor, and contributions can
take many forms – not only code – as detailed in the `contributing
guide <https://github.com/sktime/sktime/blob/main/CONTRIBUTING.rst>`__.

For more details on how we acknowledge contributions, see :ref:`acknowledging-contributions` below.

All contributors are listed in `CONTRIBUTORS.md <https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md>`__.

.. _algorithm-maintainers:

Algorithm maintainers
~~~~~~~~~~~~~~~~~~~~~

Algorithm maintainers are contributors who have contributed an
algorithm. They have the same voting rights as core developers with
regard to their algorithm.

In sktime, algorithms are encapsulated in classes with specific
interface requirements and are called estimators. To facilitate
maintainership questions, we try to write algorithms in separate files
when possible.

To clarify, "algorithm" in the above sense means "implemented estimator class".
That is, algorithm maintainers gain rights and responsibilities with respect to
that python code.
They do not gain any rights on abstract methodology, e.g., in a case where
the class implements methodology invented by third parties.

In particular, algorithm maintainers do not gain rights or responsibilities on other,
potential implementations of the same methodology in their estimator class.

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

Recall, "algorithm" refers to estimator classes.

Therefore, the above rights and responsibilities exclude any power on further, potential implementations of the same or similar methodology.

For instance, an algorithm maintainer of algorithm A implemented in class X cannot prohibit implementation of algorithm A in class Y.
They can only make decisions about changes on class X. Class Y can be owned by a different algorithm maintainer.

In particular, there can be multiple classes implementing algorithm A, and the algorithm maintainer of class X cannot prohibit implementation of, or make decisions on class Y.

Expectations
^^^^^^^^^^^^

Without restriction to eligibility, it is generally expected that algorithm maintainers
have a very good technical and methodological understanding of the algorithm they maintain.

This understanding is typically present in inventors or proponents of said algorithm,
but it is not necessary to be the inventor of an algorithm to be its maintainer.

Eligibility
^^^^^^^^^^^

Anyone is eligible to be an algorithm maintainer.

Anyone is eligible to be an algorithm maintainer of a specific algorithm that does not already have an algorithm maintainer.

The presence of a specific implementation of a given abstract algorithm does not prevent anyone from becoming
the algorithm maintainer of a different implementation of the same (or similar) abstract algorithm.

Appointment
^^^^^^^^^^^

The contributor who contributes an algorithm is automatically appointed
as its first maintainer.

Algorithm maintainers are listed in the ``"maintainers"`` tag of the estimator class,
by their GitHub ID. The GitHub ID can be linked to further information via
the ``all-contributorsrc`` file.
The tag can be inspected directly in the source code of the class,
or via ``EstimatorName.get_class_tag("maintainers").``
Inverse lookup such as "which algorithms does maintainer M maintain"
can be carried out using ``registry.all_estimators``.

When an algorithm maintainer resigns, they can appoint another contributor as the
new algorithm maintainer. No vote is required.
This change should be reflected in the ``"maintainers"`` tag.

Algorithm maintainers can be appointed by CC simple majority for any algorithm without maintainers.

End of tenure
^^^^^^^^^^^^^

If algorithm maintainers can no longer fulfil their maintenance
responsibilities, maintainers are expected to resign.

Algorithm maintainers that have been unresponsive for a 3 month period automatically
give up their rights and responsibilities as algorithm maintainers.

Unresponsiveness is defined as:

* not engaging with decision making procedures within the reasonably time frames defined there
* not reacting to issues or bug reports related to the algorithm, within ten working days

.. _core-developers:

Core developers
~~~~~~~~~~~~~~~

Core developers are contributors who have shown that they are dedicated
to the continued development of the project through ongoing engagement
with the community.

Current core developers are listed in the `core-developers
team <https://www.sktime.net/en/latest/about/team.html>`__
within the sktime organisation on GitHub.

.. _rights-and-responsibilities-1:

Rights and responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Right/responsibility
     - Description
   * - Direct access
     - Being a core developer allows contributors to more easily carry on with their project related activities by giving them direct access to the project's repository.
   * - Issue/PR management
     - Core developers are responsible for reviewing and managing issues and pull requests. This includes commenting on issues, reviewing code contributions, merging approved pull requests, and closing issues once resolved.
   * - Decision making
     - They can participate in the decision making process by vetoing changes and casting votes.
   * - Nomination
     - They can nominate new core developers, CoC committee members and CC members.

Eligibility
^^^^^^^^^^^

Anyone is eligible to be a core developer.

.. _appointment-1:

Appointment
^^^^^^^^^^^

New core developers can be nominated by any current core developer. Once
they have been nominated, there will be a vote by the current core
developers.

Voting on appointments is one of the few activities that takes
place on the project's private communication channels. The vote will be
anonymous.

While it is expected that most votes will be unanimous, a 2/3 majority of
the cast votes is enough. The vote needs to be open for five days excluding
weekends.

End of tenure
^^^^^^^^^^^^^

Core developers can resign voluntarily at any point in time, by informing the CC in writing.

Core developers that have not contributed to the project in the past
one-year-period will automatically become *inactive*
and give up their rights and responsibilities. When they become active
again, they can retake their role without having to be appointed.

Becoming inactive in the above sense means not contributing for the period via:

* creating pull requests
* commenting on pull requests or issues
* attending one of the regular meetings

Becoming active (after becoming inactive) in the above sense requires one of:

* an approved pull request authored by the core developer
* a contribution to the community that is minuted in one of the regular meetings

.. _coc-committee-members:

CoC committee members
~~~~~~~~~~~~~~~~~~~~~

CoC members are contributors with special rights and responsibilities.
The current members of the CoC committee are listed in the
`CoC <https://www.sktime.net/en/latest/about/team.html>`__.

.. _rights-and-responsibilities-2:

Rights and responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

CoC committee members are responsible for investigating potential CoC
incidents and enforcing the CoC.
They are the point of contact for reporting potential CoC incidents.

In addition, they are responsible for maintaining and improving the CoC.

Eligibility
^^^^^^^^^^^

Anyone is eligible to be a CoC committee member.

.. _appointment-2:

Appointment
^^^^^^^^^^^

Membership of the CoC is by nomination by a core developer and a vote by
all core developers. A nomination will result in discussion which will stay
open for 5 days excluding weekends and then a vote by the core
developers which will stay open for 5 days excluding weekends. CoC committee
membership votes are subject to:

* a 2/3 majority of all cast votes, and
* a simple majority approval of all the current CoC committee members.

The vote will take place in private communication channels and will be
anonymous.

To avoid deadlocks if there is an even number of CoC committee members, one
of them will have a tie breaking privilege.

.. _cc-members:

CC members
~~~~~~~~~~

CC (community council) members are core developers with additional rights and
responsibilities to avoid deadlocks and ensure a smooth progress of the
project.

Current CC members are listed in the `community-council
team <https://www.sktime.net/en/latest/about/team.html>`__
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

Eligibility and appointment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The community council is elected by contributors to sktime in regular elections.

The elections process is detailed in the `community council elections repository <https://github.com/sktime/elections/>`__.

The repository linked contains information about the election process, including the election schedule, the election rules, and past election results.

Communications
^^^^^^^^^^^^^^

The CC has regular public meetings that the full community is welcome to attend.
Meetings take place on the social channels of the project, currently on Discord.

For more details about our meetings and minutes of previous meetings,
please go to our `community-council repository <https://github.com/sktime/community-org/tree/main/community_council/previous_meetings/>`__.

To contact the CC directly, please send an email to sktime.toolbox@gmail.com.

.. _cc-observers:

CC observers
~~~~~~~~~~~~

CC (community council) observers are core developers with additional rights and
responsibilities. Current CC observers are listed in the `community-council
observers <https://www.sktime.net/en/latest/about/team.html`__.

.. _rights-and-responsibilities-4:

Rights and responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

CC observers have a full view of reserved CC proceedings, the private CC
channels and the sktime email account. CC observers can participate in
discussions on private CC channels to ensure that more members of the community
have direct input on CC decisions.

CC observers' responsibilities include to critically scrutinize CC decision
making and give their input on what is of community's interest or benefit.

CC observers do not possess the voting or decision making rights of full
CC members.

Eligibility
^^^^^^^^^^^

Only core developers are eligible for appointment as CC observers.
Non-core-developers can be nominated, but this must be accompanied
by a nomination for core developer, and a core developer appointment vote
(see below).

.. _appointment-4:

Appointment
^^^^^^^^^^^

Membership of the CC observers is by nomination by a core developer and a vote
by CC members. A nomination will result in a vote by the CC members which will
stay open for 5 days excluding weekends. CC observer membership votes are
subject to a simple majority approval of all the current CC committee members.

In case of ties, the CC member with shortest tenure breaks the tie.

.. _treasurer:

Special operational role: treasurer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The treasurer is an appointed role on the ``sktime`` project.
This is primarily a supportive and transparency enhancing role.

If the treasurer role remains unfilled for longer than a month,
the CC must exercise the responsibilities of the treasurer role, as a committee.

Rights and responsibilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The treasurer will work closely with the CC to set financial goals, allocate resources, and ensure ethical fiscal management.

Responsibilities include budgeting, fiscal management, financial reporting, internal policy compliance, and cash management.

The treasurer's primary responsibility is to produce the financial statements and budgets for the project in a timely manner.

Eligibility and appointment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The treasurer role is open to core developers of the sktime project.

Non-core developers must be confirmed as core developers before being considered for the treasurer role.

The CC appoints the treasurer through majority vote, among suitable candidates for the role.

The CC should solicit nominations from the community in transparent communication channels, when the role  needs to be filled.

Tenure and removal
^^^^^^^^^^^^^^^^^^

The treasurer serves a one-year term with the possibility of re-appointment.

Inactivity may result in removal if the treasurer fails to compile budgets or financial statements as required.

Removal for code of conduct violations related to fiscal transparency requires a CoC investigation.

.. _decision-making:

Decision making
---------------

The purpose of this section is to formalize the decision-making process
used by the sktime project. We clarify:

* what types of changes we make decision on,
* how decisions are made, and
* who participates in the decision making.

sktime's decision-making process is designed to take into account
feedback from all community members and strives to find consensus, while
avoiding deadlocks when no consensus can be found.

All discussion and votes takes place on the project's `issue
tracker <https://github.com/sktime/sktime/issues>`__,
`pull requests <https://github.com/sktime/sktime/pulls>`__ or an :ref:`steps`. Some
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
     - Lazy consensus, requires a :ref:`steps`
   * - Changes to sktime's governance (this document and the CoC)
     - Lazy consensus, requires a :ref:`steps`
   * - Appointment
     - Directly starts with voting (stage 2)

.. _stage-1:

Stage 1: lazy consensus with veto right
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

sktime uses a "consensus seeking" process for making decisions. The
community tries to find a resolution that has no open objections among
core developers.

-  Proposed changes should be in the form of GitHub pull requests (PR).
   Some changes also require a worked out :ref:`steps`. This depends on the type of change, see
   `decision making process <#Decision-making>`__ above.
-  For a proposed change to be approved via lazy consensus, it needs to
   approval by at least one core developer (lazy consensus) and no rejection by a core developer (veto right).
   The approval required for this condition must be by a core developer different from a proposer.
-  For a proposed change to be rejected via lazy consensus, it needs to receive a
   rejection by at least one core developer, and no acceptance by a core developer.
-  Approvals must be in the form of a GitHub PR approval of the PR in question.
   Rejections can be expressed as -1 comments, or any written comments
   containing "I formally reject" in the PR, in reference to it.
-  Proposers are expected to give reasonable time for consideration, that is,
   time and opportunity for core developers to review and
   give their opinion on the PR.
   Ten working days excluding week-ends constitute "reasonable time" in the above sense.
   The period resets at every new change made to the PR.
   It starts only when all GitHub checks pass.
-  During this period, the PR can be merged if it has an approval and no rejection, but should be
   reverted if it receives a rejection in addition.
-  If the "reasonable time" period elapses and no approval or rejection has been expressed on a PR,
   the PR is scheduled at the top of agenda for the next developer meetup.
   In that meeting, a core developer is assigned to review the PR and either approve or reject within five days of the meeting excluding weekends.

Failure of lazy consensus, in the above sense, can arise only under the following condition:
at least one approval and at least one rejection in the PR.

When no consensus can be found, the decision is escaled to :ref:`stage-2`.

.. _stage-2:

Stage 2: voting
~~~~~~~~~~~~~~~

Voting takes place:

* when no lazy consensus can be found in stage 1 above
* for appointments

-  The start of a voting period after stage 1 is at the moment the lazy consensus fails.
-  Start and end time of the vote must be announced in the core developer channel, and on the PR (if on a PR).
-  The vote will conclude 5 days excluding weekends from the call for the vote.
-  Votes are voluntary. Abstentions are allowed. Core developers can
   abstain by simply not casting a vote.
-  All votes are a binary vote: for or against accepting the proposal.
-  Votes are casts as comments: +1 (approval) or -1 (rejection).

For all types of changes, except appointments, votes take place on the
related public issue or pull request. The winning condition is a 2/3
majority of the votes cast by core developers (including CC members) for the proposal.
If the proposal cannot gather a 2/3 majority of the votes cast by core
developers, the decision is escalated to the :ref:`stage-3`.

For appointments, votes take place in private communication channels
and are anonymous. The winning conditions vary depending on the role as
described in :ref:`roles` above. Appointment decisions are not escalated to
the CC. If a nomination cannot gather sufficient support, the nomination is
rejected.

.. _stage-3:

Stage 3: conflict resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the proposed change cannot gather a 2/3 majority of the votes cast,
the CC tries to resolve the deadlock.

-  The CC will use consensus seeking.
-  If no consensus can be found within twenty working days excluding weekends
   since the beginning of the stage-1 "reasonable time for consideration" period,
   the decision is made through a simple majority vote (with tie breaking) among the CC
   members.
-  Any proposal reaching stage 3 must be supported by an :ref:`steps`,
   which has been made public at least 5 days, excluding weekends, before the vote.

.. _steps:

sktime enhancement proposal
~~~~~~~~~~~~~~~~~~~~~~~~~~~

sktime enhancement proposals (STEPs) are required for:

* certain types of proposed changes, by default, see `decision making process <#Decision-making>`__
* for all stage 3 decisions

If a STEP is required by a vote, it must have been made public at least 5 working days (excluding week-ends) before that vote.

A STEP is a consolidated document, with a concise
problem statement, a clear description of the proposed solution and a
comparison with alternative solutions, as outlined in our
`template <https://github.com/sktime/enhancement-proposals/blob/master/TEMPLATE.md>`__.

A complete STEP must always include at least a high-level design for the proposed change,
not just a wishlist of features.

Usually, we collect and discuss proposals in sktime's `repository for
enhancement-proposals <https://github.com/sktime/enhancement-proposals>`__.

For smaller changes, such as punctual changes to the API or governance documents,
the STEP can also be be part of an issue or pull request.

.. _algorithm-inclusion-guidelines:

Algorithm inclusion guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Curation is about how we select contributions, which criteria we use in
order to decide which contributions to include, and in which cases we
deprecate and remove contributions.

We have the following guidelines:

-  ``sktime`` aims to provide a repository for algorithms to enhance reproducible research,
   putting no lower bounds on number of citations, algorithmic performance, or frequency of use.
-  For inclusion, a scientific reference must be available and linked to the python estimator.
   A scientific reference is a formal description of the algorithm which
   satisfies basic scientific requirements, e.g., be formally correct, complete, and
   adhere with common conventions on presentation in the field of data science.
-  The scientific reference must be free from unfounded scientific claims, pseudo-science,
   commercial marketing, or other content inappropriate for a scientific reference.
   The scientific reference must adhere to proper scientific citation standards,
   i.e., citing primary sources, giving proper credit.
   The form of the scientific reference can be a description in the class docstring,
   or a link to a scientific document, e.g., on the arXiv. Such a scientific document
   need not be peer-reviewed or journal published, but must adhere to scientific standards.
-  We strive to consolidate existing functionality if it helps to improve
   the usability and maintainability of the project. For example, when
   there are multiple techniques for the same purpose, we may choose to present one variant as the "primary default",
   and rarer variants as less accessible or findable alternatives. The choice of the "primary default"
   may change with use and relevance in the user community.
   We are aware that the choice of the "primary default" may give or remove visibility,
   and aim to make the choice for usability and quality of the selection.
-  We are happy to accept historical algorithms of interest, as references to use in
   reproduction studies, including historical versions that are faulty implementations.
   Algorithms of historical interest will be clearly labelled as such, and inclusion
   is primarily guided by relevance, e.g., as a reference in an important study,
   relevance in the scientific discourse, or as an important algorithmic baseline.
-  From the algorithms or techniques that meet the above criteria, only
   those which fit well within the current framework of sktime are accepted.
   For algorithms that do not fit well into one of the current API definitions, the API
   will have to be extended first. For extending current API, see the
   `decision making process <#Decision-making>`__ for major changes.

Note that an algorithm need not be in sktime to be fully compatible with
sktime interfaces. You can implement your favorite algorithm in a sktime
compatible way in a third party codebase - open or closed - following
the guide for implementing compatible estimators (see :ref:`developer_guide_add_estimators:`).

We are happy to list any compatible open source project under `related
software <https://github.com/sktime/sktime/wiki/related-software>`__.
Contributions are also welcome to any one of `our companion
repositories <https://github.com/sktime>`__ on GitHub.

Dependencies are managed on the level of estimators, hence it is entirely possible
to maintain an algorithm primarily in a third or second party package, and add a
thin interface to sktime proper which has that package as a dependency.

.. _acknowledging-contributions:

Acknowledging contributions
---------------------------

sktime is collaboratively developed by its diverse community of
developers, users, educators, and other stakeholders. We value all kinds
of contributions and are committed to recognising each of them fairly.

We follow the `all-contributors <https://allcontributors.org>`__
specification to recognise all contributors, including those that don't
contribute code. Please see `our list of all
contributors <https://github.com/sktime/sktime/blob/main/CONTRIBUTORS.md>`__.

If you think, we've missed anything, please let us know or open a PR
with the appropriate changes to
`sktime/.all-contributorsrc <https://github.com/sktime/sktime/blob/main/.all-contributorsrc>`__.

Note that contributors do not own their contributions. sktime is an
open-source project, and all code is contributed under `our open-source
license <https://github.com/sktime/sktime/blob/main/LICENSE>`__.
All contributors acknowledge that they have all the rights to the code
they contribute to make it available under this license.

The project belongs to the sktime community, and all parts of it are
always considered "work in progress" so that they can evolve over time
with newer contributions.

.. _outlook:

Outlook
-------

We are open to improvement suggestions for our governance model. Once
the community grows more and sktime's code base becomes more
consolidated, we will consider the following changes:

-  Allow for more time to discuss changes, and more time to cast vote
   when no consensus can be found,
-  Require more positive votes (less lazy consensus) to accept changes
   during consensus seeking stage,
-  Reduce time for maintainers to reply to issues

In addition, we plan to add more roles for managing/coordinating
specific project:

* Community manager (mentorship, outreach, social media, etc),
* Sub-councils for project-specific technical leadership (e.g.  for documentation, learning tasks, continuous integration)

.. _references:

References
----------

Our governance model is inspired by various existing governance
structures. In particular, we'd like to acknowledge:

* scikit-learn's `governance model <https://scikit-learn.org/stable/governance.html>`__
* `The Turing Way <https://github.com/alan-turing-institute/the-turing-way>`__ project
* `The Art of Community <https://www.jonobacon.com/books/artofcommunity/>`__ by Jono Bacon
* The `astropy <https://www.astropy.org>`__ project
* The `nipy <https://nipy.org>`__ project
* The `scikit-hep <https://scikit-hep.org>`__ project
