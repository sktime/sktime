Governance
==========

Overview
--------
sktime is a meritocratic, consensus-based community project. Anyone with an interest in the project can join the community, contribute to the project, and participate in the governance process. This document describes how that participation takes place and how to set about earning merit within the project community. We are particularly motivated to support new and/or anxious collaborators, people who are looking to learn and develop their skills, and anyone who has experienced discrimination in the past.

In particular, we will outline

* how we expect all members of the sktime community to behave,
* what priviledges and responsibilities authors and maintainers have,
* how we curate and select new contributions and existing functionality,
* how we make decisions,
* how we acknowledge contributions.


Code of Conduct
---------------
We value the participation of every member of our community and want to ensure an that every contributor has an enjoyable and fulfilling experience. Accordingly, everyone who participates in the sktime project is expected to show respect and courtesy to other community members at all times.

To make clear what is expected, we ask all members of the community to
conform to our `Code of Conduct <https://github
.com/alan-turing-institute/sktime/blob/master/CODE_OF_CONDUCT.rst>`_.

Authorship and maintainership of algorithms
-------------------------------------------

Authors and maintainers of algorithms have special rights and duties with regard to changes and maintenance of their algorithms and associated documentation, including doc strings and tutorial notebooks.

In sktime, algorithms are encapsulated in classes and are called estimators. Authorship and maintainership operate on the level of classes. To faciliate authorship and maintainership questions, we try to write algorithms in separate files when possible. If an algorithm uses multiple files, we try to collect them in a folder.

The author is the contributor who implements and contributes the algorithm to the project. The maintainer is the contributor who is responsible for maintaining an algorithm and the first point of contact for users and other contributors for all issues, questions, and proposals regarding their algorithm. The author automatically becomes the first maintainer of the algorithm.

* **Veto right.** The maintainer has the right to veto any proposed changes to their algorithm within a time frame of 1 week. This does not extend to change to the common framework and API.

* **Maintenance duty.** The maintainer is responsible for maintaining an
algorithm. This includes understandable and instructive documentation, bugs fixes, unit testing, coding quality and style improvements, and compliance with sktime's common API. Maintainers should reply to a new issue on their algorithm within 10 working days, and demonstrate a significant effort to fix any issue within 30 working days, ignoring any public holidays.

Any maintainer is expected to give up their role if they can no longer
fulfil their maintenance duty. If a  maintainer fails to fulfil their
responsibilities in the given time frame, the maintainer loses their role and all rights that come with it.

Note that authors and maintainers do not *own* the algorithms or files they contributed or maintain. sktime is an open-source project, and all code is contributed under our 3-clause-BSD license. By contributing to sktime, ownership is transferred to all sktime developers and the project as a whole.


Curation and inclusion criteria for algorithms
----------------------------------------------

Curation is about how we select contributions, which criteria we use in order to decide which contributions to include, and in which cases we deprecate and remove contributions.

We have the following guidelines:

* We only consider published algorithms which have been shown to be competitive in comparative benchmarking studies or practically useful in applied projects. A technique that provides a clear-cut improvement (e.g. an enhanced data structure or a more efficient approximation technique) on a widely-used method will also be considered for inclusion.
* From the algorithms or techniques that meet the above criteria, only those which fit well within the current framework and API of sktime are accepted. For extending current frameworks and API, see the [process for major changes]().
* The contributor should support the importance of the proposed addition with research papers and/or implementations in other similar packages, demonstrate its usefulness via common use-cases/applications and corroborate performance improvements, if any, with benchmarks and/or plots. It is expected that the proposed algorithm should outperform the methods that are already implemented in sktime in at least some areas.
* We strive to consolidate existing functionality if helps to improve the usability and maintainability of the project. For example, when there are multiple techniques for the same purpose, we prefer to combine them into a single class and make case distinctions based on hyper-parameters.

Note that your implementation need not be in sktime to be used together with sktime tools. You can implement your favorite algorithm in a sktime
compatible way in one of `our companion repositories <https://github
.com/sktime>`_ on GitHub. We will be happy to list it under `related
software <https://github.com/alan-turing-institute/sktime/wiki/related
-software>`_.

If algorithms require major dependencies, we encourage to create a separate companion repository. For example, for deep learning techniques based on TensorFlow and Keras, we have `sktime-dl <https://github.com/sktime/sktime-dl>`_. For smaller dependencies which are limited to a few files, we encourage to use soft dependencies, which are only required for particular modules, but not for most of sktime's functionality and not for installing sktime.

If significant issues are not fixed by the maintainer, and no other contributor volunteers to fix the algorithm within 90 working days, we reserve the right to eventually deprecate and remove the algorithm from sktime.

Decision making
---------------

The purpose of this section is to formalize the decision-making process used by the sktime project. We clarify how decisions are made and who can make them.

sktime's decision-making process is designed to take into account feedback from all community members and strives to find consensus, while avoiding deadlocks when no consensus can be found.

Decisions about the future of the project are made through discussion with all members of the community. All discussion takes place on the project’s `issue tracker <https://github.com/alan-turing-institute/sktime/issues>`_ or pull requests. Sensitive discussion can occur on private chats.

sktime uses a "consensus seeking" process for making decisions. The community tries to find a resolution that has no open objections among core developers.

Roles
~~~~~
Throughout the decision making process, we differentiate between three roles:

* Contributors
* Core developers
* Technical committee members

Contributors
++++++++++++

Contributors are community members who contribute in concrete ways to the
project. Anyone can become a contributor, and contributions can take many
forms – not only code – as detailed in the `contributing guide <https://sktime.org/how_to_contribute.html>`_.

Core developers
+++++++++++++++

Core developers are community members who have shown that they are dedicated to the continued development of the project through ongoing engagement with the community. They have shown they can be trusted to maintain sktime with care.

* **Direct access.** Being a core developer allows contributors to more easily carry on with their project related activities by giving them direct access to the project’s repository.
* **Issue/PR management.** Core developers can review and manage issues and pull requests. This includes commenting on issues, reviewing code contributions, merging approved pull requests, and closing issues once resolved.
* **Voting.** They can cast votes for and against merging a pull-request, and can be involved in deciding major changes to the API.

New core developers can be nominated by any existing core developers. Once they have been nominated, there will be a vote by the current core developers.

Voting on new core developers is one of the few activities that takes place on the project's private chat or management list. While it is expected that most votes will be unanimous, a two-thirds majority of the cast votes is enough. The vote needs to be open for at least 1 week.

Core developers that have not contributed to the project (commits or GitHub comments) in the past 12 months will be asked if they want to become *emeritus core developers* and give up their direct-access, management and voting rights until they become active again.

The list of core developers, active and emeritus (with dates at which they became active) is public on the sktime website.

Technical committee
+++++++++++++++++++

The technical committee (TC) members are core developers who have additional rights and responsibilities to avoid deadlocks and to ensure the smooth running of the project. TC members are expected to participate in strategic planning, and approve changes to the governance model.

The purpose of the TC is to ensure a smooth progress from the big-picture perspective. Changes that impact the full project require a synthetic analysis and a consensus that is both explicit and informed. In cases that the core developer community (which includes the TC members) fails to reach a consensus, the TC is the entity to resolve the issue.

Membership of the TC is by nomination by a core developer and a vote by all core developers. A nomination will result in discussion which cannot take more than a week and then a vote by the core developers which will stay open for a week. TC membership votes are subject to a two-third majority of all cast votes as well as a simple majority approval of all the current TC members.

TC members who do not actively engage with the TC duties are expected to resign.

The initial members of the TC are:

* Markus Löning - @mloning
* Franz Király - @fkiraly
* Anthony Bagnall - @TonyBagnall

Voting: lazy consensus with veto right
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When no consensus can be found, any core developer can call for a vote at
any point during the discussion. The vote will conclude 10 working days from the call for the vote.

Votes are public and voluntary. Abstentions are allowed. All votes are a binary vote: for (+1) or against (-1) accepting the proposed changes. Votes take place on the issue or pull request. Votes are casts as comments: +1 or -1.

If no option can gather two thirds of the votes cast, the decision is escalated to the TC, which in turn will use consensus seeking with the fallback option of a simple majority vote if no consensus can be found within a month. Any TC decision must be backed by an enhancement proposal.

Decisions (in addition to adding core developers and TC membership as above) are made according to the following rules:

* **Additions**, such as new algorithms: Requires +1 by one core developer, no -1 by a core developer (lazy consensus), happens on the issue or PR page.
* **Minor documentation changes**, such as typo fixes, or addition/correction of a sentence: Requires +1 by one core developer, no -1 by a core developer (lazy consensus), happens on the pull request page. Core developers are expected to give “reasonable time” to others to give their opinion on the pull request if they’re not confident others would agree.
* **Code changes and major documentation changes** require +1 by one core developers, no -1 by a core developer or code maintainer (lazy consensus), happens on the pull-request page.
* **Changes to the API design and changes to dependencies or supported versions** happen via an enhancement proposal and follows the decision-making process outlined above.
* **Changes to the governance model** use the same decision process outlined above.

If a veto (-1) vote is cast on a lazy consensus, the proposer can appeal to the community and core developers. The change can be approved or rejected using the decision making process outlined above.

sktime enhancement proposals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For all decision of the TC, a proposal must have been made public and discussed before the vote. An enhancement proposal must be a consolidated document, rather than a long discussion on an issue.


Future directions
~~~~~~~~~~~~~~~~~
Once sktime's API, frameworks, and content becomes more consolidated or when the community has grown more, we will consider the following changes to ensure the smooth running of the project:

* Allow for more time to discuss changes, and more time to cast vote when no consensus had been found,
* Require more positive votes to accept changes during the decision making process,
* Reduce time for maintainers to reply to issues


Acknowledgments
---------------
We follow the `all-contributors <https://allcontributors.org>`_ specification to recognise all contributors, including those that don't contribute code. Please see `our list of all contributors <https://github.com/alan-turing-institute/sktime/blob/master/CONTRIBUTORS.md>`_.

If you think, we've missed anyone, please let us know or open a PR with the appropriate changes to `sktime/.all-contributorsrc <https://github
.com/alan-turing-institute/sktime/blob/master/.all-contributorsrc>`_.


References
----------

Large parts of sktime's governance model are adapted from `scikit-learn's
governance model <https://sktime.org/stable/governance.html>`_.
