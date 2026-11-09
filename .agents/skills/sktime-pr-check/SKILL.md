---
name: sktime-pr-check
description: Check an sktime pull request or local branch against what maintainers look for, before asking for review or when re-checking after review comments. Covers duplicate work, the PR template, title prefix, contributors file, docstrings, API reference, tags, tests and whether earlier review comments were really addressed. Use this whenever someone is about to open an sktime PR, asks if their branch or PR is ready, wants a self-review, or asks whether review feedback has been handled. It reports findings and drafts text, and never posts to GitHub.
argument-hint: "[PR number, or leave empty for the current branch]"
context: fork
background: false
---

# sktime PR check

Check the target below against the rubric, and report what's ready, what needs work and
what you couldn't verify.

Target: $ARGUMENTS

If the target is empty, check the current branch and any uncommitted changes.

You are checking someone else's work with fresh eyes. You have no conversation history, so
take everything from the repo and GitHub. Be honest: a clean-looking report on unverified
work is worse than none, and it's the contributor who pays for it in review.

Only read and report. Don't edit files, commit or push. Never post comments, approve or
request changes on GitHub.

## 1. Collect the facts

For a PR number or URL:

```bash
gh pr view <N> --repo sktime/sktime --json title,author,body,files,commits,reviews,headRefName,isDraft
gh pr diff <N> --repo sktime/sktime
gh pr checks <N> --repo sktime/sktime
```

To read the code as it stands in the PR, fetch it to a local ref without changing the checkout:
`git fetch https://github.com/sktime/sktime pull/<N>/head:pr-<N>`.

For the current branch:

```bash
git fetch https://github.com/sktime/sktime main
git status --short
git diff FETCH_HEAD...HEAD --stat
git diff FETCH_HEAD...HEAD
git diff HEAD
git log FETCH_HEAD..HEAD --format='%h %s'
```

Then work out what kind of change it is, since that decides which parts of the rubric apply.
A new estimator shows a new class that subclasses a `Base*` class. A foundation model has
`"foundation_model"` in its `category` tag.

## 2. Check for duplicate work

```bash
gh pr list --repo sktime/sktime --state open --search "<key terms>"
gh issue view <linked issue> --repo sktime/sktime --json title,assignees,comments
```

Report an open PR for the same thing, or a linked issue assigned to someone else, as the top
finding. No other polish matters if the work is a duplicate.

## 3. Re-review, if there are earlier comments

Read all review threads:

```bash
gh api --paginate repos/sktime/sktime/pulls/<N>/comments
gh api --paginate repos/sktime/sktime/pulls/<N>/reviews
gh api --paginate repos/sktime/sktime/issues/<N>/comments
```

Maintainer guidance is often in the conversation rather than inline. For each request, read the
**current code**, not the reply. A reply saying "done" can mark something that's only partly
handled, and resolved threads can still be wrong. List each one as Addressed, Partly addressed
or Not addressed, with evidence.

## 4. Apply the rubric

Read `references/rubric.md`, and go through it. Also:

- **Foundation model:** apply
  `../sktime-add-foundation-model/references/adapter-checklist.md`, and
  `../sktime-add-foundation-model/references/vendoring.md` if it vendors code.
- **Changed public signature, default, import or behaviour:** check it against
  `../sktime-deprecation/SKILL.md`.
- **New estimator:** the scitype's file in `../sktime-add-estimator/references/` has the
  specifics.

Give every rubric item one status:

- **OK**: you checked it, and it holds. Give the evidence.
- **Needs work**: give `file:line`, what's wrong, and a concrete fix.
- **N/A**: doesn't apply here.
- **Not verified**: you couldn't check it. Say why, and how the contributor can.

Never mark something OK because it looks fine. If you didn't run or read it, it's Not verified.

Run what you can cheaply: `pre-commit run --files <changed files>` if available, and the tests
for changed code, using the commands in "Running tests" in `AGENTS.md`. Don't install dependencies or download
weights. Mark those items Not verified.

## 5. Report

Use this format. Keep findings short and ordered by how much they'd hold up the PR.

```markdown
# PR check: <title or branch>

**Verdict:** Ready for review | Needs work | Blocked (duplicate or scope)
**Change type:** <new estimator / bug fix / foundation model / docs / ...>

## Needs work
1. `path/file.py:42` - <problem>. Fix: <concrete fix>.

## Not verified
- <item>: <why>. To check: `<command>`.

## Earlier review comments
(only on a re-review)
- <comment summary>: Addressed / Partly addressed / Not addressed - <evidence>

## Rubric
| Item | Status | Evidence |
|---|---|---|
```

If the PR description is missing or incomplete, add a drafted version at the end, following
`.github/PULL_REQUEST_TEMPLATE.md`, for the contributor to paste in. See the disclosure
item in the rubric.
