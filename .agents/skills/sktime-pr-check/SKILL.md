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

The rubric is `.github/PULL_REQUEST_TEMPLATE.md`, plus the checks below that it doesn't
cover. Read the template, and make each of its sections and each item of its PR checklist a
rubric item. Leave out the ones for new estimators if no estimator was added. For the
`.all-contributorsrc` item, check with `grep -n '"login": "<handle>"' .all-contributorsrc`
and `python -m json.tool .all-contributorsrc > /dev/null`.

Then check:

- **Scope.** No other open PR does this (step 2). Changes unrelated to the stated purpose,
  including formatting churn in untouched code, go in a separate PR. No local scripts,
  scratch notebooks, lockfiles, editor settings or generated output in the diff.
- **AI disclosure.** The template's hidden last comment asks for the header in "Pull
  requests" in `AGENTS.md`. If the PR description or an issue was written by an LLM,
  including one you draft here, that header goes first. Contributors easily miss hidden
  comments, so point it out when it's absent from text that looks generated. A human who
  wrote the text themselves doesn't need it.
- **Docstrings.** Compare each docstring to its signature line by line, since this is the
  most common review finding. Every `__init__` parameter is documented as
  `name : type, default=value`, with spaces around the colon and the default matching the
  code. Public fitted attributes are under `Attributes`. Links go under `References`, cited
  with `[1]_`, not inline. The prose matches the code: units, sign, shapes.
- **Tests.** New behaviour is tested beyond `check_estimator`, with assertions on values.
  Tests needing soft dependencies are gated as the dependency guide describes. No
  monkeypatched fake models, which only prove the mock works. `tests:skip_by_name` has a
  comment saying why, ideally with an issue, and `tests:skip_all` needs a strong reason. If
  the test summary shows nothing passed, or everything skipped, the item is Not verified.
- **A bug fix has a test that fails without the fix.** Prove it where possible, in a separate
  worktree so the checkout stays untouched:

  ```bash
  BASE="$(mktemp -d)/base"
  git worktree add "$BASE" FETCH_HEAD
  cp <test file> "$BASE/<test file>"
  (cd "$BASE" && python -m pytest <test file> --only_changed_modules False --matrixdesign False -n0)
  git worktree remove --force "$BASE"
  ```

  `python -m pytest` matters here, because it imports `sktime` from the worktree rather than
  the editable install.
- **Dependencies.** Soft dependencies follow "Soft dependencies" in `AGENTS.md`. New core
  dependencies need agreement from core developers. Version bounds are justified, and lower
  bounds were actually tried.
- **Code.** No checks that repeat the base class, no dead branches, no defensive shape
  conditionals where inner mtypes guarantee the input. No debug prints or commented-out code.
  Comments explain non-obvious reasons. Standard private signatures are kept, even if CodeQL
  flags a mismatch.
- **CI.** Pre-commit passes. For a PR, check `gh pr checks`, and for each failure say whether
  this change caused it. Check `main` before calling a failure unrelated. For a local branch
  CI hasn't run, so the item is Not verified.

Depending on the change type, also apply:

- **New estimator:** steps 3 to 6 of `../sktime-add-estimator/SKILL.md`, and the scitype's
  file in `../sktime-add-estimator/references/`.
- **Foundation model:** apply
  `../sktime-add-foundation-model/references/adapter-checklist.md`, and
  `../sktime-add-foundation-model/references/vendoring.md` if it vendors code.
- **Changed public signature, default, import or behaviour:** check it against
  `../sktime-deprecation/SKILL.md`. A deprecation needs a warning, a docstring note,
  `todo <version>` markers and tests.

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
`.github/PULL_REQUEST_TEMPLATE.md`, for the contributor to paste in. Start it with the
disclosure header, since you wrote it.
