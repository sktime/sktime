---
name: sktime-deprecation
description: Handle deprecation and breaking changes in sktime. Covers renaming or removing a parameter, changing a default, moving or removing a class, deprecating tags, picking the removal version, writing the warning and the todo marker, and auditing or actioning overdue version todo markers. Use this whenever an sktime change would alter a public signature, default, import path or the behaviour of existing user code, or when someone asks about a deprecation cycle, a "1.3.0 deprecations" PR or what's due for removal, even if they never use the word deprecation.
---

# sktime deprecation

The policy and the recipes are in `docs/source/developer_guide/deprecation.rst`. Read it
first. This skill adds what the guide leaves out, and the places where recent code has moved
on from the guide's examples. Where they disagree, follow the skill.

Work out which situation applies:

- **A proposed code change** that may need a deprecation: start at step 1.
- **An audit, or actioning what's due**: go to "Audit" and "Carrying out a removal".

## 1. Does this need a deprecation?

Use the lists at the top of the guide, and the guide's "Deprecating and change of
parameters" section. They apply to any public name, so to estimators, functions and tags as
well. Two cases the guide doesn't mention:

- Private names, with a leading underscore, don't need one.
- Bug fixes are the grey area. A fix that makes the code do what's documented doesn't need
  one, but if users could reasonably rely on the old result, ask the user whether to raise it
  with maintainers, rather than deciding alone.

If no deprecation is needed, say why and stop.

## 2. Pick the version

Apply the timeline in "Deprecation policy" in the guide to the current version, read from
upstream, since a fork may be out of date:

```bash
git fetch https://github.com/sktime/sktime main && git show FETCH_HEAD:sktime/__init__.py | grep __version__
```

From `1.2.x`, the warning ships in the next release, `1.3.0` is the full cycle with the
warning, and **the change happens at `1.4.0`**. If the PR might merge after the next MINOR
is cut, the target moves out by one, so say so in the PR. If the user or a maintainer wants
a different version, that's their call, but state that it shortens the notice period.

## 3. Apply the recipe

Follow the guide's recipe for the change, under "Special deprecations", and copy its worked
example from "Examples to illustrate recipes" where there is one. Then apply these
corrections, since the examples are older than current practice:

- **`warn` without `obj=`.** Use `warn` from `sktime.utils.warnings` with
  `DeprecationWarning`, and leave out `obj=self`, which the guide's examples still pass.
  With it, a user's `warnings="off"` config silences the warning, and the docstring of `warn`
  says deprecation warnings must always be raised. Recent code such as
  `sktime/transformations/rocket/_multirocket.py` omits it.
- **Lowercase markers:** `# todo 1.4.0: ...`, not the guide's `TODO (release ...)`, so the
  audit below finds them. Put one at each edit site.
- **`super().__init__()`** comes after the parameters are stored, before the warning block.
- **For a rename, both parameters are written to `self`** during the change period, and the
  docstring describes only the new name.

The PR also needs these, which the guide doesn't list for the developer:

- **Tests** that the old usage warns and the new usage doesn't, and that both give the same
  result where behaviour is retained. Mark them `# todo 1.4.0: remove this test`.
- **Migrated internal callers,** so sktime's own code doesn't trigger the warning. Check
  `get_test_params`, docstring examples and notebooks for the old usage.
- **A PR note** stating the change and version, for the release notes. Leave
  `docs/source/changelog.rst` alone, because the release manager writes it.

### Cases the guide doesn't cover

**A parameter shared by several classes:** put the warning in one helper, as
`_check_n_features_per_kernel` in `sktime/transformations/rocket/_multirocket.py` does. Its
todo names every class whose signature and docstring need changing.

**Moving a class or function:** keep a forwarding definition at the old location that
warns and returns the new object, as `sktime/forecasting/model_selection/__init__.py` does
for splitters, but with `warn` from `sktime.utils.warnings` rather than the standard
library's. Import the new location inside it if the two modules could form an import
cycle, and update the API reference to the new location.

**Removing a class or function:** warn on use, in `__init__`, `__post_init__` or the function
body. Name the replacement, or say there is none. Put the todo above the definition.

## 4. Verify

- Run the tests for the changed object, with the commands in "Running tests" in
  `AGENTS.md`.
- Confirm the warning text once by actually triggering it, and read it as a user would. Can
  they fix their code from the message alone?
- Trial the todo actions if they're not trivial. The guide asks that the result doesn't
  break tests when the release manager applies them. A ready branch, linked from the todo, is
  even better.

## Audit

List every version marker in the repo. The guide's spelling and the code's spelling differ,
and older markers such as `TODO (release 0.42.0)` are easy to miss, so match both, ignoring
case. `sktime/libs` is left out because its todos belong to the vendored upstream code:

```bash
grep -rnEi "todo[ :(]*(release |remove in )?v?[0-9]+\.[0-9]+(\.[0-9]+)?" sktime \
  --include='*.py' --exclude-dir=libs
```

Then group by version, and compare against the current version from step 2:

| Marker version | Status |
|---|---|
| Lower than or equal to current | Overdue: it should already be done |
| The next MINOR | Due: action before that release |
| Later | Scheduled, leave alone |

Sort each group by the kind of action, because they aren't all removals:

- **Remove or change** ("remove this function", "change default"): actionable.
- **Check** ("check whether the sklearn bound can be removed", "regularly check..."): needs
  a look at upstream, and the outcome may be to bump the version in the marker.
- **Decide** ("check whether we should remove, otherwise bump"): a maintainer's decision.

Report markers with file and line, plus what each needs. Don't fix anything in an audit
unless the user asks for that.

## Carrying out a removal

Normally the release manager does this, but contributors may prepare a PR.

1. Search for open PRs first. Removals are usually claimed by version:
   `gh pr list --repo sktime/sktime --state open --search "deprecations"`.
2. Find **all** markers for the item. A single removal often has several, across the class,
   helpers and tests, and the marker text may name more sites.
3. Do what the markers say, nothing more, and delete the markers and the warning.
4. Delete the sentinel logic, update the docstring and remove the deprecation tests.
5. Look for other uses of the removed name: `grep -rn "<name>" sktime docs examples`.
6. Run the tests for every touched object. Check downstream effects: flipping the
   `remember_data` default needed fixes in estimators that relied on it (`croston`,
   `greykite`, `timesfm3`), which a marker alone wouldn't reveal.
7. Title it `[MNT] <version> deprecations - <what>`, matching past PRs such as
   `[MNT] 1.3.0 deprecations - remove legacy benchmarking framework`.
