---
name: sktime-deprecation
description: Handle deprecation and breaking changes in sktime. Covers renaming or removing a parameter, changing a default, moving or removing a class, deprecating tags, picking the removal version, writing the warning and the todo marker, and auditing or actioning overdue version todo markers. Use this whenever an sktime change would alter a public signature, default, import path or the behaviour of existing user code, or when someone asks about a deprecation cycle, a "1.3.0 deprecations" PR or what's due for removal, even if they never use the word deprecation.
---

# sktime deprecation

sktime's policy: never break user code without a clear, actionable warning given at
least one full MINOR release in advance. "Break" includes silently changing the
algorithm, not only exceptions. The full guide is
`docs/source/developer_guide/deprecation.rst`.

Work out which situation applies:

- **A proposed code change** that may need a deprecation: start at step 1.
- **An audit, or actioning what's due**: go to "Audit" and "Carrying out a removal".

## 1. Does this need a deprecation?

**Needs one:** changing a parameter's default, renaming or removing a parameter, changing
parameter order, changing an import location, or changing the algorithm for given
arguments. This applies to any public name, so also to estimators, functions and tags.

**Doesn't need one:**

- a new parameter at the end of the list, with a default that keeps prior behaviour
- refactoring that leaves the public API alone
- a different implementation of the same algorithm, e.g. for speed
- removing a parameter whose non-default values always raised anyway
- private names (leading underscore), and bug fixes that make the code do what's documented

Bug fixes are the grey area. If users could reasonably rely on the old result, ask the
user whether to raise it with maintainers, rather than deciding alone.

If no deprecation is needed, say why and stop.

## 2. Pick the version

```bash
python -c "import sktime; print(sktime.__version__)"
git fetch https://github.com/sktime/sktime main && git show FETCH_HEAD:sktime/__init__.py | grep __version__
```

The second command matters when a fork is out of date, since the current version decides
everything below.

The warning has to ship in a release for a **full MINOR cycle** before the change, so the
change usually lands at the **second next MINOR**. From `1.2.x`:

- the warning first ships in the next release
- `1.3.0` is the full cycle with the warning
- **the change happens at `1.4.0`**

Changes happen only at MINOR or MAJOR releases, never PATCH. If the PR might merge after
the next MINOR is cut, the target moves out by one. Say so in the PR. If the user or a
maintainer wants a different version, that's their call, but state that it shortens
the policy's notice period.

## 3. Apply the recipe

Read `references/recipes.md`, and follow the section for the change. It has worked code for
the default change, rename, removal, reorder, new behaviour-changing parameter, class or
function move, and tags.

Whichever recipe applies, the PR always has these parts:

1. **A warning** that names the object, the version of change, and the exact code to write
   instead.
2. **The docstring**, updated with the timeline.
3. **A `todo <version>` comment** at each place that needs editing, stating every action
   explicitly, so that the release manager doesn't have to re-derive anything.
4. **Tests**: the warning fires for the old usage, and the new usage is clean.
5. **Migrated internal callers**, so sktime's own code doesn't trigger the warning.
6. **A PR note** stating the change and version, for the release notes. Leave
   `docs/source/changelog.rst` alone, because the release manager writes it.

Things that often go wrong:

- **`__init__` must store parameters unchanged.** An estimator can't overwrite
  `self.param` with the resolved value, as that breaks `get_params` and `clone`. Keep the
  resolved value in `self._param`, and use that everywhere else.
- **`DeprecationWarning` must always fire.** Use `warn` from `sktime.utils.warnings`, and
  leave out `obj=`. With `obj=self`, a user's `warnings="off"` config silences it, and the
  docstring of that function says deprecation warnings must always be raised. The dev guide's
  example still shows `obj=self`, but recent code such as
  `sktime/transformations/rocket/_multirocket.py` omits it.
- **Use the sentinels from the guide:** `"changing_value"` for a default change,
  `"deprecated"` for a rename or removal, `"position_change"` for reordering.
- **Check `get_test_params`, docstring examples and notebooks** for the old usage.
- **Write markers the audit can find:** lowercase `# todo 1.4.0: ...`, matching recent
  code. Put one at each edit site.
- **A removal test goes too.** Mark deprecation tests with `# todo 1.4.0: remove this test`.

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
