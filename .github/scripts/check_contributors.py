"""Check .all-contributorsrc against merged PR and issue records."""

import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone

import httpx

HEADERS = {"Accept": "application/vnd.github.v3+json"}
if os.getenv("GITHUB_TOKEN"):
    HEADERS["Authorization"] = f"token {os.getenv('GITHUB_TOKEN')}"

OWNER = "sktime"
REPO = "sktime"
API_BASE = f"https://api.github.com/repos/{OWNER}/{REPO}"

RC_FILE = ".all-contributorsrc"
LOOKBACK_DAYS = 30

# PR label -> contribution types for PR author
LABEL_MAP = {
    "enhancement": ["code"],
    "bug": ["code"],
    "documentation": ["doc"],
    "maintenance": ["maintenance"],
}

# Title prefix fallback, used only when no label matched
PREFIX_MAP = {
    "[ENH]": ["code"],
    "[BUG]": ["code"],
    "[DOC]": ["doc"],
    "[MNT]": ["maintenance"],
}

CLOSES_RE = re.compile(
    r"(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)",
    re.IGNORECASE,
)


def gh_get(url, **kwargs):
    """GET with basic rate-limit retry."""
    r = httpx.get(url, headers=HEADERS, **kwargs)
    if r.status_code == 403 and "rate limit" in r.text.lower():
        import time

        reset = int(r.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset - int(time.time()), 5)
        time.sleep(wait)
        r = httpx.get(url, headers=HEADERS, **kwargs)
    r.raise_for_status()
    return r


def fetch_merged_prs(since):
    """Return PRs merged after `since`."""
    page, merged = 1, []
    while True:
        r = gh_get(
            f"{API_BASE}/pulls",
            params={
                "state": "closed",
                "per_page": 50,
                "page": page,
                "sort": "updated",
                "direction": "desc",
            },
        )
        pulls = r.json()
        if not pulls:
            break
        for pr in pulls:
            if pr.get("merged_at"):
                merged_at = datetime.fromisoformat(
                    pr["merged_at"].replace("Z", "+00:00")
                )
                if merged_at >= since:
                    merged.append(pr)
        # stop when we've gone past our window
        oldest_updated = datetime.fromisoformat(
            pulls[-1]["updated_at"].replace("Z", "+00:00")
        )
        if oldest_updated < since:
            break
        page += 1
    return merged


def fetch_bug_reporters(merged_prs):
    """Find bug-issue reporters from 'Fixes #N' references in merged PRs."""
    reporters = {}
    for pr in merged_prs:
        body = pr.get("body") or ""
        for match in CLOSES_RE.finditer(body):
            issue_num = int(match.group(1))
            if issue_num in reporters:
                continue
            r = gh_get(f"{API_BASE}/issues/{issue_num}")
            issue = r.json()
            labels = {lb["name"].lower() for lb in issue.get("labels", [])}
            if "bug" in labels:
                reporters[issue_num] = issue["user"]["login"]
    return reporters


def contribution_types_for_pr(pr):
    """Return contribution types for a PR, checking labels then title prefix."""
    pr_labels = {lb["name"].lower() for lb in pr.get("labels", [])}
    title = (pr.get("title") or "").strip()
    types = set()

    for label, contribs in LABEL_MAP.items():
        if label in pr_labels:
            types.update(contribs)

    if not types:
        for prefix, contribs in PREFIX_MAP.items():
            if title.upper().startswith(prefix):
                types.update(contribs)

    return sorted(types)


def build_required(merged_prs, bug_reporters):
    """Build {login: {contribution_types}} from PRs and bug reporters."""
    required = {}
    for pr in merged_prs:
        author = pr.get("user", {}).get("login", "")
        if not author or author.lower().endswith("[bot]"):
            continue
        types = contribution_types_for_pr(pr)
        if types:
            required.setdefault(author.lower(), set()).update(types)

    for reporter in bug_reporters.values():
        if reporter and not reporter.lower().endswith("[bot]"):
            required.setdefault(reporter.lower(), set()).add("bug")

    return required


def fetch_user_info(login):
    """Fetch name, avatar, and profile URL for a GitHub user."""
    try:
        r = gh_get(f"https://api.github.com/users/{login}")
        data = r.json()
        return {
            "name": data.get("name") or login,
            "avatar_url": data.get("avatar_url", ""),
            "profile": data.get("html_url", ""),
        }
    except Exception:
        return {"name": login, "avatar_url": "", "profile": ""}


def load_rc(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_rc(path, data):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def existing_contributions(rc_data):
    """Return {login_lower: set(contribution_types)} from rc_data."""
    mapping = {}
    for entry in rc_data.get("contributors", []):
        login = entry.get("login", "").lower()
        if login:
            mapping[login] = set(entry.get("contributions", []))
    return mapping


def audit_and_fix(rc_data, required):
    """Patch rc_data for missing contributions. Return (rc_data, changes)."""
    current = existing_contributions(rc_data)
    changes = []

    for login, req_types in sorted(required.items()):
        missing = req_types - current.get(login, set())
        if not missing:
            continue

        changes.append(f"{login}: +{sorted(missing)}")

        patched = False
        for entry in rc_data.get("contributors", []):
            if entry.get("login", "").lower() == login:
                entry["contributions"] = sorted(
                    set(entry.get("contributions", [])) | missing
                )
                patched = True
                break

        if not patched:
            info = fetch_user_info(login)
            rc_data.setdefault("contributors", []).append(
                {
                    "login": login,
                    "name": info["name"],
                    "avatar_url": info["avatar_url"],
                    "profile": info["profile"],
                    "contributions": sorted(missing),
                }
            )

    return rc_data, changes


if __name__ == "__main__":
    since = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    merged_prs = fetch_merged_prs(since)
    if not merged_prs:
        print("No merged PRs in lookback window.")
        sys.exit(0)

    bug_reporters = fetch_bug_reporters(merged_prs)
    required = build_required(merged_prs, bug_reporters)

    rc_data = load_rc(RC_FILE)
    updated, changes = audit_and_fix(rc_data, required)

    if not changes:
        print("All contributions up to date.")
        sys.exit(0)

    print(f"Fixed {len(changes)} missing contribution(s):")
    for c in changes:
        print(f"  {c}")

    save_rc(RC_FILE, updated)