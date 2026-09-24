#!/usr/bin/env python3
"""Script to check if contributors are correctly added to .all-contributorsrc.

Checks merged PRs and closed issues in the last 30 days to see if their authors
have the correct contribution types in .all-contributorsrc.
"""

import json
import os
import ssl
import sys
import urllib.request
from datetime import datetime, timezone

from dateutil.relativedelta import relativedelta


def get_contributors():
    """Read .all-contributorsrc and return a dictionary of contributors."""
    try:
        with open(".all-contributorsrc") as f:
            data = json.load(f)
            return {c["login"].lower(): c["contributions"] for c in data["contributors"]}
    except FileNotFoundError:
        print("Error: .all-contributorsrc not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: .all-contributorsrc is not an invalid JSON.")
        sys.exit(1)


def github_req(url):
    """Make requests to github API."""
    token = os.environ.get("GITHUB_TOKEN")
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github.v3+json")
    if token:
        req.add_header("Authorization", f"token {token}")
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            return json.loads(response.read())
    except urllib.error.URLError as e:
        print(f"Error making request to {url}: {e}")
        return []


def check_recent_contributions():
    """Check recent PRs and Issues and verify .all-contributorsrc."""
    repo = "sktime/sktime"
    # We check PRs merged between 30 days ago and 7 days ago.
    # The 7-day buffer allows time for the all-contributors bot to run and merge.
    now = datetime.now(timezone.utc)
    thirty_days_ago = (now - relativedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    seven_days_ago = (now - relativedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

    contributors = get_contributors()
    missing_contributions = []

    # Get recent merged PRs
    print(f"Fetching merged PRs from {thirty_days_ago} to {seven_days_ago}...")
    prs_url = (
        f"https://api.github.com/search/issues"
        f"?q=repo:{repo}+is:pr+is:merged+merged:{thirty_days_ago}..{seven_days_ago}"
    )
    prs_data = github_req(prs_url)
    prs = prs_data.get("items", []) if isinstance(prs_data, dict) else []

    for pr in prs:
        author = pr["user"]["login"].lower()
        title = pr["title"].lower()
        labels = [l["name"].lower() for l in pr.get("labels", [])]

        if author == "dependabot[bot]" or author == "github-actions[bot]" or "bot" in author:
            continue

        needed_contribs = set()

        # Check title prefixes or labels
        if "[enh]" in title or "[bug]" in title or "enhancement" in labels or "bug" in labels or "bugfix" in labels:
            needed_contribs.add("code")
        if "[doc]" in title or "documentation" in labels:
            needed_contribs.add("doc")
        if "[mnt]" in title or "maintenance" in labels:
            needed_contribs.add("maintenance")
            
        # In case the PR has no tags but is merged, at least we might expect something, 
        # but to avoid false positives we only check if strictly matched with tags.
        
        has_contribs = contributors.get(author, [])
        for needed in needed_contribs:
            if needed not in has_contribs:
                missing_contributions.append({
                    "author": pr["user"]["login"],
                    "missing": needed,
                    "reason": f"PR: {pr['html_url']}"
                })

    # Get recent closed bug issues
    print(f"Fetching closed bug issues from {thirty_days_ago} to {seven_days_ago}...")
    issues_url = (
        f"https://api.github.com/search/issues"
        f"?q=repo:{repo}+is:issue+is:closed+closed:{thirty_days_ago}..{seven_days_ago}"
    )
    issues_data = github_req(issues_url)
    issues = issues_data.get("items", []) if isinstance(issues_data, dict) else []

    for issue in issues:
        author = issue["user"]["login"].lower()
        title = issue["title"].lower()
        labels = [l["name"].lower() for l in issue.get("labels", [])]

        if author == "dependabot[bot]" or author == "github-actions[bot]" or "bot" in author:
            continue

        # Check if it was closed as completed
        if issue.get("state_reason") != "completed":
            continue

        # For issues we check if they are bug reports
        if "[bug]" in title or "bug" in labels or "bugfix" in labels:
            needed = "bug"
            has_contribs = contributors.get(author, [])
            if needed not in has_contribs:
                missing_contributions.append({
                    "author": issue["user"]["login"],
                    "missing": needed,
                    "reason": f"Issue: {issue['html_url']}"
                })

    if missing_contributions:
        print("\nMissing contributions found:")
        for missing in missing_contributions:
            print(f"- @{missing['author']} needs '{missing['missing']}' due to {missing['reason']}")
        sys.exit(1)
    else:
        print("\nAll recent contributors are properly recorded.")


if __name__ == "__main__":
    check_recent_contributions()
