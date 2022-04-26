# -*- coding: utf-8 -*-

"""RestructuredText changelog generator."""

import os
from collections import defaultdict
from typing import Dict, List

import httpx
from dateutil import parser

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}

if os.getenv("GITHUB_TOKEN") is not None:
    HEADERS["Authorization"] = f"token {os.getenv('GITHUB_TOKEN')}"

OWNER = "alan-turing-institute"
REPO = "sktime"


def fetch_merged_pull_requests(page: int = 1) -> List[Dict]:  # noqa
    "Fetch a page of pull requests"
    params = {"state": "closed", "page": page, "per_page": 50}
    r = httpx.get(
        f"https://api.github.com/repos/{OWNER}/{REPO}/pulls",
        headers=HEADERS,
        params=params,
    )
    return [pr for pr in r.json() if pr["merged_at"]]


def fetch_pull_requests_since_last_release() -> List[Dict]:  # noqa

    release = httpx.get(
        f"https://api.github.com/repos/{OWNER}/{REPO}/releases/latest", headers=HEADERS
    ).json()
    published_at = parser.parse(release["published_at"])
    print(  # noqa
        f"Latest release {release['tag_name']} was published at {published_at}"
    )

    is_exhausted = False
    page = 1
    all_pulls = []
    while not is_exhausted:
        pulls = fetch_merged_pull_requests(page=page)
        all_pulls.extend(
            [p for p in pulls if parser.parse(p["merged_at"]) > published_at]
        )
        is_exhausted = any(parser.parse(p["merged_at"]) < published_at for p in pulls)
        page += 1
    return all_pulls


def render_contributors(prs: List, fmt: str = "rst"):  # noqa
    "Find unique authors and print a list in  given format"
    authors = sorted(set(pr["user"]["login"] for pr in prs), key=lambda x: x.lower())

    header = "Contributors"
    if fmt == "github":
        print(f"### {header}")  # noqa
        print(", ".join(f"@{user}" for user in authors))  # noqa
    elif fmt == "rst":
        print(header)  # noqa
        print("~" * len(header), end="\n\n")  # noqa
        print(",\n".join(f":user:`{user}`" for user in authors))  # noqa


def assign_prs(prs, categs):  # noqa

    assigned = defaultdict(list)

    for i, pr in enumerate(prs):
        for cat in categs:
            pr_labels = [label["name"] for label in pr["labels"]]
            if not set(cat["labels"]).isdisjoint(set(pr_labels)):
                assigned[cat["title"]].append(i)

    #             if any(l.startswith("module") for l in pr_labels):
    #                 print(i, pr_labels)

    assigned["Other"] = list(
        set(range(len(prs))) - set([i for _, l in assigned.items() for i in l])
    )
    return assigned


def render_row(pr):  # noqa
    "Render a single row with PR in restructuredText format"
    print(  # noqa
        "*", pr["title"], f"(:pr:`{pr['number']}`)", f":user:`{pr['user']['login']}`"
    )


def render_changelog(prs, assigned):  # noqa
    "Render changelog"
    for title, _ in assigned.items():
        pr_group = [prs[i] for i in assigned[title]]
        if pr_group:
            print(f"\n{title}")  # noqa
            print("~" * len(title), end="\n\n")  # noqa

            for pr in sorted(pr_group, key=lambda x: parser.parse(x["merged_at"])):
                render_row(pr)


if __name__ == "__main__":

    categories = [
        {"title": "Enhancements", "labels": ["feature", "enhancement"]},
        {"title": "Fixes", "labels": ["bug", "fix", "bugfix"]},
        {"title": "Maintenance", "labels": ["maintenance", "chore"]},
        {"title": "Refactored", "labels": ["refactor"]},
        {"title": "Documentation", "labels": ["documentation"]},
    ]

    pulls = fetch_pull_requests_since_last_release()
    print(f"Found {len(pulls)} merged PRs since last release")  # noqa
    assigned = assign_prs(pulls, categories)
    render_changelog(pulls, assigned)
    print()  # noqa
    render_contributors(pulls)
