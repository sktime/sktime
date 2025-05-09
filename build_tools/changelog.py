"""RestructuredText changelog generator."""

import os
from collections import defaultdict

HEADERS = {
    "Accept": "application/vnd.github.v3+json",
}

if os.getenv("GITHUB_TOKEN") is not None:
    HEADERS["Authorization"] = f"token {os.getenv('GITHUB_TOKEN')}"

OWNER = "sktime"
REPO = "sktime"
GITHUB_REPOS = "https://api.github.com/repos"


def fetch_merged_pull_requests(page: int = 1) -> list[dict]:
    """Fetch a page of merged pull requests.

    Parameters
    ----------
    page : int, optional
        Page number to fetch, by default 1.
        Returns all merged pull request from the ``page``-th page of closed PRs,
        where pages are in descending order of last update.

    Returns
    -------
    list
        List of merged pull requests from the ``page``-th page of closed PRs.
        Elements of list are dictionaries with PR details, as obtained
        from the GitHub API via ``httpx.get``, from the ``pulls`` endpoint.
    """
    import httpx

    params = {
        "base": "main",
        "state": "closed",
        "page": page,
        "per_page": 50,
        "sort": "updated",
        "direction": "desc",
    }
    r = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/pulls",
        headers=HEADERS,
        params=params,
    )
    return [pr for pr in r.json() if pr["merged_at"]]


def fetch_latest_release():  # noqa: D103
    """Fetch the latest release from the GitHub API.

    Returns
    -------
    dict
        Dictionary with details of the latest release.
        Dictionary is as obtained from the GitHub API via ``httpx.get``,
        for ``releases/latest`` endpoint.
    """
    import httpx

    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/releases/latest", headers=HEADERS
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


def fetch_pull_requests_since_last_release() -> list[dict]:
    """Fetch all pull requests merged since last release.

    Returns
    -------
    list
        List of pull requests merged since the latest release.
        Elements of list are dictionaries with PR details, as obtained
        from the GitHub API via ``httpx.get``, through ``fetch_merged_pull_requests``.
    """
    from dateutil import parser

    release = fetch_latest_release()
    published_at = parser.parse(release["published_at"])
    print(f"Latest release {release['tag_name']} was published at {published_at}")

    is_exhausted = False
    page = 1
    all_pulls = []
    while not is_exhausted:
        pulls = fetch_merged_pull_requests(page=page)
        all_pulls.extend(
            [p for p in pulls if parser.parse(p["merged_at"]) > published_at]
        )
        is_exhausted = any(parser.parse(p["updated_at"]) < published_at for p in pulls)
        page += 1
    return all_pulls


def github_compare_tags(tag_left: str, tag_right: str = "HEAD"):
    """Compare commit between two tags."""
    import httpx

    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/compare/{tag_left}...{tag_right}"
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


def render_contributors(prs: list, fmt: str = "rst"):
    """Find unique authors and print a list in  given format."""
    authors = sorted({pr["user"]["login"] for pr in prs}, key=lambda x: x.lower())

    header = "Contributors"
    if fmt == "github":
        print(f"### {header}")
        print(", ".join(f"@{user}" for user in authors))
    elif fmt == "rst":
        print(header)
        print("~" * len(header), end="\n\n")
        print(",\n".join(f":user:`{user}`" for user in authors))


def assign_prs(prs, categs: list[dict[str, list[str]]]):
    """Assign PR to categories based on labels."""
    assigned = defaultdict(list)

    for i, pr in enumerate(prs):
        for cat in categs:
            pr_labels = [label["name"] for label in pr["labels"]]
            if not set(cat["labels"]).isdisjoint(set(pr_labels)):
                assigned[cat["title"]].append(i)

    #             if any(l.startswith("module") for l in pr_labels):
    #                 print(i, pr_labels)

    assigned["Other"] = list(
        set(range(len(prs))) - {i for _, j in assigned.items() for i in j}
    )

    return assigned


def render_row(pr):
    """Render a single row with PR in restructuredText format."""
    print(
        "*",
        pr["title"].replace("`", "``"),
        f"(:pr:`{pr['number']}`)",
        f":user:`{pr['user']['login']}`",
    )


def render_changelog(prs, assigned):
    # sourcery skip: use-named-expression
    """Render changelog."""
    from dateutil import parser

    for title, _ in assigned.items():
        pr_group = [prs[i] for i in assigned[title]]
        if pr_group:
            print(f"\n{title}")
            print("~" * len(title), end="\n\n")

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
    print(f"Found {len(pulls)} merged PRs since last release")
    assigned = assign_prs(pulls, categories)
    render_changelog(pulls, assigned)
    print()
    render_contributors(pulls)

    release = fetch_latest_release()
    diff = github_compare_tags(release["tag_name"])
    if diff["total_commits"] != len(pulls):
        raise ValueError(
            "Something went wrong and not all PR were fetched. "
            f"There are {len(pulls)} PRs but {diff['total_commits']} in the diff. "
            "Please verify that all PRs are included in the changelog."
        )
