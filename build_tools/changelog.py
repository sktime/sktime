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
    """Fetch a page of merged pull requests."""
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


def fetch_latest_release():
    """Fetch the latest release from the GitHub API."""
    import httpx

    response = httpx.get(
        f"{GITHUB_REPOS}/{OWNER}/{REPO}/releases/latest", headers=HEADERS
    )

    if response.status_code == 200:
        return response.json()
    else:
        raise ValueError(response.text, response.status_code)


def fetch_pull_requests_since_last_release() -> list[dict]:
    """Fetch all pull requests merged since last release."""
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
    """Find unique authors and print a list in given format."""
    authors = sorted({pr["user"]["login"] for pr in prs}, key=lambda x: x.lower())

    header = "Contributors"
    if fmt == "github":
        print(f"### {header}")
        print(", ".join(f"@{user}" for user in authors))
    elif fmt == "rst":
        print(header)
        print("~" * len(header), end="\n\n")
        print(",\n".join(f":user:`{user}`" for user in authors))


# CHANGE 1: Assign both category and module 
def assign_prs(prs, categs):
    assigned = defaultdict(list)
    pr_modules = {}

    for i, pr in enumerate(prs):
        pr_labels = [label["name"] for label in pr["labels"]]

        for cat in categs:
            if not set(cat["labels"]).isdisjoint(set(pr_labels)):
                assigned[cat["title"]].append(i)

        for label in pr_labels:
            if label.startswith("module:"):
                pr_modules[i] = label.replace("module:", "")

    assigned["Other"] = list(set(range(len(prs))) - {i for lst in assigned.values() for i in lst})

    return assigned, pr_modules


def render_row(pr):
    """Render a single row with PR in restructuredText format."""
    print(
        "*",
        pr["title"].replace("`", "``"),
        f"(:pr:`{pr['number']}`)",
        f":user:`{pr['user']['login']}`",
    )


# CHANGE 2: Group under modules using pr_modules
def render_changelog(prs, assigned, pr_modules):
    from dateutil import parser

    module_to_title = {
        "forecasting": "Forecasting",
        "classification": "Time Series Classification",
        "regression": "Time Series Regression",
        "clustering": "Time Series Clustering",
        "transformations": "Transformations",
        "annotation": "Annotation, Changepoints",
        "metrics": "Benchmarking, Metrics",
        "tests": "Test Framework",
        "vis": "Visualization",
        "registry": "Registry and Search",
        "pipeline": "Pipelines",
        "base": "Base Classes",
    }

    for category, pr_indices in assigned.items():
        if not pr_indices:
            continue

        print(f"\n{category}")
        print("~" * len(category), end="\n\n")

        if category in ["Enhancements", "Fixes"]:
            by_module = defaultdict(list)
            others = []

            for i in pr_indices:
                if i in pr_modules:
                    key = pr_modules[i]
                    title = module_to_title.get(key, "Other")
                    by_module[title].append(prs[i])
                else:
                    others.append(prs[i])

            for module_title in sorted(by_module):
                print(module_title)
                print("^" * len(module_title), end="\n\n")
                for pr in sorted(by_module[module_title], key=lambda p: parser.parse(p["merged_at"])):
                    render_row(pr)

            if others:
                print("Other")
                print("^^^^^\n")
                for pr in sorted(others, key=lambda p: parser.parse(p["merged_at"])):
                    render_row(pr)
        else:
            for i in sorted(pr_indices, key=lambda i: parser.parse(prs[i]["merged_at"])):
                render_row(prs[i])


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
    # CHANGE 3: unpack both values
    assigned, pr_modules = assign_prs(pulls, categories)
    render_changelog(pulls, assigned, pr_modules)
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
    