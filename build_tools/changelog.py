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
    # Track module tags for each PR
    pr_modules = {}

    for i, pr in enumerate(prs):
        pr_labels = [label["name"] for label in pr["labels"]]
        
        # Assign to main categories
        for cat in categs:
            if not set(cat["labels"]).isdisjoint(set(pr_labels)):
                assigned[cat["title"]].append(i)
        
        # Track module tags for each PR
        for label in pr_labels:
            if label.startswith("module:"):
                pr_modules[i] = label.replace("module:", "")

    # Assign unmatched PRs to "Other" category
    assigned["Other"] = list(
        set(range(len(prs))) - {i for _, j in assigned.items() for i in j}
    )

    return assigned, pr_modules


def render_row(pr):
    """Render a single row with PR in restructuredText format."""
    print(
        "*",
        pr["title"].replace("`", "``"),
        f"(:pr:`{pr['number']}`)",
        f":user:`{pr['user']['login']}`",
    )


def render_changelog(prs, assigned, pr_modules=None):
    """Render changelog with subsections based on module tags.
    
    Parameters
    ----------
    prs : list
        List of pull requests.
    assigned : dict
        Dictionary mapping category titles to list of PR indices.
    pr_modules : dict, optional
        Dictionary mapping PR indices to module tags, by default None.
    """
    from dateutil import parser

    # Define module mapping for subsection titles
    module_to_title = {
        "forecasting": "Forecasting",
        "classification": "Time Series Classification",
        "regression": "Time Series Regression",
        "clustering": "Time Series Clustering",
        "transformations": "Transformations",
        "annotation": "Time Series Anomalies, Changepoints, Segmentation",
        "detection": "Time Series Anomalies, Changepoints, Segmentation",
        "base": "BaseObject and Base Framework",
        "metrics": "Benchmarking, Metrics, Splitters",
        "benchmarking": "Benchmarking, Metrics, Splitters",
        "distances": "Distances, Kernels",
        "registry": "Registry and Search",
        "dataset": "Data Sets and Data Loaders",
        "datatypes": "Data Types, Checks, Conversions",
        "alignment": "Time Series Alignment",
        "networks": "Neural Networks",
        "tests": "Test Framework",
        "vis": "Visualization",
        "pipeline": "Pipelines",
        "parameter_est": "Parameter Estimation and Hypothesis Testing",
    }

    for title, _ in assigned.items():
        pr_group = [prs[i] for i in assigned[title]]
        if pr_group:
            print(f"\n{title}")
            print("~" * len(title), end="\n\n")

            # If this is a section we want to divide into subsections and we have module info
            if title in ["Enhancements", "Fixes"] and pr_modules:
                # Group PRs by module
                by_module = defaultdict(list)
                other_prs = []
                
                for pr in sorted(pr_group, key=lambda x: parser.parse(x["merged_at"])):
                    pr_idx = next(i for i, p in enumerate(prs) if p["number"] == pr["number"])
                    if pr_idx in pr_modules and pr_modules[pr_idx] in module_to_title:
                        module_title = module_to_title[pr_modules[pr_idx]]
                        by_module[module_title].append(pr)
                    else:
                        other_prs.append(pr)
                
                # Render PRs by module subsections
                for module_title in sorted(by_module.keys()):
                    print(f"{module_title}")
                    print("^" * len(module_title), end="\n\n")
                    for pr in by_module[module_title]:
                        render_row(pr)
                    print()
                
                # Render PRs without module tags
                if other_prs:
                    if title == "Enhancements" and not by_module:
                        # Don't show "Other" header if there are no module subsections
                        for pr in other_prs:
                            render_row(pr)
                    else:
                        print("Other")
                        print("^" * 5, end="\n\n")
                        for pr in other_prs:
                            render_row(pr)
            else:
                # Regular rendering for Documentation, Maintenance, etc.
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
