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


def render_row(pr):
    """Render a single row with PR in restructuredText format."""
    # Process the title to handle user credits at beginning of title
    title = pr["title"]
    extra_users = []

    # Check for prefixed user credits like &username
    if title.startswith("&"):
        parts = title.split(" ", 1)
        user_part = parts[0].strip()
        if len(parts) > 1:
            title = parts[1].strip()
            # Extract username without the & prefix
            username = user_part[1:]
            extra_users.append(username)

    # Handle dependabot PRs: enclose package names and versions in double backticks
    if "dependabot" in pr["user"]["login"].lower():
        # Match patterns like "Update package requirement from <1.0.0 to <2.0.0"
        import re

        # Pattern to match package name and version bounds
        pattern = r"Update ([\w\-\.]+) requirement from (<[\d\.]+,?>?[>=]*[\d\.]+) to ([>=]*[\d\.]+,?<[\d\.]+)"  # noqa: E501
        match = re.search(pattern, title)

        pattern2 = r"Bump ([\w\-/]+) from (\d+(?:\.\d+)*) to (\d+(?:\.\d+)*)"  # noqa: E501
        match2 = re.search(pattern2, title)

        if match or match2:
            m = match if match else match2
            package, from_ver, to_ver = m.groups()

            # add double backticks if not already present
            def add_backticks(text):
                if not text.startswith("``") and not text.endswith("``"):
                    return f"`{text}`"
                return text

            package = add_backticks(package)
            from_ver = add_backticks(from_ver)
            to_ver = add_backticks(to_ver)

            # Replace with proper backticks
            if match:
                title = f"Update {package} requirement from {from_ver} to {to_ver}"
            elif match2:
                title = f"Bump {package} from {from_ver} to {to_ver}"

    # Replace single backticks with double backticks
    title = title.replace("`", "``")
    # Replace quadruple backticks with double backticks,
    # in case double backticks were already present
    title = title.replace("````", "``")

    # Print the PR line
    print(
        f"* {title} "
        f"(:pr:`{pr['number']}`)"
        + ", ".join([f":user:`{user}`" for user in extra_users])
        + (", " if extra_users else "")
        + f" :user:`{pr['user']['login']}`"
    )


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

    # Assign unmatched PRs to "Other" category
    assigned["Other"] = list(
        set(range(len(prs))) - {i for _, j in assigned.items() for i in j}
    )

    return assigned


def render_changelog(prs, assigned, label_to_subsection=None, module_order=None):
    """Render changelog with subsections based on module tags.

    Parameters
    ----------
    prs : list
        List of pull requests.
    assigned : dict
        Dictionary mapping category titles to list of PR indices.
    label_to_subsection : dict, optional
        Dictionary mapping module tags to subsection titles.
    module_order : list, optional
        List of subsection titles in the desired order.
    """
    from dateutil import parser

    SECTION_ORDER = ["Enhancements", "Documentation", "Maintenance", "Fixes", "Other"]
    for title in SECTION_ORDER:
        if title not in assigned:
            continue

        pr_group = [prs[i] for i in assigned[title]]
        if not pr_group:
            continue
        print()
        print(title)
        print("~" * len(title))

        # Group PRs by module labels
        def group_prs_by_module(pr_group: list[dict]) -> dict[str, list[dict]]:
            """Group PRs by module labels and return a dictionary."""
            subsection_map: dict[str, list[dict]] = defaultdict(list)

            for pr in pr_group:
                labels = [label["name"] for label in pr["labels"]]
                added = False

                for label in labels:
                    if label in LABEL_TO_SUBSECTION:
                        subsection_map[LABEL_TO_SUBSECTION[label]].append(pr)
                        added = True
                if not added:
                    subsection_map["Other"].append(pr)

            return subsection_map

        def render_subsection(subsection_map: dict[str, list[dict]]) -> None:
            """Render subsections and their PRs in order."""
            # Render subsections
            for subsection_title in MODULE_ORDER:
                pr_list = subsection_map.get(subsection_title, [])
                if pr_list is None or not pr_list:
                    continue
                print()
                print(subsection_title)
                print("^" * len(subsection_title))
                print()
                for pr in sorted(pr_list, key=lambda x: parser.parse(x["merged_at"])):
                    render_row(pr)

        if title in ["Enhancements", "Fixes"]:
            subsection_map = group_prs_by_module(pr_group)
            render_subsection(subsection_map)
        else:
            for pr in sorted(pr_group, key=lambda x: parser.parse(x["merged_at"])):
                render_row(pr)


def render_contributors(prs: list, fmt: str = "rst"):
    """Find unique authors and print a list in given format.

    Parameters
    ----------
    prs : list
        List of pull requests
    fmt : str, default="rst"
        Format of the output, either "github" or "rst"
    """
    authors = sorted({pr["user"]["login"] for pr in prs}, key=lambda x: x.lower())

    header = "Contributors"
    if fmt == "github":
        print(f"### {header}")
        print(", ".join(f"@{user}" for user in authors))
    elif fmt == "rst":
        print(header)
        print("~" * len(header), end="\n\n")
        print(",\n".join(f":user:`{user}`" for user in authors))


if __name__ == "__main__":
    # configuration of categories, sections, label mapping, and order
    # ---------------------------------------------------------------
    categories = [
        {"title": "Enhancements", "labels": ["feature", "enhancement"]},
        {"title": "Documentation", "labels": ["documentation"]},
        {"title": "Maintenance", "labels": ["maintenance", "chore"]},
        {"title": "Fixes", "labels": ["bug", "fix", "bugfix"]},
    ]

    # sourcery skip: use-named-expression
    LABEL_TO_SUBSECTION = {
        "module:base-framework": "BaseObject and base framework",
        "module:deep-learning&networks": "Other",
        "module:detection": "Time series anomalies, changepoints, segmentation",
        "module:distances&kernels": "Time series anomalies, changepoints, segmentation",
        "module:datasets&loaders": "Data sets and data loaders",
        "module:datatypes": "Datatypes, checks, conversions",
        "module:forecasting": "Forecasting",
        "module:metrics&benchmarking": "Benchmarking, Metrics, Splitters",
        "module:parameter-estimators": "Parameter estimation and hypothesis testing",
        "module:plotting&utilities": "Other",
        "module:probability&simulation": "Other",
        "module:splitters&resamplers": "Benchmarking, Metrics, Splitter",
        "module:classification": "Time series classification",
        "module:clustering": "Time series clustering",
        "module:regression": "Time series regression",
        "module:transformations": "Transformations",
        "module:tests": "Test framework",
    }
    # labels start with module
    # Subsection title comes after label

    MODULE_ORDER = [
        "BaseObject and base framework",
        "Benchmarking, Metrics, Splitters",
        "Data sets and data loaders",
        "Datatypes, checks, conversions",
        "Forecasting",
        "Parameter estimation and hypothesis testing",
        "Registry and search",
        "Time series alignment",
        "Time series anomalies, changepoints, segmentation",
        "Time series classification",
        "Time series clustering",
        "Time series regression",
        "Transformations",
        "Test framework",
        "Other",
    ]

    # end configuration

    pulls = fetch_pull_requests_since_last_release()
    print(f"Found {len(pulls)} merged PRs since last release")
    assigned = assign_prs(pulls, categories)
    render_changelog(
        pulls,
        assigned,
        label_to_subsection=LABEL_TO_SUBSECTION,
        module_order=MODULE_ORDER,
    )
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
