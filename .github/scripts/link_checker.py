"""
Script to check for broken HTTP/HTTPS links in Markdown files.

This is intended to be used in CI to validate that no dead links exist in documentation.
"""

import re
import sys
from pathlib import Path

import requests

# Accepted status codes
ACCEPTED = set(range(200, 205)) | {403, 405, 418, 429, 999}

# Files or domains to exclude
EXCLUDE_FILES = {"CONTRIBUTORS.md"}
EXCLUDE_DOMAINS = {"github.com"}


def extract_links(text):
    """
    Extract HTTP/HTTPS links from Markdown-style syntax in a string.

    Args:
        text (str): The text content to search for links.

    Returns
    -------
        list: A list of extracted URLs.
    """
    return re.findall(r"\[.*?\]\((https?://.*?)\)", text)


def is_excluded(url):
    """
    Check whether a URL should be excluded from validation.

    Args:
        url (str): The URL to check.

    Returns
    -------
        bool: True if the URL is in the exclude list, False otherwise.
    """
    for domain in EXCLUDE_DOMAINS:
        if domain in url:
            return True
    return False


def check_links_in_file(file_path):
    """
    Check all links in a Markdown file for validity.

    Args:
        file_path (str or Path): Path to the Markdown file.

    Returns
    -------
        list: A list of tuples (link, error_code or exception) for broken links.
    """
    errors = []
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    links = extract_links(content)
    for link in links:
        if is_excluded(link):
            continue
        try:
            response = requests.head(link, allow_redirects=True, timeout=10)
            if response.status_code not in ACCEPTED:
                errors.append((link, response.status_code))
        except Exception as e:
            errors.append((link, str(e)))
    return errors


def main():
    """
    Find and validate links in all Markdown files.

    Exits with code 1 if broken links are found, 0 otherwise.
    """
    print("Checking Markdown links...")
    root = Path(".")
    md_files = list(root.rglob("*.md"))
    all_errors = []

    for file in md_files:
        if file.name in EXCLUDE_FILES:
            continue
        errors = check_links_in_file(file)
        if errors:
            print(f"\nBroken or problematic links in {file}:")
            for url, code in errors:
                print(f"  {url} -> {code}")
            all_errors.extend(errors)

    if all_errors:
        print(f"\n❌ Found {len(all_errors)} broken/problematic link(s).")
        sys.exit(1)
    else:
        print("\n✅ All links are valid.")


if __name__ == "__main__":
    main()
