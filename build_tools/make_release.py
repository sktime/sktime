#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Do-nothing script for making a release.

This idea comes from here:
- https://blog.danslimmon.com/2019/07/15/do-nothing-scripting-the-key-to
-gradual-automation/

The script is adapted from:
- https://github.com/alan-turing-institute/CleverCSV/blob/master
/make_release.py
"""

__author__ = ["mloning"]

import codecs
import os
import re
import webbrowser

import colorama

ROOT_DIR = os.path.abspath(os.path.dirname(__file__)).replace("build_tools", "")
PACKAGE_NAME = "sktime"


class URLs:
    """Container class for URLs."""

    DOCS_LOCAL = "file://" + os.path.realpath(
        os.path.join(ROOT_DIR, "docs/_build/html/index.html")
    )
    DOCS_ONLINE = "https://www.sktime.net"
    PYPI = f"https://pypi.org/simple/{PACKAGE_NAME}/"


def read(*parts):
    """Read from parts."""
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(ROOT_DIR, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    """Find version string."""
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


def open_website(website):
    """Open website."""
    webbrowser.open(website, new=1)


def colored(msg, color=None, style=None):
    """Print message with colors."""
    colors = {
        "red": colorama.Fore.RED,
        "green": colorama.Fore.GREEN,
        "cyan": colorama.Fore.CYAN,
        "yellow": colorama.Fore.YELLOW,
        "magenta": colorama.Fore.MAGENTA,
        None: "",
    }
    styles = {
        "bright": colorama.Style.BRIGHT,
        "dim": colorama.Style.DIM,
        None: "",
    }
    pre = colors[color] + styles[style]
    post = colorama.Style.RESET_ALL
    return f"{pre}{msg}{post}"


def cprint(msg, color=None, style=None):
    """Coloured printing."""
    print(colored(msg, color=color, style=style))


def wait_for_enter():
    """Wait for Enter."""
    input(colored("\nPress Enter to continue", style="dim"))
    print()


class Step:
    """Abstraction for release step."""

    def pre(self, context):
        """Pre-step."""

    def post(self, context):
        """Post-step."""
        wait_for_enter()

    def run(self, context):
        """Run step."""
        try:
            self.pre(context)
            self.action(context)
            self.post(context)
        except KeyboardInterrupt:
            cprint("\nInterrupted.", color="red")
            raise SystemExit(1)

    @staticmethod
    def instruct(msg):
        """Instruction."""
        cprint(msg, color="green")

    def print_run(self, msg):
        """Print run step."""
        cprint("Run:", color="cyan", style="bright")
        self.print_cmd(msg)

    @staticmethod
    def print_cmd(msg):
        """Print cmd step."""
        cprint("\t" + msg, color="cyan", style="bright")

    @staticmethod
    def do_cmd(cmd):
        """Wait for confirmation."""
        cprint(f"Going to run: {cmd}", color="magenta", style="bright")
        wait_for_enter()
        os.system(cmd)

    def action(self, context):
        """Carry out action."""
        raise NotImplementedError("abstract method")


class ConfirmGitStatus(Step):
    """Confirm git status."""

    def __init__(self, branch):
        self.branch = branch

    def action(self, context):
        """Carry out action."""
        self.instruct(
            f"Make sure you're on: {self.branch}, you're local "
            f"branch is up-to-date, and all new changes are merged "
            f"in."
        )
        self.do_cmd(f"git checkout {self.branch}")
        self.do_cmd("git pull")


class UpdateChangelog(Step):
    """Update the changelog."""

    def action(self, context):
        """Carry out action."""
        self.instruct(f"Update CHANGELOG for version: {context['version']}")


class UpdateReadme(Step):
    """Update the readme."""

    def action(self, context):
        """Carry out action."""
        self.instruct(f"Update README for version: {context['version']}")


class UpdateVersion(Step):
    """Update sktime version."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Update __init__.py with new version")

    def post(self, context):
        """Post-action step."""
        wait_for_enter()
        context["version"] = find_version(context["package_name"], "__init__.py")


class MakeClean(Step):
    """Make clean."""

    def action(self, context):
        """Carry out action."""
        self.do_cmd("make clean")


class MakeDocs(Step):
    """Make docs."""

    def action(self, context):
        """Carry out action."""
        self.do_cmd("make docs")


class MakeDist(Step):
    """Make dist."""

    def action(self, context):
        """Carry out action."""
        self.do_cmd("make dist")


class UploadToTestPyPI(Step):
    """Upload to test pypi."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Upload to TestPyPI")
        cmd = "twine upload --repository-url https://test.pypi.org/legacy/ dist/*"
        self.do_cmd(cmd)


class InstallFromTestPyPI(Step):
    """Install from test pypi."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Check installation from TestPyPI")
        self.do_cmd(
            f"sh build_tools/check_install_from_test_pypi.sh {context['version']}"
        )


class CheckVersionNumber(Step):
    """Check version number."""

    def action(self, context):
        """Carry out action."""
        self.instruct(
            f"Ensure that the following command gives version: {context['version']}"
        )
        self.do_cmd(
            f"python -c 'import {context['package_name']}; print("
            f"{context['package_name']}.__version__)'"
        )


class GitTagRelease(Step):
    """Git tag release."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Tag version as a release")
        self.do_cmd(f"git tag v{context['version']}")


class GitTagPreRelease(Step):
    """Git tag prerelease."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Tag version as a pre-release (increment as needed)")
        self.print_run(f"git tag v{context['version']}-rc.1")


class PushToGitHub(Step):
    """Push to GitHub."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Add and commit to git, then push to GitHub")


class CheckCIStatus(Step):
    """Check CI status."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Wait for CI to complete and check status")


class CheckOnlineDocs(Step):
    """Check online docs."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Check online docs")
        open_website(URLs.DOCS_ONLINE)


class CheckLocalDocs(Step):
    """Check local docs."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Check local docs")
        open_website(URLs.DOCS_LOCAL)


class CheckPyPIFiles(Step):
    """Check pypi files."""

    def action(self, context):
        """Carry out action."""
        self.instruct("Check PyPI files")
        open_website(URLs.PYPI)


class PushTagToGitHub(Step):
    """Push tag to GitHub."""

    def action(self, context):
        """Carry out action."""
        self.do_cmd(f"git push origin v{context['version']}")


def main():
    """Run release, main script."""
    colorama.init()
    steps = [
        # prepare and run final checks
        ConfirmGitStatus(branch="main"),
        MakeClean(),
        UpdateVersion(),
        CheckVersionNumber(),
        UpdateReadme(),
        UpdateChangelog(),
        MakeDocs(),
        CheckLocalDocs(),
        MakeDist(),
        UploadToTestPyPI(),
        InstallFromTestPyPI(),
        PushToGitHub(),
        CheckCIStatus(),
        # check pre-release online
        # GitTagPreRelease(),
        # PushTagToGitHub(),
        # make release
        GitTagRelease(),
        PushTagToGitHub(),
        CheckCIStatus(),
        CheckOnlineDocs(),
        CheckPyPIFiles(),
    ]
    context = dict()
    context["package_name"] = PACKAGE_NAME
    context["version"] = find_version(context["package_name"], "__init__.py")
    context["testdir"] = "temp/"
    for step in steps:
        step.run(context)
    cprint("\nDone!", color="yellow", style="bright")


if __name__ == "__main__":
    main()
