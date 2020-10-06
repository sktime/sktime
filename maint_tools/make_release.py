#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Do-nothing script for making a release

This idea comes from here:
- https://blog.danslimmon.com/2019/07/15/do-nothing-scripting-the-key-to
-gradual-automation/

The script is adapted from:
- https://github.com/alan-turing-institute/CleverCSV/blob/master
/make_release.py
"""

__author__ = ["Markus Löning"]

import codecs
import os
import re
import webbrowser

import colorama

ROOT_DIR = os.path.abspath(os.path.dirname(__file__)).replace("maint_tools", "")
PACKAGE_NAME = "sktime"


class URLs:
    DOCS_LOCAL = "file://" + os.path.realpath(
        os.path.join(ROOT_DIR, "docs/_build/html/index.html")
    )
    DOCS_ONLINE = "https://www.sktime.org"
    PYPI = f"https://pypi.org/simple/{PACKAGE_NAME}/"
    GITHUB_NEW_PR = "https://github.com/alan-turing-institute/sktime/compare"


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(ROOT_DIR, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


def open_website(website):
    webbrowser.open(website, new=1)


def colored(msg, color=None, style=None):
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
    """Coloured printing"""
    print(colored(msg, color=color, style=style))  # noqa


def wait_for_enter():
    input(colored("\nPress Enter to continue", style="dim"))
    print()  # noqa


class Step:
    def pre(self, context):
        pass

    def post(self, context):
        wait_for_enter()

    def run(self, context):
        try:
            self.pre(context)
            self.action(context)
            self.post(context)
        except KeyboardInterrupt:
            cprint("\nInterrupted.", color="red")
            raise SystemExit(1)

    @staticmethod
    def instruct(msg):
        cprint(msg, color="green")

    def print_run(self, msg):
        cprint("Run:", color="cyan", style="bright")
        self.print_cmd(msg)

    @staticmethod
    def print_cmd(msg):
        cprint("\t" + msg, color="cyan", style="bright")

    @staticmethod
    def do_cmd(cmd):
        cprint(f"Going to run: {cmd}", color="magenta", style="bright")
        wait_for_enter()
        os.system(cmd)

    def action(self, context):
        raise NotImplementedError("abstract method")


class ConfirmGitStatus(Step):
    def __init__(self, branch):
        self.branch = branch

    def action(self, context):
        self.instruct(
            f"Make sure you're on: {self.branch}, you're local "
            f"branch is up-to-date, and all new changes are merged "
            f"in."
        )
        self.do_cmd(f"git checkout {self.branch}")
        self.do_cmd("git pull")


class RunTests(Step):
    def action(self, context):
        self.do_cmd("make test")


class RunLinting(Step):
    def action(self, context):
        self.do_cmd("make lint")


class UpdateChangelog(Step):
    def action(self, context):
        self.instruct(f"Update CHANGELOG for version: {context['version']}")


class BumpVersion(Step):
    def action(self, context):
        self.instruct("Update __init__.py with new version")

    def post(self, context):
        wait_for_enter()
        context["version"] = find_version(context["package_name"], "__init__.py")


class MakeClean(Step):
    def action(self, context):
        self.do_cmd("make clean")


class MakeDocs(Step):
    def action(self, context):
        self.do_cmd("make docs")


class MakeDist(Step):
    def action(self, context):
        self.do_cmd("make dist")


class PushToTestPyPI(Step):
    def action(self, context):
        self.instruct("Upload to TestPyPI")
        cmd = "twine upload --repository-url https://test.pypi.org/legacy/ " "dist/*"
        self.do_cmd(cmd)


class InstallFromTestPyPI(Step):
    def action(self, context):
        self.instruct("Check installation from TestPyPI")
        self.print_run(f"mkdir {context['testdir']}")
        self.print_run(f"cd {context['testdir']}")
        self.print_cmd("conda remove -n testenv --all -y")
        self.print_cmd("conda create -n testenv python=3.7")
        self.print_cmd("conda activate testenv")

        # use extra-index-url to install dependencies
        self.print_cmd(
            f"pip install --index-url https://test.pypi.org/simple/ "
            f"--extra-index-url https://pypi.org/simple "
            f"{context['package_name']}=={context['version']}"
        )


class CheckVersionNumber(Step):
    def action(self, context):
        self.instruct(
            f"Ensure that the following command gives version: "
            f""
            f"{context['version']}"
        )
        self.do_cmd(
            f"python -c 'import {context['package_name']}; print("
            f"{context['package_name']}.__version__)'"
        )


class DeactivateTestEnvironment(Step):
    def action(self, context):
        self.instruct("Deactivate and remove test environment.")
        self.print_run("conda deactivate")

        self.instruct("Go back to the project directory")
        self.print_cmd("cd ..")
        self.print_cmd(f"rm -r {context['testdir']}")


class GitTagRelease(Step):
    def action(self, context):
        self.instruct("Tag version as a release")
        self.do_cmd(f"git tag v{context['version']}")


class GitTagPreRelease(Step):
    def action(self, context):
        self.instruct("Tag version as a pre-release (increment as needed)")
        self.print_run(f"git tag v{context['version']}-rc.1")


class PushToGitHub(Step):
    def action(self, context):
        self.instruct("Add and commit to git, then push to GitHub")


class CheckCIStatus(Step):
    def action(self, context):
        self.instruct("Wait for CI to complete and check status")


class CheckOnlineDocs(Step):
    def action(self, context):
        self.instruct("Check online docs")
        open_website(URLs.DOCS_ONLINE)


class CheckLocalDocs(Step):
    def action(self, context):
        self.instruct("Check local docs")
        open_website(URLs.DOCS_LOCAL)


class CheckPyPIFiles(Step):
    def action(self, context):
        self.instruct("Check PyPI files")
        open_website(URLs.PYPI)


class OpenGitHubPR(Step):
    def action(self, context):
        self.instruct("Open PR from dev to master on GitHub")
        open_website(URLs.GITHUB_NEW_PR)


class MergeGitHubPR(Step):
    def action(self, context):
        self.instruct("Review and merge PR from dev into master on GitHub")


class PushTagToGitHub(Step):
    def action(self, context):
        self.do_cmd("git push -u --tags origin master")


def main():
    colorama.init()
    steps = [
        # ConfirmGitStatus(branch="dev"),
        # # run checks locally
        # MakeClean(),
        # RunLinting(),
        # RunTests(),
        # UpdateChangelog(),
        # MakeDocs(),
        # CheckLocalDocs(),
        # BumpVersion(),
        # CheckVersionNumber(),
        # # run CI checks online
        # PushToGitHub(),
        # OpenGitHubPR(),
        # CheckCIStatus(),
        # MergeGitHubPR(),
        # CheckCIStatus(),
        # CheckOnlineDocs(),
        # check TestPyPI locally
        ConfirmGitStatus(branch="master"),
        MakeClean(),
        BumpVersion(),
        CheckVersionNumber(),
        UpdateChangelog(),
        MakeDocs(),
        CheckLocalDocs(),
        MakeDist(),
        PushToTestPyPI(),
        InstallFromTestPyPI(),
        DeactivateTestEnvironment(),
        PushToGitHub(),
        # check pre-release online
        # GitTagPreRelease(),
        # PushTagToGitHub(),
        CheckCIStatus(),
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
