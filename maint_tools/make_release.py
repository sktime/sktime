#!/usr/bin/env python3 -u
# coding: utf-8
<<<<<<< HEAD
=======
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
"""
Do-nothing script for making a release

This idea comes from here:
<<<<<<< HEAD
- https://blog.danslimmon.com/2019/07/15/do-nothing-scripting-the-key-to-gradual-automation/

The script is adapted from:
- https://github.com/alan-turing-institute/CleverCSV/blob/master/make_release.py
=======
- https://blog.danslimmon.com/2019/07/15/do-nothing-scripting-the-key-to
-gradual-automation/

The script is adapted from:
- https://github.com/alan-turing-institute/CleverCSV/blob/master
/make_release.py
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
"""

__author__ = ["Markus Löning"]

import codecs
import os
import re
import webbrowser
<<<<<<< HEAD

import colorama

ROOTDIR = os.path.abspath(os.path.dirname(__file__)).strip("maint_tools")
PACKAGE_NAME = "sktime"
URLS = {
    "docs_local": f"file:///{ROOTDIR}docs/_build/html/index.html",
    "docs_online": "https://alan-turing-institute.github.io/sktime/",
    "pypi": f"https://pypi.org/simple/{PACKAGE_NAME}/"
=======
import platform

import colorama

ROOT_DIR = os.path.abspath(os.path.dirname(__file__)).strip("maint_tools")
PACKAGE_NAME = "sktime"
URLS = {
    "docs_local": f"file:///{ROOT_DIR}docs/_build/html/index.html",
    "docs_online": "https://alan-turing-institute.github.io/sktime/",
    "pypi": f"https://pypi.org/simple/{PACKAGE_NAME}/",
    "github_new_pr": "https://github.com/alan-turing-institute/sktime/compare"
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
}


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
<<<<<<< HEAD
    with codecs.open(os.path.join(ROOTDIR, *parts), 'r') as fp:
=======
    with codecs.open(os.path.join(ROOT_DIR, *parts), 'r') as fp:
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
<<<<<<< HEAD
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
=======
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
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
    print(colored(msg, color=color, style=style))


def wait_for_enter():
    input(colored("\nPress Enter to continue", style="dim"))
    print()


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
<<<<<<< HEAD
    def action(self, context):
        self.instruct("Make sure you're on master and changes are merged in")
        self.print_run("git checkout master")
=======

    def __init__(self, branch):
        self.branch = branch

    def action(self, context):
        self.instruct(f"Make sure you're on: {self.branch}, you're local "
                      f"branch is up-to-date, and all new changes are merged "
                      f"in.")
        self.do_cmd(f"git checkout {self.branch}")
        self.do_cmd("git pull")
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed


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
        self.instruct(f"Update __init__.py with new version")

    def post(self, context):
        wait_for_enter()
<<<<<<< HEAD
        context["version"] = find_version(context['package_name'], '__init__.py')
=======
        context["version"] = find_version(context['package_name'],
                                          '__init__.py')
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed


class MakeClean(Step):
    def action(self, context):
        self.do_cmd("make clean")


class MakeDocs(Step):
    def action(self, context):
        self.do_cmd("make docs")


class MakeDist(Step):
    def action(self, context):
<<<<<<< HEAD
=======

        if platform.system() == "Darwin":
            self.instruct(
                "On macOS, we need to set up the C compiler.")
            self.do_cmd(
                'export CC=/usr/bin/clang '
                'export CXX=/usr/bin/clang++ '
                'export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp" '
                'export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include" '
                'export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include" '
                'export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp" '
                'export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib'
            )

>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        self.do_cmd("make dist")


class PushToTestPyPI(Step):
    def action(self, context):
        self.do_cmd(
<<<<<<< HEAD
            "twine upload --repository-url https://test.pypi.org/legacy/ dist/*"
=======
            "twine upload --repository-url https://test.pypi.org/legacy/ "
            "dist/*"
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        )


class InstallFromTestPyPI(Step):
    def action(self, context):
        self.instruct(
            f"Check installation from TestPyPI"
        )
<<<<<<< HEAD
        self.print_run("cd /tmp/")
        self.print_cmd("rm -rf ./venv")
        self.print_cmd("virtualenv ./venv")
        self.print_cmd("source ./venv/bin/activate")
        self.print_cmd(
            "pip install --index-url https://test.pypi.org/simple/ "
            "--extra-index-url https://pypi.org/simple {context['package_name']}=={context['version']}"
=======
        self.print_run("makedir /tmp/")
        self.print_run("cd /tmp/")
        self.print_cmd("conda remove -n testenv --all -y")
        self.print_cmd("conda create -n testenv python=3.7")
        self.print_cmd("conda activate testenv")

        # use extra-index-url to install dependencies
        self.print_cmd(
            f"pip install --index-url https://test.pypi.org/simple/ "
            f"--extra-index-url https://pypi.org/simple "
            f"{context['package_name']}=={context['version']}"
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        )


class CheckVersionNumber(Step):
    def action(self, context):
        self.instruct(
<<<<<<< HEAD
            f"Ensure that the following command gives version: {context['version']}"
        )
        self.do_cmd(f"python -c 'import {context['package_name']}; print({context['package_name']}.__version__)'")
=======
            f"Ensure that the following command gives version: "
            f"{context['version']}"
        )
        self.do_cmd(
            f"python -c 'import {context['package_name']}; print("
            f"{context['package_name']}.__version__)'")
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed


class DeactivateVenv(Step):
    def action(self, context):
        self.print_run("deactivate")
        self.instruct("Go back to the project directory")


<<<<<<< HEAD
class GitTagVersion(Step):
    def action(self, context):
=======
class GitTagRelease(Step):
    def action(self, context):
        self.instruct("Tag version as a release")
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        self.do_cmd(f"git tag v{context['version']}")


class GitTagPreRelease(Step):
    def action(self, context):
        self.instruct("Tag version as a pre-release (increment as needed)")
        self.print_run(f"git tag v{context['version']}-rc.1")


<<<<<<< HEAD
class GitAddCommit(Step):
    def action(self, context):
        self.instruct("Add everything to git and commit")


class GitAddRelease(Step):
    def action(self, context):
        self.instruct("Add CHANGELOG & README to git")
        self.instruct(
            f"Commit with title: {context['package_name']} {context['version']}"
        )


class PushToPyPI(Step):
    def action(self, context):
        self.do_cmd("twine upload dist/*")


class PushToGitHub(Step):
    def action(self, context):
        self.do_cmd("git push -u --tags origin master")
=======
class PushToGitHub(Step):
    def action(self, context):
        self.instruct("Add and commit to git, then push to GitHub")
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed


class CheckCIStatus(Step):
    def action(self, context):
        self.instruct(
            "Wait for CI to complete and check status"
        )


class CheckOnlineDocs(Step):
    def action(self, context):
        self.instruct(
            "Check online docs"
        )
        open_website(URLS["docs_online"])


class CheckLocalDocs(Step):
    def action(self, context):
        self.instruct(
            "Check local docs"
        )
        open_website(URLS["docs_local"])


class CheckPyPIFiles(Step):
    def action(self, context):
<<<<<<< HEAD
        self.instruct(
            "Check PyPI files"
        )
        open_website(URLS["pypi"])


def main():
    colorama.init()
    steps = [
        ConfirmGitStatus(),
        MakeClean(),
        RunLinting(),
        RunTests(),
        MakeDocs(),
        CheckLocalDocs(),
        PushToGitHub(),  # trigger CI to run tests
        CheckCIStatus(),
        CheckOnlineDocs(),
        BumpVersion(),
        GitAddCommit(),
        GitTagPreRelease(),
        PushToGitHub(),  # trigger CI to run tests
        CheckCIStatus(),
        UpdateChangelog(),
        MakeClean(),
        MakeDocs(),
        MakeDist(),
        PushToTestPyPI(),
        InstallFromTestPyPI(),
        CheckVersionNumber(),
        DeactivateVenv(),
        GitAddRelease(),
        PushToPyPI(),
        GitTagVersion(),
        PushToGitHub(),  # triggers Travis to build and deploy on tag
=======
        self.instruct("Check PyPI files")
        open_website(URLS["pypi"])


class OpenGitHubPR(Step):
    def action(self, context):
        self.instruct("Open PR from dev to master on GitHub")
        open_website(URLS["github_new_pr"])


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
        MakeDist(),
        PushToTestPyPI(),
        InstallFromTestPyPI(),
        DeactivateVenv(),
        # check pre-release online
        GitTagPreRelease(),
        PushTagToGitHub(),
        CheckCIStatus(),
        # make release
        GitTagRelease(),
        PushTagToGitHub(),
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
        CheckCIStatus(),
        CheckOnlineDocs(),
        CheckPyPIFiles()
    ]
    context = dict()
    context["package_name"] = PACKAGE_NAME
<<<<<<< HEAD
=======
    context["version"] = find_version(context["package_name"], "__init__.py")
>>>>>>> 67c56be8b1e838f2628df829946f795b7dba9aed
    for step in steps:
        step.run(context)
    cprint("\nDone!", color="yellow", style="bright")


if __name__ == "__main__":
    main()
