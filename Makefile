# Makefile for easier installation and cleanup.
#
# Uses self-documenting macros from here:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

PACKAGE=sktime
DOC_DIR=./docs
BUILD_TOOLS=./build_tools
TEST_DIR=testdir

.PHONY: help release install test lint clean dist doc docs

.DEFAULT_GOAL := help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

release: ## Make a release
	python3 $(BUILD_TOOLS)/make_release.py

install: ## Install for the current user using the default python command
	python3 setup.py build_ext --inplace && python setup.py install --user

test: ## Run unit tests
	-rm -rf ${TEST_DIR}
	mkdir -p ${TEST_DIR}
	cp .coveragerc ${TEST_DIR}
	cp setup.cfg ${TEST_DIR}
	cd ${TEST_DIR}; python -m pytest --cov-report html --cov=sktime -v -n 2 --showlocals --durations=20 --pyargs $(PACKAGE)

tests: test

clean: ## Clean build dist and egg directories left after install
	rm -rf ./dist
	rm -rf ./build
	rm -rf ./pytest_cache
	rm -rf ./htmlcov
	rm -rf ./junit
	rm -rf ./$(PACKAGE).egg-info
	rm -rf ./cover
	rm -rf coverage.xml
	rm -f MANIFEST
	rm -rf ./wheelhouse/*
	find . -type f -iname "*.so" -delete
	find . -type f -iname '*.pyc' -delete
	find . -type d -name '__pycache__' -empty -delete

dist: ## Make Python source distribution
	python3 setup.py sdist bdist_wheel

build:
	python -m build --sdist --wheel --outdir wheelhouse

docs: doc

doc: ## Build documentation with Sphinx
	$(MAKE) -C $(DOC_DIR) html

nb: clean
	rm -rf .venv || true
	python3 -m venv .venv
	. .venv/bin/activate && python -m pip install .[all_extras,binder] && ./build_tools/run_examples.sh
