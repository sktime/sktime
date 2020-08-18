# Makefile for easier installation and cleanup.
#
# Uses self-documenting macros from here:
# http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html

PACKAGE=sktime
DOC_DIR='./docs'
MAINT_DIR='./maint_tools'

.PHONY: help release install test lint clean dist doc docs

.DEFAULT_GOAL := help

help:
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) |\
		 awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m\
		 %s\n", $$1, $$2}'

release: ## Make a release
	python $(MAINT_DIR)/make_release.py

install: ## Install for the current user using the default python command
	python setup.py build_ext --inplace && python setup.py install --user

test: ## Run unit tests
	pytest --cov-report html --cov=sktime --showlocals --durations=20 --pyargs $(PACKAGE)

tests: test

lint:  ## Run linting
	$(MAINT_DIR)/linting.sh

clean: ## Clean build dist and egg directories left after install
	rm -rf ./dist
	rm -rf ./build
	rm -rf ./pytest_cache
	rm -rf ./htmlcov
	rm -rf ./junit
	rm -rf ./$(PACKAGE).egg-info
	rm -rf ./cover
	rm -rf $(VENV_DIR)
	rm -f MANIFEST
	rm -f ./$(PACKAGE)/*.so
	find . -type f -iname '*.pyc' -delete
	find . -type d -name '__pycache__' -empty -delete

dist: ## Make Python source distribution
	python setup.py sdist bdist_wheel

docs: doc

doc: ## Build documentation with Sphinx
	rm -rf $(DOC_DIR)/source/contributors.rst && m2r CONTRIBUTORS.md && mv CONTRIBUTORS.rst $(DOC_DIR)/source/contributors.rst
	rm -rf $(DOC_DIR)/source/contributing.rst && m2r CONTRIBUTING.md && mv CONTRIBUTING.rst $(DOC_DIR)/source/contributing.rst
	$(MAKE) -C $(DOC_DIR) html

