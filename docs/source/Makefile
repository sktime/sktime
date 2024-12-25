# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXBUILD        = sphinx-build
SPHINXOPTS         =
SPHINXAUTOBUILD    = sphinx-autobuild
SPHINXAUTOOPTS     =
SOURCEDIR          = .
BUILDDIR           = ../build
SKTIMEDIR          = ../../sktime

.PHONY: help build autobuild

# "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	rm -rf $(BUILDDIR)
	@echo "Deleted directory $(BUILDDIR)."

# $(O) is meant as a shortcut for custom options.
# i.e to log stderr into a separate file:
# make build O="--no-color 2> build_warnings.log"
html:
#	./symlink_examples.sh
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# $(O) is meant as a shortcut for custom options.
autobuild:
#   use --pre-build COMMAND to call ./symlink_examples.sh
	$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" -d "$(BUILDDIR)/doctrees" $(SPHINXAUTOOPTS) $(O) --watch "$(SKTIMEDIR)" --re-ignore ".*\.json"
