# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
PREPROCESS    = sphinx-apidoc
APIDOCTEMPLATEDIR = source/_templates/apidoc
AUTOGENDIR    = source/api_reference/auto_generated
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
PROJECTDIR    = "../sktime/"

# exclude unit tests from API reference
EXCLUDEPATHS = "../*tests* ../*test* ../*contrib* ../*setup* ../*all*"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	./symlink_examples.sh
	rm -rf $(AUTOGENDIR)
	@$(PREPROCESS) -eMT --templatedir $(APIDOCTEMPLATEDIR) -o $(AUTOGENDIR) $(PROJECTDIR) "$(EXCLUDEPATHS)"
	-rm -rf $(BUILDDIR)
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
