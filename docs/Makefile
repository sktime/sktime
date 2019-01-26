# Minimal makefile for Sphinx documentation

# You can set these variables from the command line.
PREPROCESS    = sphinx-apidoc
AUTOGENDIR    = source/autogen
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
PROJECTDIR    = "../sktime/"
EXCLUDEPATH   = "../sktime/tests ../sktime/utils"

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	-rm -rf source/autogen
	@$(PREPROCESS) -eMT -o $(AUTOGENDIR) $(PROJECTDIR) "$(EXCLUDEPATH)"
	-rm -rf build
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
