# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build
INSTALLDIR    = ../../docs.dival

EXCLUDE = ../dival ../dival/examples \
          ../dival/datasets/lidc_idri_dival/create_file_list.py \
          ../dival/datasets/lidc_idri_dival/create_series_dict.py \
          ../dival/datasets/lidc_idri_dival/create_uid_list.py \
          ../dival/datasets/lidc_idri_dival/download_images.py \
          ../dival/datasets/lidc_idri_dival/download_images_single.py \
          ../dival/datasets/lidc_idri_dival/loop_over_ct_dirs.py \
          ../dival/datasets/lidc_idri_dival/tciaclient.py
EXCLUDE += $(shell git ls-files -z --others --exclude-standard ../dival | xargs -0)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

apidoc:
	@sphinx-apidoc -f -e -T -o "$(SOURCEDIR)" $(EXCLUDE)

install:
	@cp -r "$(BUILDDIR)/html/." "$(INSTALLDIR)"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)