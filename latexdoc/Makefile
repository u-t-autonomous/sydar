SOURCES = documentation
PDFFILE = documentation.pdf
STYLEFILES = ieeeconf.cls
BIBFILE = refs.bib
#--------------------------------------
LATEX   = latex
BIBTEX  = bibtex
DVIPDF  = dvipdf
TEXFILES = $(addsuffix .tex,$(SOURCES))
AUXFILES = $(addsuffix .aux,$(SOURCES))
LOGFILES = $(addsuffix .log,$(SOURCES))
BBLFILES = $(addsuffix .blg,$(SOURCES)) $(addsuffix .bbl,$(SOURCES))
DVIFILES = $(addsuffix .dvi,$(SOURCES))
#--------------------------------------
default: $(PDFFILE)

%.dvi: %.tex $(BIBFILE) $(TEXFILES) $(STYLEFILES)
	$(LATEX) $<
	$(BIBTEX) $*
	$(LATEX) $<
	$(LATEX) $<

%.pdf: %.dvi
	$(DVIPDF) $<

.PHONY: clean
clean:
	rm -f $(PDFFILE) $(DVIFILES)
	rm -f $(AUXFILES) $(LOGFILES) $(BBLFILES)
