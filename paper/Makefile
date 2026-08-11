PAPER = slopformer

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex memes.tex references.bib
	pdflatex -interaction=nonstopmode $(PAPER).tex
	bibtex $(PAPER)
	pdflatex -interaction=nonstopmode $(PAPER).tex
	pdflatex -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc

distclean: clean
	rm -f $(PAPER).pdf

.PHONY: all clean distclean
