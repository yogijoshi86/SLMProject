all: main.pdf

main.pdf: main.tex references.bib figures
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

figures: ab_results.png architecture.png clustering_sweep.png

ab_results.png:
	python3 generate_figures.py

architecture.png:
	python3 generate_figures.py

clustering_sweep.png:
	python3 generate_figures.py

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc

.PHONY: all figures clean
