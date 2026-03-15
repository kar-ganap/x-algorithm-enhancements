.PHONY: pdf clean wordcount md

pdf: main.tex references.bib
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

quick: main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.toc main.fls main.fdb_latexmk

wordcount: main.tex
	@texcount -inc -total main.tex 2>/dev/null || echo "Install texcount: brew install texcount"

md: main.tex
	pandoc main.tex -o main.md --wrap=none --reference-links
