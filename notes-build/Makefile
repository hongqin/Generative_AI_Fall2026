# Compile every LaTeX lecture-notes file in the repository.
#   make        build all PDFs (two pdflatex passes each)
#   make clean  remove auxiliary files, keep PDFs
#   make distclean  also remove PDFs
TEX := $(shell cd .. && find . -name '*.tex' -not -path './notes-build/*' | sort)
PDF := $(TEX:.tex=.pdf)

all: $(addprefix ../,$(PDF))

../%.pdf: ../%.tex
	cd $(dir $<) && pdflatex -interaction=nonstopmode -halt-on-error $(notdir $<) >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error $(notdir $<) >/dev/null
	cd $(dir $<) && rm -f $(basename $(notdir $<)).aux $(basename $(notdir $<)).log $(basename $(notdir $<)).out $(basename $(notdir $<)).toc $(basename $(notdir $<)).fls $(basename $(notdir $<)).fdb_latexmk

clean:
	cd .. && find . \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.fls' -o -name '*.fdb_latexmk' \) -not -path './notes-build/*' -delete

distclean: clean
	cd .. && for t in $(TEX); do rm -f "$${t%.tex}.pdf"; done

.PHONY: all clean distclean
