# Building the lecture notes

Every `.tex` file in this repository is self-contained (the shared preamble in `preamble.tex` is inlined at the top of each file) and compiles with `pdflatex` from TeX Live 2023 or later. Compiled PDFs are committed next to their sources so readers do not need a TeX installation.

```
cd notes-build
make            # rebuild every PDF
make clean      # remove aux files
```

To compile a single document, run `pdflatex` twice inside its folder, for example:

```
cd ../week02-vae/notes
pdflatex week02-notes.tex && pdflatex week02-notes.tex
```

`preamble.tex` is the reference copy of the shared macros (`\E`, `\KL`, `\given`, theorem environments, TikZ libraries). If you change it, copy the new block into each `.tex` file.
