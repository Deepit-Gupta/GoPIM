# GOPIM report source

This directory is the complete, self-contained source package for the report.
It follows the IEEE conference template supplied with the assignment and uses
the standard `IEEEtran` class available on Overleaf.

## Files

- `main.tex`: report source.
- `references.bib`: BibTeX references.
- `results/`: CSV data copied from the verified project runs.
- `make_figures.py`: rebuilds the two vector figures from those CSV files.
- `figures/`: generated PDF figures used by the report.

## Build

On Overleaf, upload the ZIP and select `main.tex` as the main document. The
IEEE conference class is provided by the Overleaf LaTeX environment.

To regenerate figures with Python and Matplotlib:

```powershell
python make_figures.py
```

To compile locally, install a LaTeX distribution with `IEEEtran` and run:

```powershell
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Replace the author and affiliation placeholders in `main.tex` before
submission. Check your course page limit and instructions before submitting.
The report explicitly labels the results as reduced-scale model results and
does not claim full NeuroSim integration or a complete numerical reproduction.
