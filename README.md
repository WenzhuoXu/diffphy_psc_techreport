# Toward Physics-Faithful Video Generation

Technical report (draft): a depth-guided generator, a human-annotated flaw
benchmark, and an auditable multi-tool critic. The report uses the Adobe Research
technical-report template from the Arceus project
`69fad61ed0f1a0808bc0f18f`.

- `main.tex` — the report source
- `references.bib` — bibliography
- `adobe_research.cls` and `assets/` — Arceus technical-report template
- `diffphy-support.sty` — project-specific packages and status components
- `main.pdf` — compiled report

Build: `latexmk -pdf main.tex` (or
`pdflatex main; bibtex main; pdflatex main; pdflatex main`).

This is a working draft. Methods detail, comparative results, and figures are
in progress; status is marked in-document.
