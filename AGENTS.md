# Repository instructions

- Leave changes uncommitted unless explicitly asked to commit.
- Do not push, publish releases, or change GitHub settings unless explicitly requested.
- Prefer small, focused changes. Avoid long training runs and broad refactors unless requested.
- Preserve this project's identity as a 2024 course project. Distinguish historical results from later corrections and new evaluations.
- Never invent measurements, experiment settings, contributions, or reproduction claims. Record unresolved evidence explicitly.
- Keep externally reported accuracy separate from locally measured accuracy and device latency.
- Keep report sources, figures, and bibliography in report/. Put auxiliary files and temporary outputs in ignored build/.
- Keep the finished report at report/sys843-report-en.pdf and link it from README.md.
- After report changes, compile and visually inspect the PDF. Report missing dependencies or checks that could not be completed.
- After code changes, run focused checks appropriate to the change. Do not imply historical results were regenerated merely because the code was fixed.

## Report build

Run from the repository root, using pdfLaTeX unless the source requires another engine:

    mkdir -p report/build
    latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=build report/main.tex

Only after a successful build:

    cp report/build/main.pdf report/sys843-report-en.pdf

The output directory is relative to report/ because latexmk uses -cd.
Keep build/ ignored; keep the finished report PDF tracked.
If latexmk is unavailable, report that limitation rather than claiming a successful build.