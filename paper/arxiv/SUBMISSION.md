# arXiv Submission Notes

Paper source:

- `main.tex`

Recommended arXiv metadata:

- Title: `Aliasing-Aware RTNeural-Compatible WaveNet Modeling of Guitar Amplifier and Pedal Captures`
- DOI: `10.5281/zenodo.21109537`
- DOI URL: `https://doi.org/10.5281/zenodo.21109537`
- Primary category: `eess.AS` Audio and Speech Processing
- Cross-list candidates: `cs.SD` Sound, optionally `cs.LG` Machine Learning
- Comments: `Draft; source code and experiment notes available at https://github.com/shortwavlabs/rtneural-trainer`

Why `eess.AS` first:

- arXiv describes `eess.AS` as theory and methods for processing audio, speech,
  and language signals, including analysis, synthesis, enhancement,
  transformation, system evaluation, implementation aspects, and machine
  learning applied to audio.
- The closest external papers cited by this draft use `eess.AS` and `cs.SD`
  for neural amp/audio DSP work.

Source package checklist:

- Upload `main.tex` as the source file.
- Do not upload generated files such as `.aux`, `.log`, `.pdf`, `.toc`, `.out`,
  `.synctex.gz`, or local build folders.
- Keep compilation from the root of the uploaded source directory.
- This draft uses an inline `thebibliography`, so no `.bib` or `.bbl` file is
  required.
- The source intentionally does not use `\pdfoutput`; arXiv's current TeX help
  says PDFLaTeX is automatically recognized and `\pdfoutput` should not be used
  to change output format.
- There are no figures in the current source package, so no image conversion or
  figure-file upload is required.

Before submission:

1. Replace the author block in `main.tex` with final author names,
   affiliations, ORCID links if desired, and contact email if desired.
2. Decide whether the source-code URL should point to the public GitHub repo,
   a release tag, or the Zenodo DOI above.
3. Decide whether private capture audio should remain unavailable, be released
   as ancillary files, or be replaced by a public reproducibility dataset.
4. Compile locally with a TeX distribution before upload:

   ```bash
   cd paper/arxiv
   pdflatex main.tex
   pdflatex main.tex
   ```

5. Inspect the resulting PDF carefully during arXiv submission. arXiv requires
   the submitter to view the generated PDF before completing submission.
6. If the paper is meant to be a product/process report rather than an academic
   methods paper, keep the limitations section as-is. If it is meant to be a
   stronger scientific claim, add repeated-seed runs, matched ablations, and a
   controlled listening study before upload.

Current local verification:

- TeX source has been written with conservative packages: `article`, `amsmath`,
  `amssymb`, `booktabs`, `array`, `geometry`, `hyperref`, `url`, and
  `microtype`.
- Local compilation was not performed here because `pdflatex`, `bibtex`, and
  `latexmk` are not installed in this workspace.
