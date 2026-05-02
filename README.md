# NEET UG 2024 Score Distribution Analysis

This repository contains data, scripts, and figures for an exploratory statistical analysis of the NEET UG 2024 score distribution.

The analysis uses an attempt--correctness model for a negatively marked multiple-choice examination. A score is modeled as

\[
M = 5r - q,
\]

where \(q\) is the number of attempted questions and \(r\) is the number of correct answers.

## Repository structure

- `data/data.dat` — aggregate score-count data
- `data/fit_result.dat` — fitted total distribution
- `data/fit_components.dat` — fitted component distributions
- `data/fit_parameters.dat` — fitted model parameters
- `code/fit.py` — fitting script
- `code/fig1fig2.py` — script for generating Figs. 1 and 2
- `code/fig3.py` — script for generating Fig. 3
- `figures/fig1.pdf`, `figures/fig2.pdf`, `figures/fig3.pdf` — final figures

## Reproducing the analysis

To rerun the fit:

```bash
python3 code/fit.py

To regenerate the figures:

python3 code/fig1fig2.py
python3 code/fig3.py

Disclaimer

This is an exploratory statistical analysis of aggregate score-count data. It is not an official report and does not make claims 
about malpractice, paper leakage, administrative irregularity, or individual candidate behavior.
