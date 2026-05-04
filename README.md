<div align="center">

# Code123

### Orthogonal Transport Bootstrap Explanations for Stable Simulation-Grounded Model Interpretation

[![IEEEtran](https://img.shields.io/badge/format-IEEEtran%20conference-1f6feb?style=for-the-badge)](main.tex)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776ab?style=for-the-badge)](Experiment/run_experiments.py)
[![Reproducible](https://img.shields.io/badge/results-reproducible-238636?style=for-the-badge)](Experiment/results/experiment_manifest.json)
[![Paper](https://img.shields.io/badge/paper-main.pdf-bc4c00?style=for-the-badge)](main.pdf)

**OTBE** is a simulation-grounded explanation framework for stable model interpretation under feature dependence.

[Read the paper](main.pdf) · [Run the benchmark](Experiment/run_experiments.py) · [View results](Experiment/results) · [View figures](Experiment/figures)

</div>

---

## Overview

This repository contains the manuscript, experiment code, generated figures, and reproducibility artifacts for **Orthogonal Transport Bootstrap Explanations (OTBE)**.

OTBE combines three ideas:

- Transported coalition sampling for dependence-aware perturbations
- Covariance-aware residual correction for reducing duplicated attribution mass
- Bootstrap aggregation for stable repeated-run explanations

The benchmark evaluates OTBE against LIME-like, KernelSHAP-like, PEXP-like, RealExp-like, HBS-Shapley-like, and TIDE-like baselines on synthetic black-box tasks with known feature-level contributions.

## Repository Highlights

| Area | Description |
| --- | --- |
| `main.pdf` | Compiled IEEE-style manuscript |
| `main.tex` | LaTeX source using `IEEEtran` conference formatting |
| `references.bib` | Bibliography with 42 cited references |
| `Experiment/run_experiments.py` | End-to-end simulation, metrics, tables, and figure generation |
| `Experiment/results/` | Raw metrics, summaries, statistical tests, and manifest |
| `Experiment/figures/` | Publication-ready PDF and PNG figures |

## Main Evidence

| Benchmark component | Included |
| --- | --- |
| Synthetic tasks | `SparseAdd`, `CorrBias`, `Interact`, `GroupNonlin` |
| Metrics | RMSE, top-k recovery, stability, fidelity, spurious mass, runtime |
| Reproducibility | Fixed seed and generated manifest |
| Paper assets | Tables and figures generated from the experiment outputs |

The generated results show that OTBE achieves the strongest repeated-run stability in this simulation while remaining competitive on top-k recovery and surrogate fidelity.

## Quick Start

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the benchmark:

```bash
cd Experiment
python run_experiments.py
```

The script regenerates the CSV metrics, LaTeX tables, figures, and manifest under `Experiment/results/` and `Experiment/figures/`.

## Project Layout

```text
.
|-- Experiment/
|   |-- run_experiments.py
|   |-- figures/
|   `-- results/
|-- review/
|   `-- simulated_review_rounds.md
|-- main.tex
|-- main.pdf
|-- references.bib
|-- requirements.txt
`-- README.md
```

## Paper

The compiled manuscript is available as [main.pdf](main.pdf). The abstract includes the public code link:

```text
https://github.com/7WD1/Code123
```

The paper source uses IEEE conference mode with Times-style scalable fonts:

```latex
\documentclass[10pt,conference,letterpaper]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
```

## Reproducibility

The experiment seed is fixed in `Experiment/run_experiments.py`, and the generated manifest is stored at:

```text
Experiment/results/experiment_manifest.json
```

To rebuild the paper after regenerating results:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Author

**Wen-Dong Jiang**<br>
Department of Computer Science and Information Engineering<br>
Tamkang University, New Taipei City 25137, Taiwan<br>
Email: 812414018@o365.tku.edu.tw

## Repository Maintainer

**7WD1**
