# Code123: OTBE Simulation Paper

This repository contains the code and manuscript artifacts for:

**Orthogonal Transport Bootstrap Explanations for Stable Simulation-Grounded Model Interpretation**

The project proposes **OTBE**, a simulation-grounded explanation method that combines transported coalition sampling, covariance-aware residual correction, and bootstrap aggregation for stable model interpretation under feature dependence.

## Repository Structure

```text
.
|-- Experiment/
|   |-- run_experiments.py        # End-to-end simulation benchmark
|   |-- figures/                  # Generated paper figures
|   `-- results/                  # CSV metrics, LaTeX tables, manifest
|-- review/
|   `-- simulated_review_rounds.md
|-- main.tex                      # IEEE conference manuscript source
|-- main.pdf                      # Compiled manuscript
|-- references.bib                # 42 cited references
`-- README.md
```

## Quick Start

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Run the simulation benchmark:

```bash
cd Experiment
python run_experiments.py
```

The script regenerates:

- `Experiment/results/raw_instance_metrics.csv`
- `Experiment/results/summary_metrics.csv`
- `Experiment/results/stat_tests.csv`
- `Experiment/figures/main_metrics.pdf`
- `Experiment/figures/scenario_recovery.pdf`
- `Experiment/figures/runtime_stability.pdf`

## Main Evidence

The benchmark uses four synthetic black-box tasks with known feature-level contributions:

- `SparseAdd`
- `CorrBias`
- `Interact`
- `GroupNonlin`

Compared with LIME-like, KernelSHAP-like, PEXP-like, RealExp-like, HBS-Shapley-like, and TIDE-like baselines, OTBE achieves the strongest repeated-run stability in the generated simulation while remaining competitive on top-k recovery and surrogate fidelity.

## Paper

The compiled paper is available in this repository as `main.pdf`.

The abstract in `main.tex` includes the code link:

```text
https://github.com/7WD1/Code123
```

## Reproducibility

The experiment seed is fixed in `Experiment/run_experiments.py`.
The generated manifest is stored at:

```text
Experiment/results/experiment_manifest.json
```

## Author

Wen Dong Jiang  
Graduate Student Member, IEEE  
Department of Computer Science and Information Engineering  
Tamkang University, New Taipei City 25137, Taiwan  
Email: 812414018@o365.tku.edu.tw
