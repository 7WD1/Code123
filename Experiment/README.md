# Simulation Experiments for OTBE

This folder contains the reproducible simulation suite used by the paper.

Run:

```powershell
python .\run_experiments.py
```

The script creates synthetic black-box explanation tasks with known feature
contributions, evaluates LIME-like, KernelSHAP-like, PEXP-like, RealExp-like,
HBS-Shapley-like, TIDE-like, and OTBE explainers, and writes CSV summaries,
LaTeX tables, figures, and a manifest under `results/` and `figures/`.

All experiments are simulation-only. No external datasets are required.
