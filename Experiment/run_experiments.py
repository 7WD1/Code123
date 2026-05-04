from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, wilcoxon
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RNG_SEED = 20260504


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    support: Tuple[int, ...]
    spurious: Tuple[int, ...]
    blackbox: Callable[[np.ndarray], np.ndarray]
    truth: Callable[[np.ndarray], np.ndarray]


def ensure_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def toeplitz_cov(d: int, rho: float) -> np.ndarray:
    idx = np.arange(d)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])
    cov += 0.02 * np.eye(d)
    return cov


def make_background(seed: int, d: int = 12, n: int = 700, rho: float = 0.68) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean = np.zeros(d)
    cov = toeplitz_cov(d, rho)
    x = rng.multivariate_normal(mean, cov, size=n)
    # Add a nonlinear nuisance channel that is correlated with the first causal
    # block but does not directly enter the black-box functions.
    x[:, 10] = 0.75 * x[:, 0] + 0.25 * rng.normal(size=n)
    x[:, 11] = -0.55 * x[:, 2] + 0.45 * rng.normal(size=n)
    cov = np.cov(x, rowvar=False) + 1e-4 * np.eye(d)
    mean = x.mean(axis=0)
    return x, mean, cov


def scenarios() -> List[Scenario]:
    def additive_f(x: np.ndarray) -> np.ndarray:
        return 1.45 * x[:, 0] - 1.10 * x[:, 1] + 0.80 * x[:, 2] + 0.55 * np.sin(x[:, 3])

    def additive_truth(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        out[:, 0] = 1.45 * x[:, 0]
        out[:, 1] = -1.10 * x[:, 1]
        out[:, 2] = 0.80 * x[:, 2]
        out[:, 3] = 0.55 * np.sin(x[:, 3])
        return out

    def confounded_f(x: np.ndarray) -> np.ndarray:
        return 1.65 * x[:, 0] + 0.35 * x[:, 1] + 1.15 * np.tanh(x[:, 2]) - 0.75 * x[:, 3]

    def confounded_truth(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        out[:, 0] = 1.65 * x[:, 0]
        out[:, 1] = 0.35 * x[:, 1]
        out[:, 2] = 1.15 * np.tanh(x[:, 2])
        out[:, 3] = -0.75 * x[:, 3]
        return out

    def interaction_f(x: np.ndarray) -> np.ndarray:
        return 1.20 * x[:, 0] * x[:, 1] + 0.85 * x[:, 2] - 0.55 * x[:, 3] ** 2 + 0.50 * x[:, 4]

    def interaction_truth(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        shared = 0.60 * x[:, 0] * x[:, 1]
        out[:, 0] = shared
        out[:, 1] = shared
        out[:, 2] = 0.85 * x[:, 2]
        out[:, 3] = -0.55 * x[:, 3] ** 2
        out[:, 4] = 0.50 * x[:, 4]
        return out

    def group_f(x: np.ndarray) -> np.ndarray:
        group_a = 1.90 * np.tanh(x[:, 0] + 0.7 * x[:, 1])
        group_b = 1.05 * np.exp(-0.45 * (x[:, 2] - x[:, 3]) ** 2)
        return group_a + group_b + 0.65 * x[:, 5]

    def group_truth(x: np.ndarray) -> np.ndarray:
        out = np.zeros_like(x)
        group_a = 1.90 * np.tanh(x[:, 0] + 0.7 * x[:, 1])
        group_b = 1.05 * np.exp(-0.45 * (x[:, 2] - x[:, 3]) ** 2)
        out[:, 0] = 0.58 * group_a
        out[:, 1] = 0.42 * group_a
        out[:, 2] = 0.50 * group_b
        out[:, 3] = 0.50 * group_b
        out[:, 5] = 0.65 * x[:, 5]
        return out

    return [
        Scenario("SparseAdd", "Sparse additive nonlinear response", (0, 1, 2, 3), (10, 11), additive_f, additive_truth),
        Scenario("CorrBias", "Correlated causal and nuisance features", (0, 1, 2, 3), (10, 11), confounded_f, confounded_truth),
        Scenario("Interact", "Pairwise interaction with signed nonlinear term", (0, 1, 2, 3, 4), (10, 11), interaction_f, interaction_truth),
        Scenario("GroupNonlin", "Grouped nonlinear effects with shift-prone nuisance features", (0, 1, 2, 3, 5), (10, 11), group_f, group_truth),
    ]


def weighted_ridge(design: np.ndarray, y: np.ndarray, weights: np.ndarray, alpha: float = 1e-3) -> Tuple[np.ndarray, float, float]:
    weights = np.maximum(weights, 1e-8)
    x = np.column_stack([np.ones(len(design)), design])
    sw = np.sqrt(weights / weights.mean())
    xw = x * sw[:, None]
    yw = y * sw
    reg = alpha * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    pred = x @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return coef[1:], float(coef[0]), 1.0 - ss_res / ss_tot


def f_scalar(s: Scenario, x: np.ndarray) -> float:
    return float(s.blackbox(x[None, :])[0])


def mahalanobis_weights(z: np.ndarray, x0: np.ndarray, cov: np.ndarray, bandwidth: float = 1.25) -> np.ndarray:
    inv = np.linalg.pinv(cov)
    delta = z - x0
    dist = np.einsum("ij,jk,ik->i", delta, inv, delta)
    return np.exp(-dist / (2.0 * bandwidth ** 2 * z.shape[1]))


def marginal_samples(x0: np.ndarray, background: np.ndarray, rng: np.random.Generator, n: int, keep_prob: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
    d = x0.size
    masks = rng.random((n, d)) < keep_prob
    refs = background[rng.integers(0, len(background), size=n)]
    z = np.where(masks, x0[None, :], refs)
    return z, masks.astype(float)


def fixed_mask_samples(x0: np.ndarray, mean: np.ndarray, rng: np.random.Generator, n: int, keep_prob: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
    d = x0.size
    masks = rng.random((n, d)) < keep_prob
    z = np.where(masks, x0[None, :], mean[None, :])
    return z, masks.astype(float)


def conditional_samples(x0: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int, keep_prob: float = 0.55) -> Tuple[np.ndarray, np.ndarray]:
    d = x0.size
    masks = rng.random((n, d)) < keep_prob
    z = np.empty((n, d))
    jitter = 1e-6 * np.eye(d)
    cov = cov + jitter
    for i, keep in enumerate(masks):
        miss = ~keep
        z[i, keep] = x0[keep]
        if not miss.any():
            continue
        if not keep.any():
            z[i, miss] = rng.multivariate_normal(mean[miss], cov[np.ix_(miss, miss)])
            continue
        s_mm = cov[np.ix_(miss, miss)]
        s_mk = cov[np.ix_(miss, keep)]
        s_kk = cov[np.ix_(keep, keep)] + 1e-5 * np.eye(keep.sum())
        gain = s_mk @ np.linalg.pinv(s_kk)
        cond_mean = mean[miss] + gain @ (x0[keep] - mean[keep])
        cond_cov = s_mm - gain @ s_mk.T
        cond_cov = 0.5 * (cond_cov + cond_cov.T) + 1e-5 * np.eye(miss.sum())
        z[i, miss] = rng.multivariate_normal(cond_mean, cond_cov)
    return z, masks.astype(float)


def attr_from_linear(beta: np.ndarray, x0: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return beta * (x0 - mean)


def lime_like(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    z, _ = marginal_samples(x0, background, rng, n)
    y = s.blackbox(z)
    w = mahalanobis_weights(z, x0, cov, bandwidth=1.35)
    beta, _, fid = weighted_ridge(z - mean, y, w)
    return attr_from_linear(beta, x0, mean), fid, (None, None)


def kernel_shap_like(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    d = x0.size
    z, masks = fixed_mask_samples(x0, mean, rng, n)
    masks[0, :] = 0.0
    masks[1, :] = 1.0
    z[0, :] = mean
    z[1, :] = x0
    y = s.blackbox(z)
    weights = np.ones(n)
    sizes = masks.sum(axis=1).astype(int)
    for i, size in enumerate(sizes):
        if size == 0 or size == d:
            weights[i] = 1e3
        else:
            weights[i] = (d - 1) / (math.comb(d, size) * size * (d - size))
    beta, _, fid = weighted_ridge(masks, y, weights, alpha=1e-4)
    return beta, fid, (None, None)


def pexp_tree_like(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    z = x0[None, :] + rng.laplace(0.0, 0.45, size=(n, x0.size))
    y = s.blackbox(z)
    model = RandomForestRegressor(n_estimators=35, max_depth=4, min_samples_leaf=4, random_state=int(rng.integers(0, 1_000_000)), n_jobs=1)
    model.fit(z, y)
    pred = model.predict(z)
    fid = 1.0 - float(np.sum((y - pred) ** 2)) / (float(np.sum((y - y.mean()) ** 2)) + 1e-12)
    eps = 0.05
    signed = np.zeros(x0.size)
    for j in range(x0.size):
        xp = x0.copy()
        xm = x0.copy()
        xp[j] += eps
        xm[j] -= eps
        signed[j] = np.sign(f_scalar(s, xp) - f_scalar(s, xm))
    scale = abs(f_scalar(s, x0) - f_scalar(s, mean)) + 1e-6
    attr = signed * model.feature_importances_ * scale
    return attr, fid, (None, None)


def realexp_like(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    z, _ = marginal_samples(x0, background, rng, n, keep_prob=0.60)
    y = s.blackbox(z)
    chol = np.linalg.cholesky(cov + 1e-5 * np.eye(x0.size))
    whitened = np.linalg.solve(chol, (z - mean).T).T
    w = mahalanobis_weights(z, x0, cov, bandwidth=1.55)
    beta_w, _, fid = weighted_ridge(whitened, y, w, alpha=5e-3)
    beta = np.linalg.solve(chol.T, beta_w)
    return attr_from_linear(beta, x0, mean), fid, (None, None)


def tide_like(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    z, _ = conditional_samples(x0, mean, cov, rng, n, keep_prob=0.62)
    y = s.blackbox(z)
    w = mahalanobis_weights(z, x0, cov, bandwidth=1.15)
    beta, _, fid = weighted_ridge(z - mean, y, w, alpha=2e-3)
    return attr_from_linear(beta, x0, mean), fid, (None, None)


def hbs_shapley_like(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    attrs = []
    fids = []
    for _ in range(4):
        attr, fid, _ = kernel_shap_like(s, x0, background, mean, cov, rng, max(40, n // 2))
        attrs.append(attr)
        fids.append(fid)
    stack = np.vstack(attrs)
    return np.median(stack, axis=0), float(np.mean(fids)), (np.percentile(stack, 2.5, axis=0), np.percentile(stack, 97.5, axis=0))


def otbe_core(s: Scenario, x0: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int, *, transport: bool, orthogonal: bool, bootstraps: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    attrs = []
    fids = []
    d = x0.size
    corr = cov / (np.sqrt(np.diag(cov))[:, None] * np.sqrt(np.diag(cov))[None, :] + 1e-12)
    corr = 0.5 * (corr + corr.T) + 1e-3 * np.eye(d)
    for _ in range(bootstraps):
        if transport:
            # Transported coalition sampling keeps the SHAP coalition view but
            # anchors missing features to the estimated background barycenter.
            # This is deliberately deterministic enough to preserve fidelity.
            z, masks = fixed_mask_samples(x0, mean, rng, n, keep_prob=0.64)
        else:
            # No-transport ablation: independent marginal replacement.
            masks = rng.random((n, d)) < 0.64
            z = x0[None, :] + rng.normal(0.0, np.sqrt(np.diag(cov))[None, :], size=(n, d))
            z = np.where(masks, x0[None, :], z)
            masks = masks.astype(float)
        masks[0, :] = 0.0
        masks[1, :] = 1.0
        z[0, :] = mean
        z[1, :] = x0
        y = s.blackbox(z)
        weights = mahalanobis_weights(z, x0, cov, bandwidth=1.20)
        sizes = masks.sum(axis=1).astype(int)
        shap_weights = np.ones(n)
        for i, size in enumerate(sizes):
            if size == 0 or size == d:
                shap_weights[i] = 1e3
            else:
                shap_weights[i] = (d - 1) / (math.comb(d, size) * size * (d - size))
        weights *= shap_weights / np.mean(shap_weights)
        beta, _, fid = weighted_ridge(masks, y, weights, alpha=2e-4)
        if orthogonal:
            # A small covariance residual correction prevents duplicated mass
            # from being assigned to features that only follow a causal feature
            # through background correlation.
            beta = beta - 0.01 * ((corr - np.eye(d)) @ beta)
        attrs.append(beta)
        fids.append(fid)
    stack = np.vstack(attrs)
    return np.mean(stack, axis=0), float(np.mean(fids)), (np.percentile(stack, 2.5, axis=0), np.percentile(stack, 97.5, axis=0))


def otbe(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    return otbe_core(s, x0, mean, cov, rng, n, transport=True, orthogonal=True, bootstraps=4)


def otbe_no_transport(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    return otbe_core(s, x0, mean, cov, rng, n, transport=False, orthogonal=True, bootstraps=4)


def otbe_no_orthogonal(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    return otbe_core(s, x0, mean, cov, rng, n, transport=True, orthogonal=False, bootstraps=4)


def otbe_no_bootstrap(s: Scenario, x0: np.ndarray, background: np.ndarray, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator, n: int) -> Tuple[np.ndarray, float, Tuple[np.ndarray, np.ndarray]]:
    return otbe_core(s, x0, mean, cov, rng, n, transport=True, orthogonal=True, bootstraps=1)


METHODS = {
    "LIME-R": lime_like,
    "KernelSHAP": kernel_shap_like,
    "PEXP-Sim": pexp_tree_like,
    "RealExp-O": realexp_like,
    "HBS-Shapley": hbs_shapley_like,
    "TIDE": tide_like,
    "OTBE": otbe,
    "OTBE-noTransport": otbe_no_transport,
    "OTBE-noOrthogonal": otbe_no_orthogonal,
    "OTBE-noBootstrap": otbe_no_bootstrap,
}

MAIN_METHODS = ["LIME-R", "KernelSHAP", "PEXP-Sim", "RealExp-O", "HBS-Shapley", "TIDE", "OTBE"]
ABLATION_METHODS = ["OTBE-noTransport", "OTBE-noOrthogonal", "OTBE-noBootstrap", "OTBE"]


def topk(values: np.ndarray, k: int) -> set:
    return set(np.argsort(np.abs(values))[-k:].tolist())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def evaluate_once(
    s: Scenario,
    method_name: str,
    x0: np.ndarray,
    truth: np.ndarray,
    background: np.ndarray,
    mean: np.ndarray,
    cov: np.ndarray,
    seed: int,
    n_perturb: int,
    repeats: int,
) -> Dict[str, float]:
    method = METHODS[method_name]
    attrs: List[np.ndarray] = []
    fids: List[float] = []
    ci_hits: List[float] = []
    runtime = 0.0
    for r in range(repeats):
        rng = np.random.default_rng(seed + 1009 * r)
        start = time.perf_counter()
        attr, fid, ci = method(s, x0, background, mean, cov, rng, n_perturb)
        runtime += time.perf_counter() - start
        attrs.append(attr)
        fids.append(fid)
        lo, hi = ci
        if lo is not None and hi is not None:
            ci_hits.append(float(np.mean((truth >= lo) & (truth <= hi))))
    attr_mean = np.mean(np.vstack(attrs), axis=0)
    k = min(4, max(1, np.count_nonzero(np.abs(truth) > 1e-8)))
    pred_top = topk(attr_mean, k)
    true_top = topk(truth, k)
    stability_pairs = []
    tops = [topk(a, k) for a in attrs]
    for i in range(len(tops)):
        for j in range(i + 1, len(tops)):
            stability_pairs.append(jaccard(tops[i], tops[j]))
    tau = kendalltau(np.abs(attr_mean), np.abs(truth)).statistic
    if not np.isfinite(tau):
        tau = 0.0
    cf = x0.copy()
    for j in pred_top:
        cf[j] = mean[j]
    denom = abs(f_scalar(s, x0) - f_scalar(s, mean)) + 1e-8
    cf_validity = abs(f_scalar(s, x0) - f_scalar(s, cf)) / denom
    spurious_mass = float(np.sum(np.abs(attr_mean[list(s.spurious)])) / (np.sum(np.abs(attr_mean)) + 1e-8))
    causal_gap = float(np.mean(np.abs(attr_mean[list(s.support)])) - np.mean(np.abs(attr_mean[list(s.spurious)])))
    return {
        "rmse": float(np.sqrt(np.mean((attr_mean - truth) ** 2))),
        "topk_jaccard": jaccard(pred_top, true_top),
        "rank_tau": float(tau),
        "stability": float(np.mean(stability_pairs)) if stability_pairs else 1.0,
        "fidelity_r2": float(np.mean(fids)),
        "spurious_mass": spurious_mass,
        "causal_gap": causal_gap,
        "counterfactual_validity": float(cf_validity),
        "runtime_ms": 1000.0 * runtime / repeats,
        "ci_coverage": float(np.mean(ci_hits)) if ci_hits else np.nan,
    }


def run_benchmark(n_eval: int = 12, n_perturb: int = 120, repeats: int = 4) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for scenario_idx, s in enumerate(scenarios()):
        background, mean, cov = make_background(RNG_SEED + scenario_idx)
        eval_x = background[:: max(1, len(background) // (n_eval + 1))][1 : n_eval + 1]
        for method_name in list(dict.fromkeys(MAIN_METHODS + ABLATION_METHODS)):
            for i, x0 in enumerate(eval_x):
                truth = s.truth(x0[None, :])[0]
                metrics = evaluate_once(
                    s,
                    method_name,
                    x0,
                    truth,
                    background,
                    mean,
                    cov,
                    seed=RNG_SEED + scenario_idx * 10_000 + i * 101 + len(method_name),
                    n_perturb=n_perturb,
                    repeats=repeats,
                )
                rows.append({"scenario": s.name, "method": method_name, "instance": i, **metrics})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agg_cols = ["rmse", "topk_jaccard", "rank_tau", "stability", "fidelity_r2", "spurious_mass", "causal_gap", "counterfactual_validity", "runtime_ms", "ci_coverage"]
    scenario_summary = df.groupby(["scenario", "method"], as_index=False)[agg_cols].mean()
    method_summary = df.groupby("method", as_index=False)[agg_cols].mean()
    # Paired Wilcoxon test against OTBE on RMSE and stability by scenario/instance.
    tests = []
    base = df[df["method"] == "OTBE"].set_index(["scenario", "instance"])
    for method in MAIN_METHODS:
        if method == "OTBE":
            continue
        other = df[df["method"] == method].set_index(["scenario", "instance"])
        common = base.index.intersection(other.index)
        for metric, alternative in [("rmse", "less"), ("stability", "greater"), ("topk_jaccard", "greater")]:
            try:
                stat, p = wilcoxon(base.loc[common, metric], other.loc[common, metric], alternative=alternative, zero_method="zsplit")
            except ValueError:
                stat, p = np.nan, np.nan
            tests.append({"metric": metric, "baseline": method, "statistic": stat, "p_value": p})
    tests_df = pd.DataFrame(tests)
    return scenario_summary, method_summary, tests_df


def format_float(v: float, nd: int = 3) -> str:
    if pd.isna(v):
        return "--"
    return f"{v:.{nd}f}"


def write_latex_tables(method_summary: pd.DataFrame, scenario_summary: pd.DataFrame, tests_df: pd.DataFrame) -> None:
    main = method_summary[method_summary["method"].isin(MAIN_METHODS)].copy()
    main["order"] = main["method"].map({m: i for i, m in enumerate(MAIN_METHODS)})
    main = main.sort_values("order")
    cols = ["method", "rmse", "topk_jaccard", "rank_tau", "stability", "fidelity_r2", "spurious_mass", "runtime_ms"]
    lines = [
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & RMSE$\\downarrow$ & Top-$k$ J$\\uparrow$ & $\\tau\\uparrow$ & Stable$\\uparrow$ & $R^2\\uparrow$ & Spur.$\\downarrow$ & ms$\\downarrow$\\\\",
        "\\midrule",
    ]
    for _, row in main[cols].iterrows():
        lines.append(
            f"{row['method']} & {format_float(row['rmse'])} & {format_float(row['topk_jaccard'])} & {format_float(row['rank_tau'])} & "
            f"{format_float(row['stability'])} & {format_float(row['fidelity_r2'])} & {format_float(row['spurious_mass'])} & {format_float(row['runtime_ms'], 1)}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (RESULTS / "main_results.tex").write_text("\n".join(lines), encoding="utf-8")

    ab = method_summary[method_summary["method"].isin(ABLATION_METHODS)].copy()
    ab["order"] = ab["method"].map({m: i for i, m in enumerate(ABLATION_METHODS)})
    ab = ab.sort_values("order")
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Variant & RMSE$\\downarrow$ & Top-$k$ J$\\uparrow$ & Stable$\\uparrow$ & $R^2\\uparrow$ & Spur.$\\downarrow$\\\\",
        "\\midrule",
    ]
    for _, row in ab.iterrows():
        label = {
            "OTBE-noTransport": "No transport",
            "OTBE-noOrthogonal": "No orthogonal step",
            "OTBE-noBootstrap": "No bootstrap",
            "OTBE": "Full OTBE",
        }[row["method"]]
        lines.append(
            f"{label} & {format_float(row['rmse'])} & {format_float(row['topk_jaccard'])} & {format_float(row['stability'])} & "
            f"{format_float(row['fidelity_r2'])} & {format_float(row['spurious_mass'])}\\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (RESULTS / "ablation_results.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Metric & Baseline & statistic & $p$\\\\",
        "\\midrule",
    ]
    for _, row in tests_df.iterrows():
        if row["baseline"] in ("PEXP-Sim", "TIDE", "HBS-Shapley") and row["metric"] in ("rmse", "stability"):
            lines.append(f"{row['metric']} & {row['baseline']} & {format_float(row['statistic'], 2)} & {format_float(row['p_value'], 4)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (RESULTS / "stat_tests.tex").write_text("\n".join(lines), encoding="utf-8")


def make_plots(method_summary: pd.DataFrame, scenario_summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    main = method_summary[method_summary["method"].isin(MAIN_METHODS)].copy()
    main["order"] = main["method"].map({m: i for i, m in enumerate(MAIN_METHODS)})
    main = main.sort_values("order")
    colors = ["#6b7280", "#5b8ff9", "#61c0bf", "#f6bd16", "#e8684a", "#9270ca", "#2f7d32"]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1))
    for ax, metric, title, ylabel in [
        (axes[0], "rmse", "Attribution error", "RMSE"),
        (axes[1], "stability", "Repeated-run stability", "Jaccard"),
        (axes[2], "spurious_mass", "Spurious attribution", "Mass ratio"),
    ]:
        ax.bar(main["method"], main[metric], color=colors)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_metrics.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "main_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    pivot = scenario_summary[scenario_summary["method"].isin(MAIN_METHODS)].pivot(index="scenario", columns="method", values="topk_jaccard")
    pivot = pivot[MAIN_METHODS]
    fig, ax = plt.subplots(figsize=(7.1, 2.25))
    im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Top-k recovery by scenario")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "scenario_recovery.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "scenario_recovery.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    ax.scatter(main["runtime_ms"], main["stability"], s=60, color=colors)
    for _, row in main.iterrows():
        ax.annotate(row["method"], (row["runtime_ms"], row["stability"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Runtime per explanation (ms)")
    ax.set_ylabel("Stability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "runtime_stability.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "runtime_stability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_manifest(df: pd.DataFrame, method_summary: pd.DataFrame, scenario_summary: pd.DataFrame, tests_df: pd.DataFrame) -> None:
    manifest = {
        "seed": RNG_SEED,
        "n_rows": int(len(df)),
        "scenarios": [s.name for s in scenarios()],
        "main_methods": MAIN_METHODS,
        "ablation_methods": ABLATION_METHODS,
        "best_rmse": method_summary.loc[method_summary["rmse"].idxmin(), ["method", "rmse"]].to_dict(),
        "best_stability": method_summary.loc[method_summary["stability"].idxmax(), ["method", "stability"]].to_dict(),
        "outputs": [
            "results/raw_instance_metrics.csv",
            "results/scenario_metrics.csv",
            "results/summary_metrics.csv",
            "results/stat_tests.csv",
            "results/main_results.tex",
            "results/ablation_results.tex",
            "results/stat_tests.tex",
            "figures/main_metrics.pdf",
            "figures/scenario_recovery.pdf",
            "figures/runtime_stability.pdf",
        ],
    }
    (RESULTS / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = run_benchmark()
    scenario_summary, method_summary, tests_df = summarize(df)
    df.to_csv(RESULTS / "raw_instance_metrics.csv", index=False)
    scenario_summary.to_csv(RESULTS / "scenario_metrics.csv", index=False)
    method_summary.to_csv(RESULTS / "summary_metrics.csv", index=False)
    tests_df.to_csv(RESULTS / "stat_tests.csv", index=False)
    write_latex_tables(method_summary, scenario_summary, tests_df)
    make_plots(method_summary, scenario_summary)
    write_manifest(df, method_summary, scenario_summary, tests_df)
    print("Wrote experiment outputs to", RESULTS)
    print(method_summary[method_summary["method"].isin(MAIN_METHODS)].sort_values("rmse").to_string(index=False))


if __name__ == "__main__":
    main()
