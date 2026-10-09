"""Metrics: PR-AUC, precision@k, macro-F1, ECE, run aggregation and the within-system noise test."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def minority_label(y: np.ndarray) -> int:
    """The rarer class: 1 (controversial) if fewer than half the articles are YES, else 0."""
    return 1 if y.mean() <= 0.5 else 0


def _oriented(y: np.ndarray, s: np.ndarray, positive: int) -> tuple[np.ndarray, np.ndarray]:
    if positive == 1:
        return (y == 1), s
    return (y == 0), 1.0 - s


def pr_auc(y: np.ndarray, s: np.ndarray, positive: int) -> float:
    """Average precision when `positive` is the class of interest (ties handled by sklearn)."""
    target, score = _oriented(y, s, positive)
    if target.sum() == 0:
        return float("nan")
    return float(average_precision_score(target.astype(int), score))


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int, positive: int) -> float:
    """Expected precision in the top k, averaging over random tie-breaking at the cut."""
    target, score = _oriented(y, s, positive)
    score = np.round(score, 12)
    k = min(k, len(score))
    order = np.argsort(-score, kind="stable")
    score, target = score[order], target[order]
    cut = score[k - 1]
    above, tied = score > cut, score == cut
    slots = k - int(above.sum())
    tp = target[above].sum() + slots * target[tied].sum() / tied.sum()
    return float(tp / k)


def macro_f1(y: np.ndarray, s: np.ndarray, threshold: float) -> float:
    """Macro-F1 of the rule 'score >= threshold means controversial'."""
    return float(f1_score(y, (s >= threshold).astype(int), average="macro", zero_division=0))


def best_macro_f1(y: np.ndarray, s: np.ndarray) -> float:
    """Macro-F1 at the best threshold on this very data (optimistic, reported with a star)."""
    candidates = np.unique(np.round(s, 12))
    return max(macro_f1(y, s, t) for t in candidates)


def ece(y: np.ndarray, s: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error, reading the score as P(controversial)."""
    idx = np.clip((s * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - s[mask].mean())
    return float(total)


def evaluate(y: np.ndarray, s: np.ndarray, ks: list[int], bins: int) -> dict[str, float]:
    """All metrics for one score vector. Keys ending in _min use the minority class as positive."""
    pos = minority_label(y)
    out = {"prauc_min": pr_auc(y, s, pos)}
    for k in ks:
        out[f"p@{k}_min"] = precision_at_k(y, s, k, pos)
    out["prauc_yes"] = pr_auc(y, s, 1)
    out["f1@0.5"] = macro_f1(y, s, 0.5)
    out["f1@best"] = best_macro_f1(y, s)
    out["ece"] = ece(y, s, bins)
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation, ignoring NaN; std is 0 for a single value."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def decompose(table: dict[tuple[int, int], float]) -> dict[str, float]:
    """Split run-to-run spread into a paraphrase part and a seed part.

    prompt = std over paraphrases of the seed-averaged metric, seed = std over seeds of the
    paraphrase-averaged metric, total = std over all runs.
    """
    paraphrases = sorted({p for p, _ in table})
    seeds = sorted({s for _, s in table})
    grid = np.array([[table.get((p, s), np.nan) for s in seeds] for p in paraphrases])
    nan = float("nan")
    total = float(np.nanstd(grid, ddof=1)) if np.isfinite(grid).sum() > 1 else nan
    prompt = float(np.nanstd(np.nanmean(grid, axis=1), ddof=1)) if len(paraphrases) > 1 else nan
    seed = float(np.nanstd(np.nanmean(grid, axis=0), ddof=1)) if len(seeds) > 1 else nan
    return {"total": total, "prompt": prompt, "seed": seed}


def compare(mean_a: float, std_a: float, mean_b: float, std_b: float) -> str:
    """'+' if A beats B by more than the larger within-system std, '-' if B beats A, else '~'."""
    if np.isnan([mean_a, mean_b]).any():
        return "?"
    noise = np.nanmax([std_a, std_b, 0.0])
    if abs(mean_a - mean_b) <= noise:
        return "~"
    return "+" if mean_a > mean_b else "-"
