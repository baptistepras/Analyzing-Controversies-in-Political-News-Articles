#!/usr/bin/env python3
"""Agreement statistics for the human annotators: Krippendorff's alpha, kappa, agreement, confidence."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from pipeline.data import LABELS

CONF_ORDER = ["low", "medium", "high"]


def krippendorff_alpha(units: list[list[int]]) -> float:
    """Krippendorff's alpha for nominal labels; each unit is the list of labels given to one article.

    Units with fewer than two labels are ignored. Missing ratings are handled by design.
    """
    units = [u for u in units if len(u) >= 2]
    n = sum(len(u) for u in units)
    if n < 2:
        return float("nan")
    values = sorted({v for u in units for v in u})
    idx = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for u in units:
        m = len(u)
        for a, b in combinations(range(m), 2):
            o[idx[u[a]], idx[u[b]]] += 1 / (m - 1)
            o[idx[u[b]], idx[u[a]]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    off = lambda mat: mat.sum() - np.trace(mat)
    expected = np.outer(n_c, n_c)
    if off(expected) == 0:
        return float("nan")  # everybody used the same label: alpha is undefined
    return float(1 - (n - 1) * off(o) / off(expected))


def bootstrap_alpha(units: list[list[int]], n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of alpha, resampling articles."""
    units = [u for u in units if len(u) >= 2]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample = [units[i] for i in rng.integers(0, len(units), len(units))]
        a = krippendorff_alpha(sample)
        if not np.isnan(a):
            draws.append(a)
    if len(draws) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet's AC1 for two raters and binary labels; stable when one label is very frequent."""
    po = float((a == b).mean())
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def fmt(x: float, d: int = 2) -> str:
    return "n/a" if x is None or np.isnan(x) else f"{x:.{d}f}"


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def read_annotator(path: str) -> pd.DataFrame:
    """One annotator file: item_id, label, and optionally confidence and reason; unlabeled rows are dropped."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["item_id"] = df["item_id"].str.strip()
    df["y"] = df["label"].str.strip().str.upper().map(LABELS)
    for col in ("confidence", "reason"):
        if col not in df.columns:
            df[col] = ""
    df["confidence"] = df["confidence"].str.strip().str.lower()
    return df.dropna(subset=["y"]).assign(y=lambda d: d["y"].astype(int)).drop_duplicates("item_id")


def main() -> None:
    p = argparse.ArgumentParser(description="Annotator agreement statistics (Krippendorff's alpha and more).")
    p.add_argument("files", nargs="+", help="annotated CSVs, one per annotator (the same files as merge_gold.py)")
    p.add_argument("--silver", default=None, help="optional silver file, compared with each annotator and with the majority")
    p.add_argument("--out", default="outputs/annotator_stats.md")
    p.add_argument("--anonymize", action="store_true", help="call the annotators A1, A2, ... in the output")
    args = p.parse_args()

    names = [f"A{i + 1}" if args.anonymize else Path(f).stem for i, f in enumerate(args.files)]
    frames = {n: read_annotator(f) for n, f in zip(names, args.files)}
    long = pd.concat([d.assign(annotator=n) for n, d in frames.items()], ignore_index=True)
    titles = long.drop_duplicates("item_id").set_index("item_id")["title"] if "title" in long.columns else pd.Series(dtype=str)

    out = ["# Annotator statistics", ""]
    per_item = long.groupby("item_id")["y"].agg(["count", "sum"])
    n_items = len(per_item)
    shared = per_item[per_item["count"] >= 2]
    out += [f"- Articles with at least one label: {n_items}",
            f"- Articles with at least two labels (used for agreement): {len(shared)}",
            f"- Labels per article: " + ", ".join(f"{k} label{'s' if k > 1 else ''}: {v} articles"
                                                   for k, v in per_item["count"].value_counts().sort_index().items()), ""]

    rows = []
    has_reasons = any((d["reason"].str.strip() != "").any() for d in frames.values())
    for n, d in frames.items():
        conf = d["confidence"].value_counts()
        row = [n, len(d), f"{100 * d['y'].mean():.0f}%", conf.get("low", 0), conf.get("medium", 0), conf.get("high", 0)]
        if has_reasons:
            row.append(int((d["reason"].str.strip() == "").sum()))
        rows.append(row)
    header = ["Annotator", "Articles", "YES share", "Low conf.", "Medium conf.", "High conf."] + (["Empty reasons"] if has_reasons else [])
    out += ["## Per annotator", "", md_table(header, rows), ""]

    units = [g["y"].tolist() for _, g in long.groupby("item_id")]
    alpha = krippendorff_alpha(units)
    lo, hi = bootstrap_alpha(units)
    n_yes = int(long["y"].sum())
    out += ["## Overall agreement", "",
            f"- **Krippendorff's alpha (nominal): {fmt(alpha, 3)}**, 95% bootstrap interval [{fmt(lo, 3)}, {fmt(hi, 3)}] (articles resampled).",
            f"- Share of YES among all labels: {100 * n_yes / len(long):.0f}%.", ""]
    unanimous = sum(1 for u in units if len(u) >= 2 and len(set(u)) == 1)
    out += [f"- Articles with all labels equal: {unanimous} of {len(shared)} ({100 * unanimous / max(1, len(shared)):.0f}%).", ""]
    if long["y"].mean() > 0.85 or long["y"].mean() < 0.15:
        out += ["> When almost all labels are the same class, alpha and kappa can be low even if the annotators agree on most articles "
                "(the prevalence paradox). Read them next to the raw agreement and Gwet's AC1 below.", ""]

    wide = long.pivot(index="item_id", columns="annotator", values="y")
    pair_rows = []
    for a, b in combinations(names, 2):
        both = wide[[a, b]].dropna()
        if len(both) < 5:
            continue
        va, vb = both[a].to_numpy(int), both[b].to_numpy(int)
        kappa = cohen_kappa_score(va, vb) if len(set(va)) > 1 or len(set(vb)) > 1 else float("nan")
        pair_rows.append([f"{a} vs {b}", len(both), f"{100 * (va == vb).mean():.0f}%", fmt(kappa), fmt(gwet_ac1(va, vb))])
    if pair_rows:
        out += ["## Pairs of annotators (at least 5 shared articles)", "",
                md_table(["Pair", "Shared articles", "Raw agreement", "Cohen's kappa", "Gwet's AC1"], pair_rows), ""]

    dis = []
    for item, g in long.groupby("item_id", sort=False):
        if len(g) >= 2 and g["y"].nunique() > 1:
            votes = "; ".join(f"{r.annotator}: {'YES' if r.y else 'NO'} ({r.confidence or '?'})" for r in g.itertuples())
            low = any(r.confidence == "low" for r in g.itertuples())
            dis.append([item, str(titles.get(item, ""))[:70], votes, "yes" if low else "no"])
    both_low = sum(1 for d in dis if d[3] == "yes")
    out += [f"## Disagreements: {len(dis)} articles", ""]
    if dis:
        out += [f"In {both_low} of them at least one annotator reported low confidence.", "",
                md_table(["Article", "Title", "Votes", "A low confidence?"], dis), ""]

    # majority vote and the confidence link
    maj = per_item.assign(share=lambda d: d["sum"] / d["count"])
    decided = maj[(maj["count"] >= 2) & (maj["share"] != 0.5)]
    maj_label = (decided["share"] > 0.5).astype(int)
    if args.silver:
        s = pd.read_csv(args.silver, dtype=str, keep_default_na=False)
        s["y"] = s["label"].str.strip().str.upper().map(LABELS)
        s = s.dropna(subset=["y"]).set_index("item_id")["y"].astype(int)
        rows = []
        for n, d in frames.items():
            common = sorted(set(d["item_id"]) & set(s.index))
            if len(common) >= 5:
                va, vb = d.set_index("item_id").loc[common, "y"].to_numpy(int), s.loc[common].to_numpy(int)
                k = cohen_kappa_score(va, vb) if len(set(va)) > 1 or len(set(vb)) > 1 else float("nan")
                rows.append([f"{n} vs silver", len(common), f"{100 * (va == vb).mean():.0f}%", fmt(k), fmt(gwet_ac1(va, vb))])
        common = sorted(set(maj_label.index) & set(s.index))
        if len(common) >= 5:
            va, vb = maj_label.loc[common].to_numpy(int), s.loc[common].to_numpy(int)
            k = cohen_kappa_score(va, vb) if len(set(va)) > 1 or len(set(vb)) > 1 else float("nan")
            rows.append(["human majority vs silver", len(common), f"{100 * (va == vb).mean():.0f}%", fmt(k), fmt(gwet_ac1(va, vb))])
        if rows:
            out += ["## Silver labels against the humans", "",
                    md_table(["Comparison", "Articles", "Raw agreement", "Cohen's kappa", "Gwet's AC1"], rows), ""]

    text = "\n".join(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
