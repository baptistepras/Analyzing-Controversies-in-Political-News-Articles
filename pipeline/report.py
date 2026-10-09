"""Turn raw results into the readable logs: summary tables, per-article cards, CSV exports."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from scipy.stats import kendalltau
from sklearn.metrics import cohen_kappa_score

from .baselines import runs_file, scores_file

SYSTEM_ORDER = [
    "tfidf", "roberta_ft", "da_roberta_ft", "da_roberta_transfer",
    "llm_single", "audience", "audience_direct",
    "audience_deliberative", "ablation_homogeneous", "ablation_selfconsistency",
]
SYSTEM_NAMES = {
    "tfidf": "TF-IDF + LR (CV)",
    "roberta_ft": "RoBERTa-base FT (CV)",
    "da_roberta_ft": "DA-RoBERTa FT (CV)",
    "da_roberta_transfer": "DA-RoBERTa BABE, zero-shot",
    "llm_single": "LLM single agent",
    "audience": "Audience, 6 personas (entropy)",
    "audience_direct": "Audience + aggregator (direct)",
    "audience_deliberative": "Audience, deliberative",
    "ablation_homogeneous": "Ablation: 1 persona x 6 draws",
    "ablation_selfconsistency": "Ablation: no persona x 6 draws",
}
BASELINES = {"tfidf", "roberta_ft", "da_roberta_ft", "da_roberta_transfer"}
ABLATION_IDS = {"audience_deliberative", "ablation_homogeneous", "ablation_selfconsistency"}


@dataclass
class SystemRuns:
    system: str
    backbone: str
    regime: str = ""
    runs: dict[tuple[int, int], pd.Series] = field(default_factory=dict)
    meta: dict[tuple[int, int], pd.DataFrame] = field(default_factory=dict)

    @property
    def label(self) -> str:
        name = SYSTEM_NAMES.get(self.system, self.system)
        return f"{name} [{self.backbone}]" if self.backbone else name

    @property
    def card_label(self) -> str:
        """Label that also says which labels a supervised baseline was trained on."""
        return f"{self.label} (trained on {self.regime})" if self.regime else self.label


def safe_mean(values) -> float:
    """Mean ignoring NaN; NaN when nothing is left."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def md_table(header: list[str], rows: list[list]) -> str:
    """Render a markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def fmt(mean: float | None, std: float | None = None, digits: int = 3) -> str:
    """Format 'mean ± std', or n/a."""
    if mean is None or np.isnan(mean):
        return "n/a"
    if std is None or np.isnan(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def load_llm(full_dir: Path, backbones: list[str]) -> tuple[list[SystemRuns], dict[str, pd.DataFrame]]:
    """Read results_<backbone>.jsonl files into per-system run tables."""
    systems, raw = [], {}
    for backbone in backbones:
        path = full_dir / f"results_{backbone}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        frame = pd.DataFrame(rows).drop_duplicates(["system", "paraphrase", "seed", "item_id"], keep="last")
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        raw[backbone] = frame
        for system, group in frame.groupby("system"):
            sr = SystemRuns(system, backbone)
            for (p, s), sub in group.groupby(["paraphrase", "seed"]):
                indexed = sub.set_index("item_id")
                sr.runs[(int(p), int(s))] = indexed["score"]
                sr.meta[(int(p), int(s))] = pd.DataFrame({
                    "calls": indexed["n_calls"], "seconds": indexed["latency_s"], "fails": indexed["n_parse_fail"],
                })
            systems.append(sr)
    return systems, raw


def load_baselines(full_dir: Path, regime: str) -> list[SystemRuns]:
    """Read the baseline score files of one label regime into per-system run tables."""
    scores_path, runs_path = full_dir / scores_file(regime), full_dir / runs_file(regime)
    if not scores_path.exists():
        return []
    scores, runs = pd.read_csv(scores_path), pd.read_csv(runs_path)
    out = []
    for system, group in scores.groupby("system"):
        sr = SystemRuns(system, "", regime)
        for run_id, sub in group.groupby("run"):
            indexed = sub.set_index("item_id")
            seconds = runs[(runs["system"] == system) & (runs["run"] == run_id)]["seconds"].iloc[0]
            sr.runs[(0, int(run_id))] = indexed["score"]
            sr.meta[(0, int(run_id))] = pd.DataFrame({
                "calls": np.nan, "seconds": seconds / len(indexed), "fails": 0}, index=indexed.index)
        out.append(sr)
    return out


def evaluate_runs(sr: SystemRuns, y_by_item: pd.Series, items: list[str], ks: list[int], bins: int) -> list[dict]:
    """Metrics for every complete run of a system on the given articles."""
    y = y_by_item.reindex(items).to_numpy(int)
    rows = []
    for run_id, series in sr.runs.items():
        if not set(items) <= set(series.index):
            continue
        s = series.reindex(items).to_numpy(float)
        nan_rate = float(np.isnan(s).mean())
        row = M.evaluate(y, np.where(np.isnan(s), 0.5, s), ks, bins)
        meta = sr.meta[run_id].reindex(items)
        calls = meta["calls"].sum()
        row.update({
            "run": run_id, "nan_rate": nan_rate,
            "calls_per_article": meta["calls"].mean(),
            "sec_per_100": meta["seconds"].mean() * 100,
            "parse_fail": (meta["fails"].sum() / calls) if calls and not np.isnan(calls) else float("nan"),
        })
        rows.append(row)
    return rows


def ensemble_metrics(sr: SystemRuns, y_by_item: pd.Series, items: list[str], ks: list[int], bins: int) -> dict | None:
    """Metrics of the per-article mean score over all complete runs."""
    stack = [ser.reindex(items).to_numpy(float) for ser in sr.runs.values() if set(items) <= set(ser.index)]
    if not stack:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = np.nanmean(np.vstack(stack), axis=0)
    return M.evaluate(y_by_item.reindex(items).to_numpy(int), np.where(np.isnan(s), 0.5, s), ks, bins)


def summarize(rows: list[dict], keys: list[str]) -> dict[str, tuple[float, float]]:
    """Mean and std over runs for each metric key."""
    return {k: M.mean_std([r[k] for r in rows]) for k in keys}


def sort_key(sr: SystemRuns) -> tuple:
    idx = SYSTEM_ORDER.index(sr.system) if sr.system in SYSTEM_ORDER else 99
    return idx, sr.backbone


def metric_columns(ks: list[int]) -> list[tuple[str, str]]:
    cols = [("prauc_min", "PR-AUC (minority)")]
    cols += [(f"p@{k}_min", f"P@{k} (minority)") for k in ks]
    cols += [("prauc_yes", "PR-AUC (YES)"), ("f1@0.5", "F1 macro @0.5"), ("f1@best", "F1 macro @best*"), ("ece", "ECE")]
    return cols


def results_table(entries: list[tuple[SystemRuns, list[dict]]], ks: list[int], with_cost: bool) -> str:
    """Main metrics table, one row per system and backbone."""
    cols = metric_columns(ks)
    header = ["System", "Runs"] + [c[1] for c in cols]
    if with_cost:
        header += ["Calls/article", "Sec/100 articles", "Unreadable answers", "No score"]
    rows = []
    for sr, run_rows in entries:
        summ = summarize(run_rows, [c[0] for c in cols])
        row = [sr.label, len(run_rows)] + [fmt(*summ[c[0]]) for c in cols]
        if with_cost:
            calls = safe_mean(r["calls_per_article"] for r in run_rows)
            secs = safe_mean(r["sec_per_100"] for r in run_rows)
            fail = safe_mean(r["parse_fail"] for r in run_rows) if sr.backbone else float("nan")
            missing = safe_mean(r["nan_rate"] for r in run_rows)
            row += ["n/a" if np.isnan(calls) else f"{calls:.1f}", f"{secs:.0f}",
                    "n/a" if np.isnan(fail) else f"{100 * fail:.1f}%", f"{100 * missing:.1f}%"]
        rows.append(row)
    return md_table(header, rows)


def stance_profile(raw: dict[str, pd.DataFrame]) -> tuple[str, str]:
    """Persona stance tables per backbone, and how often two backbones agree on a persona's stance."""
    stats: dict[str, dict[str, list]] = {}
    per_item: dict[str, dict[str, pd.Series]] = {}
    for backbone, frame in raw.items():
        aud = frame[frame["system"] == "audience"]
        if aud.empty:
            continue
        values: dict[str, list] = {}
        table: dict[str, dict[str, list]] = {}
        for _, r in aud.iterrows():
            for persona, stance in (r.get("stances") or {}).items():
                if stance is None:
                    continue
                values.setdefault(persona, []).append(stance)
                table.setdefault(persona, {}).setdefault(r["item_id"], []).append(stance)
        stats[backbone] = values
        per_item[backbone] = {p: pd.Series({i: np.mean(v) for i, v in d.items()}) for p, d in table.items()}
    if not stats:
        return "", ""
    personas = sorted({p for v in stats.values() for p in v})
    header = ["Persona"] + [f"{b}: mean stance (share -1 / 0 / +1)" for b in stats]
    rows = []
    for p in personas:
        row = [p]
        for b in stats:
            v = np.array(stats[b].get(p, []))
            row.append("n/a" if v.size == 0 else
                       f"{v.mean():+.2f} ({(v == -1).mean():.0%} / {(v == 0).mean():.0%} / {(v == 1).mean():.0%})")
        rows.append(row)
    agree = ""
    if len(stats) >= 2:
        a, b = list(stats)[:2]
        rows2 = []
        for p in personas:
            if p in per_item[a] and p in per_item[b]:
                joint = pd.concat([per_item[a][p].round(), per_item[b][p].round()], axis=1).dropna()
                rows2.append([p, f"{(joint.iloc[:, 0] == joint.iloc[:, 1]).mean():.0%}", len(joint)])
        agree = md_table([f"Persona ({a} vs {b})", "Same majority stance per article", "Articles"], rows2)
    return md_table(header, rows), agree


def prauc_by_run(rows: list[dict]) -> dict:
    """PR-AUC (minority class) of every run, keyed by run id."""
    return {r["run"]: r["prauc_min"] for r in rows}


def win_count(rows_a: list[dict], rows_b: list[dict]) -> tuple[int, int]:
    """In how many of the shared (paraphrase, seed) runs does system A beat system B on PR-AUC?"""
    a, b = prauc_by_run(rows_a), prauc_by_run(rows_b)
    common = [k for k in a if k in b and not np.isnan(a[k]) and not np.isnan(b[k])]
    return sum(1 for k in common if a[k] > b[k]), len(common)


def verdict_text(a: tuple[float, float], n_a: int, b: tuple[float, float], n_b: int) -> str:
    """Plain-words reading of two PR-AUC (mean, std) pairs, using the within-system noise."""
    if min(n_a, n_b) < 3:
        return "too few runs to measure the noise"
    sign = M.compare(*a, *b)
    return {"+": "better, beyond the noise", "-": "worse, beyond the noise",
            "~": "no difference beyond the noise", "?": "not computable"}[sign]


def evaluate_regime(name: str, df: pd.DataFrame, systems: list[SystemRuns], abl_ids: list[str],
                    cfg: dict) -> dict:
    """All metrics of every system against one label regime."""
    ks, bins = cfg["metrics"]["k_values"], cfg["metrics"]["ece_bins"]
    y_by_item = df.set_index("item_id")["y"]
    items = df["item_id"].tolist()
    abl_items = [i for i in abl_ids if i in y_by_item.index]
    own = [sr for sr in systems if sr.regime in ("", name)]
    main = [sr for sr in own if sr.system not in ABLATION_IDS]
    for_ablation = [sr for sr in own if sr.backbone and (sr.system in ABLATION_IDS or sr.system in ("audience", "llm_single"))]
    full = [(sr, rows) for sr in main if (rows := evaluate_runs(sr, y_by_item, items, ks, bins))]
    abl = [(sr, rows) for sr in for_ablation if abl_items and (rows := evaluate_runs(sr, y_by_item, abl_items, ks, bins))]
    y = df["y"].to_numpy()
    pos = M.minority_label(y)
    y_abl = y_by_item.reindex(abl_items).to_numpy(int) if abl_items else np.array([], dtype=int)
    return {"name": name, "df": df, "y_by_item": y_by_item, "items": items, "abl_items": abl_items,
            "full": full, "abl": abl, "ks": ks, "bins": bins, "pos": pos,
            "n": len(df), "n_yes": int(y.sum()), "n_no": int((1 - y).sum()),
            "n_min": int((y == pos).sum()), "prev_min": float((y == pos).mean()), "prev_yes": float(y.mean()),
            "n_min_abl": int((y_abl == pos).sum()) if y_abl.size else 0}


def minority_name(R: dict) -> str:
    return "YES (controversial)" if R["pos"] == 1 else "NO (not controversial)"


def prauc_summary(rows: list[dict]) -> tuple[float, float]:
    return M.mean_std([r["prauc_min"] for r in rows])


def leaderboard(R: dict) -> str:
    """Systems sorted by mean PR-AUC on the minority class, with a noise-aware tie marker."""
    entries = [(sr, rows, prauc_summary(rows)) for sr, rows in R["full"]]
    entries.sort(key=lambda e: -1 if np.isnan(e[2][0]) else e[2][0], reverse=True)
    if not entries:
        return "No system has complete runs yet."
    best = entries[0]
    ks = R["ks"]
    rows_out = []
    for rank, (sr, rows, (mean, std)) in enumerate(entries, 1):
        summ = summarize(rows, [f"p@{ks[0]}_min", f"p@{ks[-1]}_min", "f1@0.5"])
        tie = "best" if rank == 1 else (
            "tied with #1" if M.compare(mean, std, *best[2]) == "~" and len(rows) >= 3 else
            ("behind #1" if M.compare(mean, std, *best[2]) == "-" else "?"))
        rows_out.append([rank, sr.label, len(rows), fmt(mean, std), f"{mean - R['prev_min']:+.3f}",
                         fmt(*summ[f"p@{ks[0]}_min"]), fmt(*summ[f"p@{ks[-1]}_min"]), fmt(*summ["f1@0.5"]), tie])
    return md_table(["Rank", "System", "Runs", "PR-AUC (minority)", "Gain over random", f"P@{ks[0]}", f"P@{ks[-1]}",
                     "F1 macro @0.5", "Versus #1 (noise-aware)"], rows_out)


def quick_answers(R: dict, cfg: dict) -> list[str]:
    """Plain-words answers to the project questions for one label regime."""
    full = {(sr.system, sr.backbone): (sr, rows) for sr, rows in R["full"]}
    abl = {(sr.system, sr.backbone): (sr, rows) for sr, rows in R["abl"]}
    backbones = sorted({sr.backbone for sr, _ in R["full"] if sr.backbone})
    out = [f"### {R['name'].capitalize()} labels: {R['n']} articles ({R['n_yes']} YES, {R['n_no']} NO)", ""]
    if R["n_min"] < cfg["metrics"]["min_minority_for_trust"]:
        out += [f"> **Fragile:** only {R['n_min']} article{'s' if R['n_min'] != 1 else ''} in the minority class {minority_name(R)}. "
                f"PR-AUC and P@k rest on those few articles, so every ranking below can flip with one article. "
                f"A random scorer gets {R['prev_min']:.3f}.", ""]

    out += ["**1. Does the audience simulation beat a single agent?**", ""]
    lines = []
    for b in backbones:
        single = full.get(("llm_single", b))
        for system in ("audience", "audience_direct"):
            cur = full.get((system, b))
            if cur and single:
                a, s_ = prauc_summary(cur[1]), prauc_summary(single[1])
                wins, n = win_count(cur[1], single[1])
                lines.append(f"- {b}, {SYSTEM_NAMES[system]}: PR-AUC {fmt(*a)} against {fmt(*s_)} for the single agent. "
                             f"Verdict: {verdict_text(a, len(cur[1]), s_, len(single[1]))}. Better in {wins} of {n} runs.")
    out += (lines or ["- Not available yet."]) + [""]

    out += ["**2. How do the LLM systems compare with the supervised baselines?**", ""]
    llm = [(sr, rows) for sr, rows in R["full"] if sr.backbone]
    sup = [(sr, rows) for sr, rows in R["full"] if not sr.backbone]
    if llm and sup:
        best_l = max(llm, key=lambda e: np.nan_to_num(prauc_summary(e[1])[0], nan=-1))
        best_s = max(sup, key=lambda e: np.nan_to_num(prauc_summary(e[1])[0], nan=-1))
        a, b_ = prauc_summary(best_l[1]), prauc_summary(best_s[1])
        out += [f"- Best LLM system: {best_l[0].label}, PR-AUC {fmt(*a)}. Best supervised baseline: {best_s[0].label}, PR-AUC {fmt(*b_)}. "
                f"Verdict for the best LLM system: {verdict_text(a, len(best_l[1]), b_, len(best_s[1]))}."]
        zero = next((rows for sr, rows in sup if sr.system == "da_roberta_transfer"), None)
        if zero:
            out += [f"- DA-RoBERTa BABE used as is (no training on our labels): PR-AUC {fmt(*prauc_summary(zero))}."]
    else:
        out += ["- Needs both LLM runs and supervised baselines."]
    out += [""]

    out += ["**3. Are the results stable across paraphrases and seeds (RQ2)?**", ""]
    stab = []
    for sr, rows in R["full"]:
        if not sr.backbone:
            continue
        table = prauc_by_run(rows)
        dec = M.decompose(table)
        mean, _ = M.mean_std(list(table.values()))
        if np.isnan(dec["prompt"]) and np.isnan(dec["seed"]):
            main_source = "n/a"
        else:
            main_source = "paraphrase" if np.nan_to_num(dec["prompt"]) >= np.nan_to_num(dec["seed"]) else "seed"
        stab.append([sr.label, len(rows), fmt(mean), fmt(dec["total"]), fmt(dec["prompt"]), fmt(dec["seed"]), main_source])
    out += [md_table(["System", "Runs", "PR-AUC mean", "Std total", "Std paraphrase", "Std seed", "Larger source"], stab)
            if stab else "- Not available yet.", ""]

    out += ["**4. Does persona diversity matter (heterogeneous audience against 6 draws of one centrist)?**", ""]
    lines = []
    for b in backbones:
        het, hom = abl.get(("audience", b)), abl.get(("ablation_homogeneous", b))
        if het and hom:
            a, h = prauc_summary(het[1]), prauc_summary(hom[1])
            wins, n = win_count(het[1], hom[1])
            lines.append(f"- {b}: heterogeneous {fmt(*a)} against homogeneous {fmt(*h)}. "
                         f"Verdict: {verdict_text(a, len(het[1]), h, len(hom[1]))}. Heterogeneous is better in {wins} of {n} runs.")
    if lines:
        lines.append(f"- Computed on {len(R['abl_items'])} articles, of which {R['n_min_abl']} in the minority class.")
    out += (lines or ["- Not available yet (run `--stage ablations`)."]) + [""]

    out += ["**5. Do the two backbones agree?**", ""]
    aud = {sr.backbone: sr for sr, _ in R["full"] if sr.system == "audience"}
    if len(aud) >= 2:
        a_name, b_name = list(aud)[:2]
        ma, mb = pd.DataFrame(aud[a_name].runs).mean(axis=1), pd.DataFrame(aud[b_name].runs).mean(axis=1)
        rho = ma.corr(mb, method="spearman")
        out += [f"- Spearman correlation of the per-article audience score between {a_name} and {b_name}: {rho:.2f}."]
        for system in ("audience", "llm_single"):
            pair = [full.get((system, b)) for b in (a_name, b_name)]
            if all(pair):
                out += [f"- {SYSTEM_NAMES[system]}: PR-AUC {fmt(*prauc_summary(pair[0][1]))} on {a_name}, "
                        f"{fmt(*prauc_summary(pair[1][1]))} on {b_name}."]
    else:
        out += ["- Needs both backbones."]
    out += [""]
    return out


def regime_comparison(results: dict[str, dict]) -> list[str]:
    """RQ1: do the label regimes agree on the labels and on the ranking of the systems?"""
    names = list(results)
    out = ["### Do the label regimes agree? (RQ1)", ""]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ya, yb = results[a]["y_by_item"], results[b]["y_by_item"]
            common = sorted(set(ya.index) & set(yb.index))
            if len(common) < 5:
                continue
            va, vb = ya.reindex(common).to_numpy(int), yb.reindex(common).to_numpy(int)
            kappa = cohen_kappa_score(va, vb) if len(set(va)) > 1 or len(set(vb)) > 1 else float("nan")
            rows.append([f"{a} vs {b}", len(common), f"{(va == vb).mean():.0%}", "n/a" if np.isnan(kappa) else f"{kappa:.2f}"])
    if rows:
        out += ["Label agreement on the articles that both regimes label:", "",
                md_table(["Pair", "Articles", "Same label", "Cohen's kappa"], rows), ""]
    table: dict[str, dict[str, float]] = {}
    for name, R in results.items():
        for sr, run_rows in R["full"]:
            table.setdefault(sr.label, {})[name] = prauc_summary(run_rows)[0]
    systems = [s for s, v in table.items() if len(v) == len(names)]
    if systems:
        sort_by = names[0]
        systems.sort(key=lambda s: -np.nan_to_num(table[s][sort_by], nan=-1))
        rank = {n: {s: 1 + sorted(systems, key=lambda x: -np.nan_to_num(table[x][n], nan=-1)).index(s) for s in systems}
                for n in names}
        out += ["PR-AUC (minority) of each system under each regime, sorted by the first regime "
                "(only systems that have results under every regime):", "",
                md_table(["System"] + [f"{n} (rank)" for n in names],
                         [[s] + [f"{table[s][n]:.3f} (#{rank[n][s]})" for n in names] for s in systems]), ""]
        tau_rows = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                va, vb = [table[s][a] for s in systems], [table[s][b] for s in systems]
                if len(systems) >= 3 and not np.isnan(va + vb).any():
                    tau = kendalltau(va, vb)[0]
                    tau_rows.append([f"{a} vs {b}", len(systems), "n/a" if np.isnan(tau) else f"{tau:.2f}"])
        if tau_rows:
            out += ["Kendall's tau between the system rankings of two regimes (1 = same order, 0 = unrelated, -1 = reversed):", "",
                    md_table(["Pair", "Systems", "Kendall tau"], tau_rows), ""]
    else:
        out += ["No system has complete runs under every regime yet."]
    return out


def regime_detail(R: dict, cfg: dict, backbones: list[str]) -> list[str]:
    """Detailed tables for one label regime."""
    ks, bins = R["ks"], R["bins"]
    full, name = R["full"], R["name"]
    out = [f"## {name.capitalize()} labels in detail", "",
           f"Articles: {R['n']} ({R['n_yes']} YES, {R['n_no']} NO, {100 * R['prev_yes']:.0f}% YES). "
           f"Minority class: {minority_name(R)}. A random scorer gets PR-AUC and P@k equal to the minority share, "
           f"{R['prev_min']:.3f}, and PR-AUC (YES) of {R['prev_yes']:.3f}.", "",
           f"### Main results, all {R['n']} articles", "",
           results_table(full, ks, True) if full else "No system has complete runs yet.", ""]
    ens_rows = []
    for sr, _ in full:
        m = ensemble_metrics(sr, R["y_by_item"], R["items"], ks, bins)
        if m:
            ens_rows.append([sr.label] + [fmt(m[c[0]]) for c in metric_columns(ks)])
    out += ["### Ensemble view (per-article mean score over all runs)", "",
            md_table(["System"] + [c[1] for c in metric_columns(ks)], ens_rows), ""]

    if len(full) > 1:
        summ = [(sr, prauc_summary(rows)) for sr, rows in full]
        legend = [f"S{i + 1} = {sr.label}" for i, (sr, _) in enumerate(summ)]
        matrix = [[f"S{i + 1}"] + [("" if i == j else M.compare(a[1][0], a[1][1], b[1][0], b[1][1]))
                                  for j, b in enumerate(summ)] for i, a in enumerate(summ)]
        out += ["### Pairwise reading of PR-AUC (minority)", "",
                "'+' means the row system beats the column system by more than the larger within-system std, "
                "'-' the opposite, '~' means the gap is within the noise.", "",
                md_table([""] + [f"S{i + 1}" for i in range(len(summ))], matrix), "", "; ".join(legend), ""]

    if R["abl"] and any(sr.system in ABLATION_IDS for sr, _ in R["abl"]):
        out += [f"### Ablation on {len(R['abl_items'])} articles ({R['n_min_abl']} in the minority class)", "",
                results_table(R["abl"], ks, True), ""]
        verdicts = []
        abl = {(sr.system, sr.backbone): rows for sr, rows in R["abl"]}
        for backbone in backbones:
            ref = abl.get(("audience", backbone))
            if ref is None:
                continue
            for system, text in (("ablation_homogeneous", "heterogeneous against homogeneous personas"),
                                 ("ablation_selfconsistency", "personas against plain self-consistency sampling"),
                                 ("audience_deliberative", "with against without deliberation"),
                                 ("llm_single", "6 personas against a single agent")):
                other = abl.get((system, backbone))
                if other:
                    verdicts.append(f"- {backbone}, {text}: PR-AUC {fmt(*prauc_summary(ref))} against "
                                    f"{fmt(*prauc_summary(other))}, {verdict_text(prauc_summary(ref), len(ref), prauc_summary(other), len(other))}.")
        out += ["Verdicts (a difference larger than the larger std counts as real):", "", *verdicts, ""]
    return out


def article_cards(articles: pd.DataFrame, regimes: dict[str, pd.DataFrame], systems: list[SystemRuns],
                  raw: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """One readable markdown file per article."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {}
    for sr in systems:
        wide = pd.DataFrame({str(k): v for k, v in sr.runs.items()})
        frames[sr.card_label] = (sr, wide.mean(axis=1), wide.std(axis=1, ddof=1) if wide.shape[1] > 1 else wide.mean(axis=1) * 0, wide.shape[1])
    labeled = {name: df.set_index("item_id") for name, df in regimes.items()}
    for _, art in articles.iterrows():
        item = art["item_id"]
        lines = [f"# {item}: {art['title']}", ""]
        shown = False
        for name, df in labeled.items():
            if item not in df.index:
                continue
            row = df.loc[item]
            text = f"**{name.capitalize()} label: {'YES' if row['y'] else 'NO'}**"
            if "confidence" in df.columns and row.get("confidence"):
                text += f" (confidence: {row['confidence']})"
            lines += [text]
            if "reason" in df.columns and row.get("reason"):
                lines += [f"Reason ({name}): {row['reason']}"]
            lines += [""]
            shown = True
        if not shown:
            lines += ["**No label in the files passed.**", ""]
        lines += ["> " + art["lead_150w"], "", "## Scores (mean ± std over runs; higher = more controversial)", ""]
        score_rows = []
        for label, (sr, mean, std, n) in sorted(frames.items(), key=lambda kv: sort_key(kv[1][0])):
            if item in mean.index and not np.isnan(mean[item]):
                score_rows.append([label, fmt(mean[item], std[item] if n > 1 else None), n])
        lines += [md_table(["System", "Score", "Runs"], score_rows), ""]
        for backbone, frame in raw.items():
            sub = frame[frame["item_id"] == item].sort_values(["paraphrase", "seed"])
            aud = sub[sub["system"] == "audience"]
            if not aud.empty:
                lines += [f"## Personas, {backbone}", ""]
                rows = []
                personas = list(aud.iloc[0]["stances"].keys())
                for p in personas:
                    vals = [r["stances"].get(p) for _, r in aud.iterrows() if r["stances"].get(p) is not None]
                    example = next((r["reactions"].get(p) for _, r in aud.iterrows() if r["reactions"].get(p)), "")
                    dist = f"{vals.count(-1)} / {vals.count(0)} / {vals.count(1)}"
                    rows.append([p, f"{np.mean(vals):+.2f}" if vals else "n/a", dist, f"\"{example}\""])
                lines += [md_table(["Persona", "Mean stance", "Runs at -1 / 0 / +1", "Example reaction (first run)"], rows), ""]
            for system, title in (("audience_direct", "Aggregator"), ("llm_single", "Single agent")):
                part = sub[sub["system"] == system]
                if part.empty:
                    continue
                text_col = "summary" if system == "audience_direct" else "reason"
                example = next((str(v) for v in part.get(text_col, []) if isinstance(v, str) and v), "")
                lines += [f"**{title}, {backbone}:** score {fmt(part['score'].mean(), part['score'].std(ddof=1) if len(part) > 1 else None)}"
                          + (f". Example: \"{example}\"" if example else ""), ""]
        (out_dir / f"{item}.md").write_text("\n".join(lines), encoding="utf-8")


def build_report(cfg: dict, prof: dict, articles: pd.DataFrame, regimes: dict[str, pd.DataFrame], abl_ids: list[str],
                 full_dir: Path, readable_dir: Path, backbones: list[str], profile_name: str,
                 notes: list[str] | None = None) -> str:
    """Write summary.md, per-article cards and CSVs; return the summary text."""
    readable_dir.mkdir(parents=True, exist_ok=True)
    llm_systems, raw = load_llm(full_dir, backbones)
    baseline_systems = [sr for name in regimes for sr in load_baselines(full_dir, name)]
    systems = sorted(llm_systems + baseline_systems, key=sort_key)
    if not systems:
        return "No results found yet. Run the baselines or the LLM stages first."
    results = {name: evaluate_regime(name, df, systems, abl_ids, cfg) for name, df in regimes.items()}
    regime_text = ", ".join(f"{n} ({R['n']} articles)" for n, R in results.items())

    out = [f"# Results summary ({profile_name} profile)", "",
           f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Label regimes evaluated: {regime_text}. "
           f"The LLM systems score each article once per run; each regime only changes which labels the scores are compared with.", "",
           "Reading guide: PR-AUC and P@k use the minority class as positive (as in the protocol). "
           "'F1 macro @best*' picks the best threshold on the same data, so it is optimistic. "
           "ECE reads each score as P(controversial). Mean ± std are over runs "
           "(LLM: paraphrase x seed; supervised baselines: cross-validation repeats). "
           "'Beyond the noise' means the gap is larger than the larger within-system std over runs.", ""]
    for note in notes or []:
        out += [f"- Note: {note}"]
    if notes:
        out += [""]

    out += ["## A. Quick answers", ""]
    for R in results.values():
        out += quick_answers(R, cfg)
    if len(results) >= 2:
        out += regime_comparison(results)
    if "silver" in results:
        out += ["> Silver labels come from another LLM. An LLM system can agree more with an LLM annotator than with humans, "
                "so read the silver ranking next to the gold ranking, not instead of it.", ""]

    out += ["## B. Leaderboards (sorted by PR-AUC on the minority class)", ""]
    for name, R in results.items():
        out += [f"### {name.capitalize()} labels (minority class: {minority_name(R)}, random scorer: {R['prev_min']:.3f})", "",
                leaderboard(R), ""]

    out += ["# Details", ""]
    for R in results.values():
        out += regime_detail(R, cfg, backbones)

    cost_rows = []
    for backbone in backbones:
        path = full_dir / f"backbone_info_{backbone}.json"
        if path.exists():
            info = json.loads(path.read_text())
            mem, lat = info.get("memory_gb"), info.get("mean_latency_s")
            cost_rows.append([backbone, info.get("backend"), info.get("model"),
                              "n/a" if mem is None else f"{mem:.1f}", "n/a" if lat is None else f"{lat:.2f}"])
    for name in regimes:
        base_runs = full_dir / runs_file(name)
        if base_runs.exists():
            br = pd.read_csv(base_runs)
            for system, g in br.groupby("system"):
                gb = g["gpu_gb"].dropna()
                backend = "sklearn (CPU)" if system == "tfidf" else "torch (MPS or CPU)"
                cost_rows.append([f"{SYSTEM_NAMES.get(system, system)} ({name})", backend, "n/a",
                                  "n/a" if gb.empty else f"{gb.mean():.1f}", "n/a"])
    role_rows = []
    for backbone in backbones:
        path = full_dir / f"backbone_info_{backbone}.json"
        if path.exists():
            for role, st in sorted(json.loads(path.read_text()).get("role_stats", {}).items()):
                num = lambda v, d=1: "n/a" if v is None else f"{v:.{d}f}"
                role_rows.append([backbone, role, st["n"], num(st["latency_s"], 2), num(st["prompt_tokens"], 0),
                                  num(st["gen_tokens"], 0), f"{100 * st['retry_share']:.1f}%"])
    out += ["## Memory and speed", "",
            md_table(["Model", "Backend", "Checkpoint", "Peak memory (GB)", "Mean s per call"], cost_rows),
            "", "Unreadable answers = share of calls whose answer could not be parsed (they are retried up to twice). No score = share of articles that still had no usable score after the retries; those get the neutral value 0.5 in the metrics, so a high value means the row is not trustworthy. "
            "Sec/100 articles for LLM systems is the sum of call latencies (sequential-equivalent time). "
            "For supervised baselines it is the full cross-validation time (training included) divided by the number of articles.", ""]
    if role_rows:
        out += ["Latency by call type (baseline = single agent, persona = persona calls, aggregate = aggregator, "
                "homog = homogeneous ablation draws, delib = deliberation, selfcons = self-consistency draws):", "",
                md_table(["Backbone", "Call type", "Calls", "Mean s", "Mean prompt tokens", "Mean generated tokens", "Retried"], role_rows), ""]

    profile_tab, agree_tab = stance_profile(raw)
    if profile_tab:
        out += ["## Do the backbones read the personas the same way?", "", profile_tab, ""]
        if agree_tab:
            out += [agree_tab, ""]

    out += ["## What this run does not cover", "",
            "- No temporal split and no Horvitz-Thompson reweighting: metrics describe this sample, not the natural feed. "
            "Supervised baselines use cross-validation on the same articles, which can leak topic and favor them.",
            "- BABE, Reddit and DELPHI side experiments are not included.",
            "- Kendall's tau (RQ1) needs at least two regimes; with one regime it is not computed.", ""]

    text = "\n".join(out)
    (readable_dir / "summary.md").write_text(text, encoding="utf-8")

    metric_rows = []
    for name, R in results.items():
        for scope, entries in (("all_articles", R["full"]), ("ablation_subset", R["abl"])):
            for sr, rows in entries:
                for r in rows:
                    metric_rows.append({"regime": name, "scope": scope, "system": sr.system, "backbone": sr.backbone,
                                        "paraphrase": r["run"][0], "seed_or_cv": r["run"][1],
                                        **{k: v for k, v in r.items() if k != "run"}})
    pd.DataFrame(metric_rows).to_csv(readable_dir / "metrics_by_run.csv", index=False)
    long_rows = []
    for sr in systems:
        wide = pd.DataFrame({str(k): v for k, v in sr.runs.items()})
        for item in wide.index:
            long_rows.append({"item_id": item, "system": sr.system, "backbone": sr.backbone, "trained_on": sr.regime,
                              "mean": wide.loc[item].mean(), "std": wide.loc[item].std(ddof=1) if wide.shape[1] > 1 else 0.0,
                              "n_runs": wide.shape[1]})
    pd.DataFrame(long_rows).to_csv(readable_dir / "scores_by_article.csv", index=False)
    article_cards(articles, regimes, systems, raw, readable_dir / "articles")
    return text
