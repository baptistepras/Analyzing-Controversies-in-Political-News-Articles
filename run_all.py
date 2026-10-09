#!/usr/bin/env python3
"""Run the whole evaluation: supervised baselines, LLM systems, ablation, and the report."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import yaml

from pipeline import baselines, data, prompts, report
from pipeline.backends import CallStore, make_backend, read_records, role_stats
from pipeline.llm_systems import (
    MAIN_SYSTEMS, Article, LLMContext, Run, load_done, resolve_ablations, run_llm_stage,
)

STAGES = ["baselines", "llm", "ablations", "report", "all"]
ABLATION_ROLES = {"audience_deliberative": "delib", "ablation_homogeneous": "homog", "ablation_selfconsistency": "selfcons"}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audience-simulation evaluation pipeline.")
    p.add_argument("--gold", default=None, help="gold label file (CSV: item_id, title, lead_150w, label)")
    p.add_argument("--silver", default=None, help="silver label file made by annotate_silver.py")
    p.add_argument("--distant", default=None, help="distant label file (CSV: item_id, label, and title and lead_150w if no other file has them)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--profile", default="smoke", help="smoke, check or final (see config.yaml)")
    p.add_argument("--stage", default="all", choices=STAGES,
                   help="all = baselines + llm + report; the ablation is NOT included, run --stage ablations for it")
    p.add_argument("--backbones", nargs="+", default=None, help="subset of the backbones in the config")
    p.add_argument("--baselines", nargs="+", default=None, help="override the list of enabled baselines")
    p.add_argument("--ablations", nargs="+", default=None, help="override the ablations to run (homogeneous, deliberative, selfconsistency)")
    p.add_argument("--max-articles", type=int, default=None, help="use only the first N articles (the cache is shared, so you can grow N later)")
    p.add_argument("--ablation-articles", default=None, help="number of articles in the ablation, or 'all'")
    p.add_argument("--paraphrases", type=int, default=None, help="override the number of paraphrases")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="override the seeds, e.g. --seeds 0 for a first pass")
    p.add_argument("--out", default="outputs")
    p.add_argument("--workers", type=int, default=None, help="parallel articles (Ollama only)")
    p.add_argument("--mock", action="store_true", help="fake models, for testing the plumbing only")
    p.add_argument("--estimate", action="store_true", help="print the number of calls and hours that remain, then stop")
    p.add_argument("--force-baselines", action="store_true")
    p.add_argument("--call-stats", action="store_true", help="print latency and token statistics per call type from the logs, then stop")
    return p.parse_args(argv)


def seconds_per_article_run(stats: dict, cfg: dict, assumed: dict, ablations: list[str]) -> tuple[dict[str, float], str]:
    """Seconds for one article in one run, for the main systems and each ablation; measured when possible."""
    n_p, n_d = len(prompts.PERSONAS), cfg["llm"]["n_draws"]
    groups = {"main": {"baseline": 1, "persona": n_p, "aggregate": 1}}
    for system in ablations:
        role = ABLATION_ROLES[system]
        groups[system] = {role: n_p if role == "delib" else n_d}
    out, source = {}, []
    for group, roles in groups.items():
        key = "main" if group == "main" else ABLATION_ROLES[group]
        if all(stats.get(role, {}).get("n", 0) >= 20 for role in roles):
            out[group] = sum(k * stats[role]["latency_s"] for role, k in roles.items())
            source.append(f"{key} measured")
        else:
            out[group] = assumed[key]
            source.append(f"{key} assumed")
    return out, ", ".join(source)


def remaining_article_runs(results_path: Path, items: list[str], runs: list[Run], systems: list[str]) -> int:
    """Number of (run, article) pairs for which at least one of the systems has no result yet."""
    done = load_done(results_path)
    return sum(1 for r in runs for i in items if any((s, r.paraphrase, r.seed, i) not in done for s in systems))


def print_estimate(cfg: dict, prof: dict, items: list[str], abl_items: list[str], names: list[str],
                   full_dir: Path, ablations: list[str]) -> None:
    """Hours that remain for the chosen profile, using logged latencies when they exist."""
    runs = [Run(p, s) for p in range(prof["paraphrases"]) for s in prof["seeds"]]
    rows, total = [], 0.0
    for name in names:
        bcfg = cfg["backbones"][name]
        stats = role_stats(read_records(full_dir / f"calls_{name}.jsonl"), bcfg["model"])
        secs, source = seconds_per_article_run(stats, cfg, bcfg["assumed_s_per_article_run"], ablations)
        results = full_dir / f"results_{name}.jsonl"
        workers = max(1, bcfg.get("workers", 1))
        n_main = remaining_article_runs(results, items, runs, MAIN_SYSTEMS)
        hours = n_main * secs["main"] / workers / 3600
        row = [name, f"{n_main:,} of {len(runs) * len(items):,}", f"{secs['main']:.0f} s", f"{hours:.1f} h"]
        abl_hours = 0.0
        for system in ablations:
            n_abl = remaining_article_runs(results, abl_items, runs, [system])
            abl_hours += n_abl * secs[system] / workers / 3600
        row += [f"{abl_hours:.1f} h", f"{hours + abl_hours:.1f} h", source]
        total += hours + abl_hours
        rows.append(row)
    print(f"\nProfile: {len(items)} articles, {len(runs)} runs ({prof['paraphrases']} paraphrases x {len(prof['seeds'])} seeds), "
          f"ablation {', '.join(ablations)} on {len(abl_items)} articles.")
    print(report.md_table(["Backbone", "Main article-runs left", "s per main", "Main time left",
                           "Ablation time left", "Total left", "Source of the seconds"], rows))
    print(f"\nTotal left for the LLM stages: about {total:.1f} h, one backbone after the other. "
          "An article-run is one article for one (paraphrase, seed) run: main = 8 calls, homogeneous ablation = 6 calls. "
          "Retries add a few percent. The silver annotation (annotate_silver.py --estimate) and the supervised baselines come on top.\n")


def print_call_stats(cfg: dict, names: list[str], full_dir: Path) -> None:
    """Latency and token statistics per call type, read from the call logs."""
    for name in names:
        bcfg = cfg["backbones"][name]
        stats = role_stats(read_records(full_dir / f"calls_{name}.jsonl"), bcfg["model"])
        if not stats:
            print(f"{name}: no logged calls in {full_dir}")
            continue
        fmt = lambda v, d: "n/a" if v is None else f"{v:.{d}f}"
        rows = [[role, st["n"], fmt(st["latency_s"], 2), fmt(st["prompt_tokens"], 0), fmt(st["gen_tokens"], 0),
                 f"{100 * st['retry_share']:.1f}%"] for role, st in sorted(stats.items())]
        print(f"\n{name} ({bcfg['model']})")
        print(report.md_table(["Call type", "Calls", "Mean s", "Mean prompt tokens", "Mean generated tokens", "Retried"], rows))


def warn_if_synced_folder() -> None:
    """Warn when the working folder sits in iCloud or the Trash, where files can be moved or deleted mid-run."""
    here = str(Path.cwd())
    if any(marker in here for marker in ("Mobile Documents", "CloudStorage", ".Trash")):
        print("\n*** WARNING: this folder is in iCloud or in the Trash:\n"
              f"    {here}\n"
              "    A long run can lose its logs if the folder is synced, moved or deleted. "
              "Use a plain local folder such as ~/dev/audience_pipeline. ***\n")


def run_llm(cfg: dict, prof: dict, articles, abl_ids: list[str], names: list[str], stage: str, full_dir: Path,
            mock: bool, workers_override: int | None, ablations: list[str]) -> None:
    """Run the LLM systems for each backbone in turn; everything is cached and resumable."""
    runs = [Run(p, s) for p in range(prof["paraphrases"]) for s in prof["seeds"]]
    all_articles = [Article(r.item_id, r.title, r.lead_150w) for r in articles.itertuples()]
    keep = set(abl_ids)
    abl_articles = [a for a in all_articles if a.item_id in keep]
    for name in names:
        bcfg = cfg["backbones"][name]
        backend = make_backend(name, bcfg, mock)
        backend.preflight()
        store = CallStore(full_dir / f"calls_{name}.jsonl")
        ctx = LLMContext(name, backend, store, cfg["llm"])
        workers = bcfg.get("workers", 1)
        if workers_override and (mock or bcfg["backend"] == "ollama"):
            workers = workers_override
        if bcfg["backend"] == "mlx" and not mock:
            workers = 1  # MLX is not safe to call from several threads
        results = full_dir / f"results_{name}.jsonl"
        if stage in ("llm", "all"):
            run_llm_stage(ctx, all_articles, runs, MAIN_SYSTEMS, results, workers, f"{name} main")
        if stage == "ablations":
            run_llm_stage(ctx, abl_articles, runs, ablations, results, workers, f"{name} ablation")
        latency, count = store.mean_latency(backend.model_id)
        info = {"backbone": name, "backend": backend.name, "model": backend.model_id,
                "memory_gb": backend.memory_gb(), "mean_latency_s": latency, "calls_logged": count,
                "role_stats": store.role_stats(backend.model_id)}
        (full_dir / f"backbone_info_{name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        store.close()
        del ctx, backend
        gc.collect()


def apply_overrides(prof: dict, args: argparse.Namespace) -> dict:
    """Copy of the profile with the command line overrides applied."""
    prof = copy.deepcopy(prof)
    if args.max_articles is not None:
        prof["max_articles"] = args.max_articles
    if args.ablation_articles is not None:
        prof["ablation_articles"] = None if str(args.ablation_articles).lower() == "all" else int(args.ablation_articles)
    if args.paraphrases is not None:
        prof["paraphrases"] = args.paraphrases
    if args.seeds is not None:
        prof["seeds"] = args.seeds
    return prof


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.baselines:
        cfg["baselines"]["enabled"] = args.baselines
    ablations = resolve_ablations(args.ablations or cfg["ablations"]["systems"])
    prof = apply_overrides(cfg["profiles"][args.profile], args)
    out_root = Path(args.out) / args.profile
    full_dir, readable_dir = out_root / "full_logs", out_root / "readable"
    names = args.backbones or list(cfg["backbones"])

    warn_if_synced_folder()
    if args.call_stats:
        print_call_stats(cfg, names, full_dir)
        return

    paths = {r: Path(getattr(args, r)) for r in data.REGIMES if getattr(args, r)}
    if not paths:
        sys.exit("Pass at least one label file: --gold FILE, --silver FILE or --distant FILE. "
                 "Only the regimes you pass are evaluated.")
    for regime, path in paths.items():
        if not path.exists():
            sys.exit(f"{regime} file not found: {path}")
    articles, regimes, notes = data.load_regimes(paths, cfg["data"]["mask_placeholders"], prof.get("max_articles"))
    if not regimes and args.stage not in ("llm", "ablations"):
        sys.exit("None of the files has a labeled article in the selected window, so there is nothing to evaluate. "
                 "(--stage llm and --stage ablations work without labels.)")
    primary = regimes.get("gold", next(iter(regimes.values()), None))
    abl_ids = data.pick_ablation_subset(articles, primary, prof.get("ablation_articles"), cfg["ablations"]["subset"])

    print(f"Profile {args.profile}: {len(articles)} articles. Output in {out_root}/")
    if not regimes:
        print("  no labeled article yet: the LLM stages run anyway, the ablation subset is chosen at random until labels exist")
    for regime, df in regimes.items():
        print(f"  {regime}: {len(df)} labeled articles ({int(df['y'].sum())} YES, {int((1 - df['y']).sum())} NO) from {paths[regime]}")
    for note in notes:
        print(f"  note: {note}")

    if args.estimate:
        print_estimate(cfg, prof, articles["item_id"].tolist(), abl_ids, names, full_dir, ablations)
        return

    if args.stage in ("baselines", "all"):
        print("\nSupervised baselines")
        for regime, df in regimes.items():
            baselines.run_baselines(df, cfg, prof, full_dir, regime, args.force_baselines)
    if args.stage in ("llm", "ablations", "all"):
        print("\nLLM systems")
        print_estimate(cfg, prof, articles["item_id"].tolist(), abl_ids, names, full_dir, ablations)
        run_llm(cfg, prof, articles, abl_ids, names, args.stage, full_dir, args.mock, args.workers, ablations)
    if args.stage in ("report", "all"):
        print("\nBuilding the report")
        text = report.build_report(cfg, prof, articles, regimes, abl_ids, full_dir, readable_dir, names, args.profile, notes)
        print("\n" + text)
        print(f"\nReadable logs: {readable_dir}/ (summary.md first, then articles/ and the CSV files)")
        print(f"Full logs:     {full_dir}/")


if __name__ == "__main__":
    main()
