"""Offline self-test: metric values, parsers, and the whole pipeline end to end with the mock model."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import annotate_silver  # noqa: E402
import run_all  # noqa: E402
from pipeline import data, metrics as M  # noqa: E402
from pipeline.llm_systems import entropy_score, parse_score, parse_stance  # noqa: E402


def check(name: str, ok: bool) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        raise SystemExit(f"self-test failed: {name}")


def test_metrics() -> None:
    print("metrics")
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    check("PR-AUC hand-computed value", abs(M.pr_auc(y, s, 1) - (1 + 2 / 3) / 2) < 1e-9)
    check("P@1 with a four-way tie", abs(M.precision_at_k(np.array([1, 0, 0, 0]), np.full(4, 0.5), 1, 1) - 0.25) < 1e-9)
    check("P@2 with a tie at the cut", abs(M.precision_at_k(np.array([1, 1, 0, 0]), np.array([0.9, 0.5, 0.5, 0.1]), 2, 1) - 0.75) < 1e-9)
    check("ECE is 0 for perfect scores", M.ece(np.array([0, 1, 1]), np.array([0.0, 1.0, 1.0])) == 0)
    check("best macro-F1 is 1 on separable data", abs(M.best_macro_f1(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) - 1) < 1e-9)
    check("noise test says '+'", M.compare(0.8, 0.01, 0.6, 0.02) == "+")
    check("noise test says '~'", M.compare(0.8, 0.1, 0.75, 0.02) == "~")
    dec = M.decompose({(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.7, (1, 1): 0.7})
    check("decomposition: all spread is prompt spread", dec["seed"] == 0 and dec["prompt"] > 0)


def test_parsers() -> None:
    print("parsers")
    check("stance from JSON", parse_stance('Sure! {"reaction": "meh", "stance": -1}')["stance"] == -1)
    check("stance clamped", parse_stance('{"reaction": "x", "stance": 2}')["stance"] == 1)
    check("stance from broken JSON", parse_stance('{"reaction": "he said "no"", "stance": 1}')["stance"] == 1)
    check("garbage gives None", parse_stance("I cannot help with that.") is None)
    parsed = parse_score('{"controversy": 85, "summary": "split"}')
    check("score on a 0 to 100 scale is rescaled", abs(parsed["controversy"] - 0.85) < 1e-9 and parsed["summary"] == "split")
    typo = parse_score('{"controversity": 0.95, "reason": "split"}')
    check("misspelled key 'controversity' is accepted", typo is not None and abs(typo["controversy"] - 0.95) < 1e-9)
    check("entropy of a split panel", abs(entropy_score([-1, 0, 1], 2) - 1.0) < 1e-9)
    check("entropy of an unanimous panel", entropy_score([1, 1, 1], 2) == 0.0)
    check("entropy with too few answers is NaN", np.isnan(entropy_score([1, None, None], 2)))
    check("silver label from JSON", annotate_silver.parse_silver('{"label": "yes", "confidence": "high", "reason": "x"}')["label"] == "YES")
    check("silver label from broken JSON", annotate_silver.parse_silver('{"label": "NO", "reason": "he said "x""}')["label"] == "NO")
    check("silver garbage gives None", annotate_silver.parse_silver("maybe") is None)
    v = [{"label": "YES", "reason": "a"}, {"label": "YES", "reason": "b"}, {"label": "NO", "reason": "c"}]
    check("majority vote with a split", annotate_silver.vote(v)[:2] == ("YES", "low"))
    check("unanimous vote is high", annotate_silver.vote(v[:2] + [v[0]])[:2] == ("YES", "high"))
    check("a tie gives no label", annotate_silver.vote([v[0], v[2]])[0] == "")


def make_dataset(path: Path, n: int = 40, unlabeled: int = 0) -> None:
    rng = np.random.default_rng(0)
    hot = "impeachment senate gun immigration abortion climate election".split()
    calm = "weather festival recipe museum marathon concert garden".split()
    noise = "said report city officials week president local new".split()
    rows = []
    for i in range(n):
        yes = rng.random() < 0.65
        words = list(rng.choice(hot if yes else calm, 6)) + list(rng.choice(noise, 14))
        rng.shuffle(words)
        rows.append({"item_id": f"T{i:03d}", "title": f"{words[0].title()} {words[1]} {words[2]}",
                     "lead_150w": " ".join(words) + ".", "label": "YES" if yes else "NO",
                     "confidence": "high", "reason": "GOLDREASON"})
    for row in rows[n - unlabeled:]:
        row["label"] = ""
    pd.DataFrame(rows).to_csv(path, index=False)


def test_data(tmp: Path) -> None:
    print("data loading")
    make_dataset(tmp / "gold.csv", 40, unlabeled=4)
    gold = pd.read_csv(tmp / "gold.csv", dtype=str, keep_default_na=False)
    gold.loc[:, ["title", "lead_150w"]] = ""  # a label-only file, like a distant file
    gold[["item_id", "label"]].head(30).to_csv(tmp / "distant_only_ids.csv", index=False)
    articles, regimes, notes = data.load_regimes({"gold": tmp / "gold.csv", "distant": tmp / "distant_only_ids.csv"})
    check("all 40 articles are kept", len(articles) == 40)
    check("unlabeled gold rows are not evaluated", len(regimes["gold"]) == 36)
    check("a distant file without text takes the text from gold", len(regimes["distant"]) == 30
          and regimes["distant"]["lead_150w"].str.len().min() > 0)
    check("a note mentions the unlabeled rows", any("no label yet" in n for n in notes))
    small, _, _ = data.load_regimes({"gold": tmp / "gold.csv"}, max_articles=10)
    check("max_articles keeps the first articles", small["item_id"].tolist() == articles["item_id"].tolist()[:10])
    sub60 = data.pick_ablation_subset(articles, regimes["gold"], 12)
    pos = 1 if regimes["gold"]["y"].mean() <= 0.5 else 0
    minority = set(regimes["gold"].loc[regimes["gold"]["y"] == pos, "item_id"])
    check("the ablation subset has the requested size", len(sub60) == 12)
    check("the ablation subset keeps the minority articles", len(minority & set(sub60)) == min(12, len(minority)))
    sub_small = data.pick_ablation_subset(articles.head(30), regimes["gold"][regimes["gold"]["item_id"].isin(articles["item_id"][:30])], 12)
    check("subset choice is stable when articles are added", len(set(sub60) & set(sub_small)) >= 8)


def test_silver(tmp: Path) -> Path:
    print("silver annotation")
    make_dataset(tmp / "gold.csv", 40, unlabeled=4)
    base = ["--config", str(Path(__file__).resolve().parent.parent / "config.yaml"), "--mock", "--passes", "1"]
    annotate_silver.main(["--articles", str(tmp / "gold.csv"), "--out", str(tmp / "silver.csv"), "--logdir", str(tmp / "silver_a")] + base)
    silver = pd.read_csv(tmp / "silver.csv", dtype=str, keep_default_na=False)
    log = (tmp / "silver_a" / "calls_silver.jsonl").read_text()
    check("silver has one row per article", len(silver) == 40)
    check("silver never saw the gold reason text", "GOLDREASON" not in log)
    flipped = pd.read_csv(tmp / "gold.csv", dtype=str, keep_default_na=False)
    flipped["label"] = flipped["label"].map({"YES": "NO", "NO": "YES", "": ""})
    flipped["reason"] = "OTHER"
    flipped.to_csv(tmp / "gold_flipped.csv", index=False)
    annotate_silver.main(["--articles", str(tmp / "gold_flipped.csv"), "--out", str(tmp / "silver_b.csv"), "--logdir", str(tmp / "silver_b")] + base)
    again = pd.read_csv(tmp / "silver_b.csv", dtype=str, keep_default_na=False)
    check("flipping the gold labels does not change the silver labels", silver["label"].tolist() == again["label"].tolist())
    check("silver cache makes a rerun free", sum(1 for _ in (tmp / "silver_a" / "calls_silver.jsonl").open()) ==
          sum(1 for _ in (tmp / "silver_b" / "calls_silver.jsonl").open()))
    return tmp / "silver.csv"


def test_pipeline(tmp: Path, silver: Path) -> None:
    print("pipeline with the mock model")
    base = yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())
    base["profiles"]["test"] = {
        "max_articles": None, "ablation_articles": 20, "paraphrases": 2, "seeds": [0, 1],
        "cv_folds": 3, "cv_seeds": [0, 1], "roberta_epochs": 1,
    }
    (tmp / "config.yaml").write_text(yaml.safe_dump(base))
    rng = np.random.default_rng(1)
    gold = pd.read_csv(tmp / "gold.csv", dtype=str, keep_default_na=False)
    dist = gold[["item_id", "label"]].copy()
    flip = rng.random(len(dist)) < 0.3
    dist.loc[flip & (dist["label"] != ""), "label"] = dist.loc[flip & (dist["label"] != ""), "label"].map({"YES": "NO", "NO": "YES"})
    dist.to_csv(tmp / "distant.csv", index=False)
    args = ["--config", str(tmp / "config.yaml"), "--profile", "test", "--gold", str(tmp / "gold.csv"),
            "--silver", str(silver), "--distant", str(tmp / "distant.csv"),
            "--mock", "--out", str(tmp / "out"), "--baselines", "tfidf", "--stage", "all"]
    try:
        run_all.main(["--config", str(tmp / "config.yaml"), "--profile", "test", "--mock", "--out", str(tmp / "out_none")])
        check("no label file is refused", False)
    except SystemExit as exc:
        check("no label file is refused", "at least one label file" in str(exc))
    run_all.main(args)
    full, readable = tmp / "out" / "test" / "full_logs", tmp / "out" / "test" / "readable"
    roles = lambda: {r["meta"].get("role") for r in map(json.loads, (full / "calls_qwen.jsonl").read_text().splitlines())}
    check("no ablation before the ablation stage", not roles() & {"delib", "homog", "selfcons"})
    check("no ablation section before the ablations ran", "### Ablation on" not in (readable / "summary.md").read_text())
    run_all.main(args[:-2] + ["--stage", "ablations"])
    check("only the homogeneous ablation ran", {"homog"} <= roles() and not roles() & {"delib", "selfcons"})
    run_all.main(args[:-2] + ["--stage", "report"])
    summary = (readable / "summary.md").read_text()
    check("summary.md written", (readable / "summary.md").exists())
    check("quick answers come first", summary.index("## A. Quick answers") < summary.index("## B. Leaderboards") < summary.index("# Details"))
    check("three regimes are reported", all(f"{n.capitalize()} labels" in summary for n in ("gold", "silver", "distant")))
    check("RQ1 section present", "Kendall" in summary and "Cohen" in summary)
    check("ablation section present", "### Ablation on" in summary and "heterogeneous" in summary)
    check("baselines are trained per regime", all((full / f"baseline_scores_{n}.csv").exists() for n in ("gold", "silver", "distant")))
    check("one card per article", len(list((readable / "articles").glob("*.md"))) == 40)
    card = (readable / "articles" / "T000.md").read_text()
    check("card shows personas and the labels", "persona_far_left" in card and "Gold label" in card and "Silver label" in card)
    check("metrics file has a regime column", "regime" in pd.read_csv(readable / "metrics_by_run.csv").columns)

    calls_before = sum(1 for _ in (full / "calls_qwen.jsonl").open())
    run_all.main(args)
    check("rerun makes no new model call", sum(1 for _ in (full / "calls_qwen.jsonl").open()) == calls_before)
    run_all.main(args[:-2] + ["--estimate"])
    check("estimate runs", True)

    # a first pass with one seed, then the full grid: the first runs must be reused
    seeds_args = args[:args.index("--out")] + ["--out", str(tmp / "out_seeds"), "--stage", "llm"]
    run_all.main(seeds_args + ["--seeds", "0"])
    n_first = sum(1 for _ in (tmp / "out_seeds" / "test" / "full_logs" / "calls_qwen.jsonl").open())
    run_all.main(seeds_args)
    n_second = sum(1 for _ in (tmp / "out_seeds" / "test" / "full_logs" / "calls_qwen.jsonl").open())
    check("growing the grid only adds the new runs", 0 < n_first < n_second and n_second < 2.2 * n_first)

    # LLM stages must run before any label exists
    pd.DataFrame({"item_id": gold["item_id"], "title": gold["title"], "lead_150w": gold["lead_150w"], "label": ""}).to_csv(tmp / "nolabel.csv", index=False)
    nl = ["--config", str(tmp / "config.yaml"), "--profile", "test", "--gold", str(tmp / "nolabel.csv"), "--mock", "--out", str(tmp / "out_nl")]
    run_all.main(nl + ["--stage", "llm", "--seeds", "0", "--paraphrases", "1"])
    check("LLM stage runs without labels", (tmp / "out_nl" / "test" / "full_logs" / "results_qwen.jsonl").exists())
    try:
        run_all.main(nl + ["--stage", "report"])
        check("the report needs labels", False)
    except SystemExit as exc:
        check("the report needs labels", "nothing to evaluate" in str(exc))

    par_args = args[:args.index("--out")] + ["--out", str(tmp / "out_par"), "--stage", "llm", "--workers", "3"]
    run_all.main(par_args)

    def keyed(path: Path) -> set:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        return {(r["system"], r["paraphrase"], r["seed"], r["item_id"], r["score"]) for r in rows
                if r["system"] in {"llm_single", "audience", "audience_direct"}}

    same = keyed(full / "results_qwen.jsonl") == keyed(tmp / "out_par" / "test" / "full_logs" / "results_qwen.jsonl")
    check("parallel run gives the same scores as the sequential run", same)


if __name__ == "__main__":
    test_metrics()
    test_parsers()
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        test_data(tmp)
        silver_path = test_silver(tmp)
        test_pipeline(tmp, silver_path)
    print("\nAll self-tests passed.")
