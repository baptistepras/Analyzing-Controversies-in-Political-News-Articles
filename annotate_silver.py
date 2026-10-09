#!/usr/bin/env python3
"""Make the silver labels with an annotator LLM that never sees the human labels.

Only the columns item_id, title and lead_150w are read from the article file. Any label, confidence or
reason column in it is never loaded, so the silver labels cannot depend on what the team said.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from pipeline import data, prompts
from pipeline.backends import CallStore, make_backend, read_records, role_stats
from pipeline.llm_systems import LLMContext, ask, extract_json

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

READ_COLUMNS = ["item_id", "title", "lead_150w"]
LABEL_RE = r'"label"\s*:\s*"?\s*(YES|NO)'
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


def parse_silver(text: str) -> dict | None:
    """Read {label, confidence, reason} from an answer; None if the label is not YES or NO."""
    obj = extract_json(text)
    label, confidence, reason = None, "", ""
    if obj is not None and "label" in obj:
        label = str(obj["label"]).strip().upper()
        confidence, reason = str(obj.get("confidence", "")).strip().lower(), str(obj.get("reason", "")).strip()
        confidence = confidence if confidence in CONFIDENCE_RANK else ""
    else:
        import re

        match = re.search(LABEL_RE, text, re.IGNORECASE)
        if match:
            label, reason = match.group(1).upper(), text.strip().replace("\n", " ")[:160]
    if label is None:
        return None
    label = "YES" if label.startswith("YES") else "NO" if label.startswith("NO") else None
    return {"label": label, "confidence": confidence, "reason": reason} if label else None


def vote(votes: list[dict | None]) -> tuple[str, str, str, int, int]:
    """Majority label, agreement level, a reason, number of YES votes, number of readable votes."""
    valid = [v for v in votes if v]
    yes = sum(1 for v in valid if v["label"] == "YES")
    n = len(valid)
    if n == 0 or yes * 2 == n:
        return "", "", "", yes, n
    label = "YES" if yes * 2 > n else "NO"
    agreement = "high" if (yes == n or yes == 0) and n == len(votes) else "low"
    reason = next((v["reason"] for v in valid if v["label"] == label and v["reason"]), "")
    return label, agreement, reason, yes, n


def self_confidence(votes: list[dict | None], label: str) -> str:
    """Lowest confidence the annotator reported among the passes that voted for the final label."""
    found = [v["confidence"] for v in votes if v and v["label"] == label and v["confidence"] in CONFIDENCE_RANK]
    return min(found, key=CONFIDENCE_RANK.__getitem__) if found else ""


def seconds_per_call(cfg: dict, log: Path, model_id: str) -> tuple[float, str]:
    """Mean latency of the logged silver calls if there are enough of them, else the assumed value."""
    stats = role_stats(read_records(log), model_id).get("silver")
    if stats and stats["n"] >= 20 and stats["latency_s"]:
        return stats["latency_s"], "measured"
    return cfg["silver"]["assumed_s_per_call"], "assumed"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Silver labels from an annotator LLM, blind to the human labels.")
    p.add_argument("--articles", required=True, help="CSV with item_id, title, lead_150w (other columns are never read)")
    p.add_argument("--out", default="data/silver.csv")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--logdir", default="outputs/silver", help="call log (resume cache) and report")
    p.add_argument("--passes", type=int, default=None, help="wordings of the guideline, majority vote (default: config)")
    p.add_argument("--limit", type=int, default=None, help="only the first N articles")
    p.add_argument("--mock", action="store_true", help="fake model, for testing the plumbing only")
    p.add_argument("--estimate", action="store_true", help="print the number of calls and hours, then stop")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    scfg = cfg["silver"]
    passes = args.passes or scfg["passes"]
    path = Path(args.articles)
    if not path.exists():
        sys.exit(f"Article file not found: {path}")
    df = pd.read_csv(path, usecols=lambda c: c.strip() in READ_COLUMNS, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in READ_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"{path} is missing the columns {missing}")
    df = df.drop_duplicates("item_id")
    if cfg["data"]["mask_placeholders"]:
        for col in ("title", "lead_150w"):
            df[col] = df[col].str.replace(data.PLACEHOLDER_RE, "[...]", regex=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"{len(df)} articles from {path}. Columns read: {READ_COLUMNS} (labels are never loaded).")

    logdir = Path(args.logdir)
    log = logdir / "calls_silver.jsonl"
    model_id = f"mock-silver" if args.mock else scfg["model"]
    n_calls = len(df) * passes
    sec, source = seconds_per_call(cfg, log, model_id)
    print(f"Silver annotator: {model_id}, {passes} passes. {n_calls} calls, about {n_calls * sec / 3600:.1f} h "
          f"at {sec:.1f} s per call ({source}).")
    if args.estimate:
        return

    backend = make_backend("silver", scfg, args.mock)
    backend.preflight()
    store = CallStore(log)
    ctx = LLMContext("silver", backend, store,
                     {"max_tokens": {"silver": scfg["max_tokens"]}, "max_retries": scfg["max_retries"]})
    rows, failures = [], 0
    bar = tqdm(total=len(df), desc="silver", unit="article") if tqdm else None
    for art in df.itertuples():
        votes = []
        for k in range(passes):
            user = prompts.silver_prompt(k, art.title, art.lead_150w)
            parsed, records, fails = ask(ctx, None, user, parse_silver, "silver", scfg["temp"],
                                         (art.item_id, "silver", k),
                                         {"item_id": art.item_id, "role": "silver", "pass": k})
            votes.append(parsed)
            failures += fails
        label, agreement, reason, yes, n = vote(votes)
        rows.append({"item_id": art.item_id, "title": art.title, "lead_150w": art.lead_150w, "label": label,
                     "confidence": self_confidence(votes, label), "agreement": agreement, "reason": reason,
                     "n_yes": yes, "n_valid": n,
                     "passes": passes, "model": backend.model_id})
        if bar:
            bar.update(1)
    if bar:
        bar.close()
    store.close()

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    lab = out[out["label"] != ""]
    n_yes = int((lab["label"] == "YES").sum())
    lines = [f"# Silver annotation report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')}. Annotator: {backend.model_id}. Passes: {passes}. "
             f"Input: {path} (columns read: {', '.join(READ_COLUMNS)}).", "",
             f"- Articles: {len(out)}",
             f"- Labeled: {len(lab)} ({n_yes} YES, {len(lab) - n_yes} NO, {100 * n_yes / max(1, len(lab)):.0f}% YES)",
             f"- Without label (tie or unreadable answers): {len(out) - len(lab)}",
             f"- Unanimous passes: {int((lab['agreement'] == 'high').sum())}, split passes: {int((lab['agreement'] == 'low').sum())}",
             f"- Confidence reported by the annotator (lowest among the passes for the final label): "
             f"{int((lab['confidence'] == 'high').sum())} high, {int((lab['confidence'] == 'medium').sum())} medium, "
             f"{int((lab['confidence'] == 'low').sum())} low",
             f"- Unreadable answers that triggered a retry: {failures}", ""]
    stats = role_stats(read_records(log), backend.model_id).get("silver")
    if stats and stats["latency_s"]:
        lines += [f"- Mean latency per call: {stats['latency_s']:.2f} s over {stats['n']} logged calls", ""]
    empty = out.loc[out["label"] == "", "item_id"].tolist()
    if empty:
        lines += ["Articles left without a label (they are scored by the pipeline but not evaluated on silver):",
                  ", ".join(empty), ""]
    (logdir / "silver_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
