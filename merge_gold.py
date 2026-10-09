#!/usr/bin/env python3
"""Merge annotated CSV files (one per annotator) into the single gold file by majority vote."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.data import LABELS


def main() -> None:
    p = argparse.ArgumentParser(description="Majority-vote merge of annotator files.")
    p.add_argument("files", nargs="+", help="annotated CSVs: item_id, title, lead_150w, label, ...")
    p.add_argument("--out", default="data/articles.csv")
    p.add_argument("--keep-unlabeled", action="store_true",
                   help="also write the articles nobody labeled yet (empty label), so the LLM runs can start before the annotation ends")
    args = p.parse_args()

    frames = []
    for name in args.files:
        frame = pd.read_csv(name, dtype=str, keep_default_na=False)
        frame["annotator"] = Path(name).stem
        frames.append(frame)
    allrows = pd.concat(frames, ignore_index=True)
    allrows["y"] = allrows["label"].str.strip().str.upper().map(LABELS)
    everything = allrows.drop_duplicates("item_id")
    unlabeled = allrows[allrows["y"].isna()]
    if len(unlabeled):
        print(f"Ignoring {len(unlabeled)} rows without a readable label (not annotated yet).")
    allrows = allrows.dropna(subset=["y"])

    rows, ties = [], []
    for item, group in allrows.groupby("item_id", sort=False):
        n_yes, n_no = int(group["y"].sum()), int((1 - group["y"]).sum())
        first = group.iloc[0]
        record = {"item_id": item, "title": first["title"], "lead_150w": first["lead_150w"],
                  "n_raters": len(group), "n_yes": n_yes, "n_no": n_no}
        if n_yes == n_no:
            ties.append(record)
            continue
        record["label"] = "YES" if n_yes > n_no else "NO"
        rows.append(record)

    if args.keep_unlabeled:
        have = {r["item_id"] for r in rows} | {r["item_id"] for r in ties}
        for _, first in everything[~everything["item_id"].isin(have)].iterrows():
            rows.append({"item_id": first["item_id"], "title": first["title"], "lead_150w": first["lead_150w"],
                         "n_raters": 0, "n_yes": 0, "n_no": 0, "label": ""})
        order = {i: k for k, i in enumerate(everything["item_id"])}
        rows.sort(key=lambda r: order[r["item_id"]])
    out = pd.DataFrame(rows)[["item_id", "title", "lead_150w", "label", "n_raters", "n_yes", "n_no"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    multi = allrows.groupby("item_id").size()
    shared = multi[multi > 1].index
    unanimous = sum(1 for i in shared if allrows[allrows["item_id"] == i]["y"].nunique() == 1)
    print(f"Wrote {len(out)} articles to {args.out}.")
    print(f"Articles read by several annotators: {len(shared)}, unanimous: {unanimous}.")
    if ties:
        tie_path = Path(args.out).with_name("needs_adjudication.csv")
        pd.DataFrame(ties).to_csv(tie_path, index=False)
        print(f"{len(ties)} articles are tied and were left out; decide them by hand: {tie_path}")


if __name__ == "__main__":
    main()
