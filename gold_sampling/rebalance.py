#!/usr/bin/env python3
"""Step 4: after wave K is annotated, set the stratified quota of wave K+1.

    python rebalance.py --after-wave K [--force] [--dry-run]

1. Reads the returned files (paths.returned_dir, else gold_sample/annotation/),
   annotation_wave{1..K}_<annotator>*.csv plus any catch-up files, and the
   admin keys, majority-votes overlap items (ties are left out and logged).
2. Reports the controversial (YES) rate overall, per selection part, per flag
   and per stratum, with the random-only and the weighted (Hajek) prevalence.
3. Minority class = config rebalance.minority, or the rarer label so far.
   Per-flag minority rates are smoothed with a Beta prior (strata hold 1-2
   articles each, so quotas move at the flag level; the year x bias x pool x
   matched balance inside each flag is kept as planned).
4. Picks the H share of the next wave so that the expected minority share of
   the final sample enters [target_min, target_max] (unchanged if it is already
   on course), clipped to high_share_bounds. It also raises the Reddit-matched
   share if the sample is behind min_matched_total.
5. Writes data/allocation_wave{K+1}.csv, which `extract_pilot_articles.py wave K+1`
   picks up. The random part is never changed; drawn articles are never replaced
   or re-drawn (waves exclude everything already drawn).
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd

from gold.common import load_config, p, start_run
from gold import assign, strata, waves

YES = {"YES", "Y", "1", "TRUE", "OUI"}
NO = {"NO", "N", "0", "FALSE", "NON"}


def norm_label(v):
    v = str(v if v is not None and not (isinstance(v, float) and np.isnan(v)) else "").strip().upper()
    return "YES" if v in YES else "NO" if v in NO else None


def read_labels(cfg, log, K):
    rows, keys = [], []
    for k in range(1, K + 1):
        key = pd.read_csv(assign.key_path(cfg, k))
        keys.append(key.assign(wave=k))
        names = sorted(set(cfg["annotators"]) | {x for r in key.raters for x in r.split("|")})
        for a, tag in [(a, t) for a in names for t in ("", "_catchup")]:
            f = assign.returned_path(cfg, k, a, tag)
            if f is None:
                if not tag:
                    log.warning(f"wave {k}: no file for {a}")
                continue
            d = pd.read_csv(f, dtype=str)
            if not tag:
                mine = set(key.item_id[key.raters.str.split("|").map(lambda r: a in r)])
                gone = sorted(mine - set(d.item_id))
                if gone:
                    log.warning(f"{f}: {len(gone)} assigned item(s) missing: {gone}")
            for r in d.itertuples():
                lab = norm_label(r.label)
                if lab is None and str(r.label).strip() not in ("", "nan"):
                    log.warning(f"{f}: item {r.item_id} has unreadable label {r.label!r}")
                rows.append({"item_id": r.item_id, "annotator": a, "label": lab})
    j = pd.DataFrame(rows)
    key = pd.concat(keys, ignore_index=True)
    j = j.merge(key[["item_id", "article_id", "wave"]], on="item_id")

    def vote(s):
        s = s.dropna()
        if s.empty:
            return None
        c = s.value_counts()
        return None if len(c) > 1 and c.iloc[0] == c.iloc[1] else c.index[0]

    art = j.groupby(["article_id", "wave"]).label.agg(vote).rename("label").reset_index()
    ties = j.dropna(subset=["label"]).groupby("article_id").label.nunique()
    n_tie = int(((art.label.isna()) & art.article_id.isin(ties[ties > 1].index)).sum())
    if n_tie:
        log.info(f"{n_tie} overlap articles tied -> left out of rates")
    return art, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after-wave", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="rebalance even if the wave is not fully labelled")
    ap.add_argument("--dry-run", action="store_true", help="print the decision, write nothing")
    a = ap.parse_args()
    K = a.after_wave
    cfg = load_config()
    log = start_run(cfg, f"rebalance_after_wave{K}")
    ac, rc = cfg["allocation"], cfg["rebalance"]
    if os.path.exists(waves.sample_path(cfg, K + 1)):
        raise SystemExit(f"wave {K + 1} is already drawn; its quota can no longer change")

    art, key = read_labels(cfg, log, K)
    samp = pd.concat([pd.read_csv(waves.sample_path(cfg, k)) for k in range(1, K + 1)], ignore_index=True)
    x = samp.merge(art[["article_id", "label"]], on="article_id", how="left")
    done = x[x.wave == K].label.notna().mean()
    log.info(f"wave {K}: {done:.0%} of articles labelled")
    if done < rc["min_labelled_share"] and not a.force:
        raise SystemExit(f"only {done:.0%} of wave {K} labelled (< {rc['min_labelled_share']:.0%}); use --force")
    lab = x[x.label.notna()].copy()
    lab["y"] = (lab.label == "YES").astype(float)

    # ---- observed rates ----------------------------------------------------
    yes = lab.y.mean()
    rand = lab[lab.selection == "random"]
    hajek = (lab.weight * lab.y).sum() / lab.weight.sum()
    log.info(f"labelled {len(lab)}: YES share {yes:.1%} (unweighted); prevalence random-only "
             f"{rand.y.mean() if len(rand) else float('nan'):.1%} (n={len(rand)}), weighted Hajek {hajek:.1%}")
    log.info("\nby selection x flag:\n" + lab.groupby(["selection", "flag"]).y.agg(["size", "mean"]).round(3).to_string())
    log.info("\nby stratum:\n" + lab.groupby("stratum").y.agg(["size", "mean"]).round(2).to_string())

    minority = rc["minority"] if rc["minority"] in ("YES", "NO") else ("YES" if yes < 0.5 else "NO")
    lab["m"] = (lab.label == minority).astype(float)
    a0, b0 = rc["prior"]
    st = lab[lab.selection == "stratified"]

    def rate(d):
        return (d.m.sum() + a0) / (len(d) + a0 + b0)

    pH, pL, pR = rate(st[st.flag == "H"]), rate(st[st.flag == "L"]), rate(lab[lab.selection == "random"])
    log.info(f"minority class = {minority}; smoothed minority rate H={pH:.2f} L={pL:.2f} random={pR:.2f}")

    r = ac["n_waves"] - K
    if r == 0:
        log.info("last wave annotated; nothing to rebalance")
        return
    n_r, n_s = ac["n_random"] / ac["n_waves"], (ac["n_total"] - ac["n_random"]) / ac["n_waves"]
    M, T = lab.m.sum(), len(lab)

    def f(s):
        return (M + r * (n_r * pR + n_s * (s * pH + (1 - s) * pL))) / (T + r * (n_r + n_s))

    last = samp[(samp.wave == K) & (samp.selection == "stratified")]
    s_cur = (last.stratum.str[0] == "H").mean()
    lo, hi = rc["high_share_bounds"]
    if rc["target_min"] <= f(s_cur) <= rc["target_max"]:
        s_new, why = s_cur, "on course"
    else:
        aim = rc["target_min"] if f(s_cur) < rc["target_min"] else rc["target_max"]
        slope = r * n_s * (pH - pL) / (T + r * (n_r + n_s))
        s_new = s_cur if abs(slope) < 1e-9 else s_cur + (aim - f(s_cur)) / slope
        why = f"aim {aim:.2f}"
    s_new = float(np.clip(s_new, lo, hi))
    log.info(f"H share: wave {K} {s_cur:.2f} -> wave {K + 1} {s_new:.2f} ({why}); "
             f"expected final minority share {f(s_cur):.2f} -> {f(s_new):.2f}")

    # ---- Reddit-matched catch-up ------------------------------------------
    cfg2 = copy.deepcopy(cfg)
    matched = (samp.matched == "m1").sum()
    need = ac["min_matched_total"] - matched - r * n_r * (samp[samp.selection == "random"].matched == "m1").mean()
    url_share = 1 - len(set(ac["no_url_years"]) & set(ac["years"])) / len(ac["years"])
    req = need / max(r * n_s * url_share, 1)
    cfg2["allocation"]["matched_share"] = float(np.clip(max(ac["matched_share"], req), 0, 1))
    log.info(f"Reddit-matched so far {matched}; matched share for wave {K + 1}: {cfg2['allocation']['matched_share']:.2f}")

    frame = waves.load_frame(cfg)
    already = set(samp.article_id) | set(waves.random_reserve(cfg, frame)[0])   # incl. future random picks
    res, mv = strata.plan(cfg2, log, frame, already=already, n=n_s, high_share=s_new,
                          seed_name=f"rebalance_wave{K + 1}")
    out = res[res.n_alloc > 0][["n_alloc"]].rename(columns={"n_alloc": "n"})
    out.index.name = "stratum"
    log.info(f"wave {K + 1} allocation: {int(out.n.sum())} stratified units, "
             f"H={int(out.n[out.index.str[0] == 'H'].sum())}")
    if a.dry_run:
        log.info("dry run: nothing written")
        return
    path = os.path.join(p(cfg, "data_dir"), f"allocation_wave{K + 1}.csv")
    out.reset_index().to_csv(path, index=False)
    log.info(f"wrote {path}")


if __name__ == "__main__":
    main()
