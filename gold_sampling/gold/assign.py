"""Step 5: annotator assignment for one wave (reuses the pilot's logic).

- Overlap subset: config `assignment.overlap_total` articles over all waves
  (split evenly), each labelled by `raters_per_overlap` annotators via the
  pilot's greedy `_build_design`, with the load carried across waves. Overlap
  items are picked systematically from the wave sorted by stratum, so they are
  spread across strata and the random part.
- Solo items are sorted by (flag, bias3, year, stratum) and dealt round-robin,
  starting one annotator later each wave, so nobody gets mostly H or mostly L.
- Annotation files are BLIND: only the pilot's ANNOTATOR_COLS (item_id,
  title, lead_150w, label, confidence, reason). Stratum, proxies, publisher and
  URL live in gold_sample/_admin/ only.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

from .common import p, sub_seed

import extract_pilot_articles as pilot

BLIND_COLS = pilot.ANNOTATOR_COLS


def ann_dir(cfg):
    return os.path.join(p(cfg, "sample_dir"), "annotation")


def admin_dir(cfg):
    return os.path.join(p(cfg, "sample_dir"), "_admin")


def ann_path(cfg, k, a):
    return os.path.join(ann_dir(cfg), f"annotation_wave{k}_{a}.csv")


def key_path(cfg, k):
    return os.path.join(admin_dir(cfg), f"assignment_key_wave{k}.csv")


def returned_dir(cfg, k):
    return p(cfg, "returned_dir").format(k=k) if "returned_dir" in cfg["paths"] else ann_dir(cfg)


def returned_path(cfg, k, a, tag=""):
    """The labelled file annotator `a` handed back for wave k: the one
    annotation_wave{k}{tag}_{a}*.csv in returned_dir (tolerates renames such as
    `raphael1`; `:Zone.Identifier` side files do not end in .csv). Falls back to
    the file that was sent out. None if neither exists."""
    import glob
    hits = sorted(glob.glob(os.path.join(returned_dir(cfg, k), f"annotation_wave{k}{tag}_{a}*.csv")))
    if len(hits) > 1:
        raise SystemExit(f"several returned files for {a}, wave {k}{tag}: {hits}")
    if hits:
        return hits[0]
    sent = os.path.join(ann_dir(cfg), f"annotation_wave{k}{tag}_{a}.csv")
    return sent if os.path.exists(sent) else None


def catchup_key_path(cfg, k):
    return os.path.join(admin_dir(cfg), f"assignment_key_wave{k}_catchup.csv")


def catchup(cfg, log, k, absent):
    """Items of wave k whose every rater is in `absent` go to the active
    annotators (cfg annotators minus absent), sorted by flag and dealt
    round-robin. Titles/leads are copied from the files already sent, so
    nothing is re-fetched. Writes annotation_wave{k}_catchup_<a>.csv (blind)
    and _admin/assignment_key_wave{k}_catchup.csv. Item ids are unchanged."""
    key = pd.read_csv(key_path(cfg, k))
    active = [a for a in cfg["annotators"] if a not in absent]
    orphan = key[key.raters.str.split("|").map(lambda r: set(r) <= set(absent))]
    orphan = orphan.sort_values(["flag", "item_id"]).reset_index(drop=True)
    if orphan.empty:
        log.info(f"wave {k}: no item depends only on {absent}")
        return {}, orphan
    texts = pd.concat([pd.read_csv(ann_path(cfg, k, a), dtype=str) for a in absent]).drop_duplicates("item_id")
    texts = texts.set_index("item_id")
    orphan["new_rater"] = [active[i % len(active)] for i in range(len(orphan))]
    rng = np.random.default_rng(sub_seed(cfg, f"wave{k}:catchup"))
    files = {}
    for a, mine in orphan.groupby("new_rater"):
        df = pd.DataFrame({"item_id": mine.item_id, "title": mine.item_id.map(texts.title),
                           "lead_150w": mine.item_id.map(texts.lead_150w),
                           "label": "", "confidence": "", "reason": ""})[BLIND_COLS]
        if df.title.isna().any():
            raise SystemExit(f"catch-up: no text for {df.item_id[df.title.isna()].tolist()}")
        files[a] = df.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)
        log.info(f"wave {k} catch-up {a:10s}: {len(df)} items, flag {mine.flag.value_counts().to_dict()}")
    return files, orphan[["item_id", "article_id", "raters", "new_rater", "selection", "flag"]]


def run_catchup(cfg, log, k, absent):
    files, key = catchup(cfg, log, k, absent)
    for a, df in files.items():
        df.to_csv(os.path.join(ann_dir(cfg), f"annotation_wave{k}_catchup_{a}.csv"), index=False)
    if len(key):
        key.to_csv(catchup_key_path(cfg, k), index=False)
        log.info(f"wrote {len(files)} catch-up files ({len(key)} items) and {catchup_key_path(cfg, k)}")


def main():
    """python extract_pilot_articles.py catchup K name [name ...]"""
    import sys
    from .common import load_config, start_run
    cfg = load_config()
    k, absent = int(sys.argv[2]), sys.argv[3:]
    if not absent:
        raise SystemExit("usage: catchup K absent_annotator [...]")
    log = start_run(cfg, f"catchup_wave{k}")
    run_catchup(cfg, log, k, absent)


def _overlap_quota(cfg, k):
    a = cfg["assignment"]
    n, w = a["overlap_total"], cfg["allocation"]["n_waves"]
    return n // w + (1 if k <= n % w else 0)


def _prior_load(cfg, k):
    load = Counter({a: 0 for a in cfg["annotators"]})
    for j in range(1, k):
        if os.path.exists(key_path(cfg, j)):
            kd = pd.read_csv(key_path(cfg, j))
            for rs in kd.raters[kd.overlap]:
                load.update(rs.split("|"))
    return load


def assign(cfg, log, k, sample, texts):
    """Pure-ish: returns ({annotator: DataFrame}, key DataFrame)."""
    annotators = cfg["annotators"]
    ac = cfg["assignment"]
    rng = np.random.default_rng(sub_seed(cfg, f"wave{k}:assign"))
    s = sample.sort_values(["flag", "bias3", "year", "stratum", "article_id"]).reset_index(drop=True)

    n_ov = _overlap_quota(cfg, k)
    step = len(s) / n_ov
    off = rng.random() * step
    ov_idx = sorted({int(off + i * step) for i in range(n_ov)})
    s["overlap"] = s.index.isin(ov_idx)

    raters = min(ac["raters_per_overlap"], len(annotators))
    design = pilot._build_design(int(s.overlap.sum()), annotators=list(annotators), raters=raters,
                                 load=_prior_load(cfg, k))
    raters_of = {}
    for aid, picks in zip(s.article_id[s.overlap], design):
        raters_of[aid] = list(picks)
    solo = s[~s.overlap]
    shift = (k - 1) % len(annotators)
    for i, aid in enumerate(solo.article_id):
        raters_of[aid] = [annotators[(i + shift) % len(annotators)]]

    ids = rng.permutation(len(s)) + 1
    s["item_id"] = [f"W{k}-{i:03d}" for i in ids]
    s["raters"] = s.article_id.map(lambda a: "|".join(raters_of[a]))
    s["n_raters"] = s.article_id.map(lambda a: len(raters_of[a]))

    t = texts.set_index("article_id")
    files = {}
    for a in annotators:
        mine = s[s.raters.str.split("|").map(lambda r: a in r)]
        df = pd.DataFrame({"item_id": mine.item_id,
                           "title": mine.article_id.map(t.title),
                           "lead_150w": mine.article_id.map(t.lead_150w),
                           "label": "", "confidence": "", "reason": ""})[BLIND_COLS]
        files[a] = df.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)
        log.info(f"wave {k} {a:10s}: {len(df)} items, overlap {int(mine.overlap.sum())}, "
                 f"flag {mine.flag.value_counts().to_dict()}, bias {mine.bias3.value_counts().to_dict()}")
    key = s[["item_id", "article_id", "overlap", "n_raters", "raters", "selection", "stratum", "flag"]]
    return files, key.sort_values("item_id")


def run(cfg, log, k, sample):
    from .textfetch import fetch
    os.makedirs(ann_dir(cfg), exist_ok=True)
    os.makedirs(admin_dir(cfg), exist_ok=True)
    texts = fetch(cfg, sample.article_id.tolist())
    files, key = assign(cfg, log, k, sample, texts)
    # catch-up items of the previous wave travel inside this wave's files
    # (they keep their W{k-1}- ids, so labels map back to the right wave)
    if k > 1 and os.path.exists(catchup_key_path(cfg, k - 1)):
        rng = np.random.default_rng(sub_seed(cfg, f"wave{k}:with_catchup"))
        for a in list(files):
            f = os.path.join(ann_dir(cfg), f"annotation_wave{k - 1}_catchup_{a}.csv")
            if os.path.exists(f):
                extra = pd.read_csv(f, dtype=str, keep_default_na=False)
                files[a] = pd.concat([files[a], extra[BLIND_COLS]], ignore_index=True) \
                    .sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)
                log.info(f"wave {k} {a:10s}: + {len(extra)} catch-up items from wave {k - 1} -> {len(files[a])}")
    for a, df in files.items():
        df.to_csv(ann_path(cfg, k, a), index=False)
    key.to_csv(key_path(cfg, k), index=False)
    g = os.path.join(ann_dir(cfg), "GUIDELINE.md")
    if not os.path.exists(g):
        with open(g, "w", encoding="utf-8") as f:
            f.write(pilot.GUIDELINE)
    log.info(f"wrote {len(files)} blind annotation files to {ann_dir(cfg)} and {key_path(cfg, k)}")
