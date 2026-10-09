#!/usr/bin/env python3
"""
Pilot annotation pipeline for ioverho/misinfo-general.

Design (configurable below):
  5 annotators x 8 articles each = 40 judgments
  3 of each person's 8 are SHARED (annotated by 3 different people)
  -> 5 shared articles + 25 solo articles = 30 unique articles
  Every pair of annotators shares at least one article, so all pairwise
  agreements are computable.

Dataset facts (verified on the HF dataset page):
  ~4.16M articles, 488 publishers, 9.62 GB, splits are YEARS "2017".."2022"
  columns: source, title, content, author, domain, raw_url,
           publication_date, article_id
  publisher-level Media Bias/Fact Check labels live in a separate metadata.db
  NOTE: for 2018, author/domain/raw_url are "N/A" -> avoid 2018.

Usage:
    python extract_pilot_articles.py inspect   # verify split names, peek rows
    python extract_pilot_articles.py labels    # download+query metadata.db only
    python extract_pilot_articles.py extract   # stream -> pilot_candidates.csv
    #   >>> manual step: fill the `stratum` column (A/B/C) for ~30 rows <<<
    python extract_pilot_articles.py assign    # build per-person folders
    #   >>> manual step: everyone fills their articles.csv <<<
    python extract_pilot_articles.py merge     # agreement stats

Gold sample (200 articles, config.yaml; code in gold/):
    python extract_pilot_articles.py content-stats  # stream HF parquet -> per-article length/mask stats
    python extract_pilot_articles.py reddit-fetch   # Arctic Shift submissions, 4 subreddits
    python extract_pilot_articles.py wiki           # Wikipedia controversial-issue lists
    python extract_pilot_articles.py candidates     # Step 1 -> candidates.parquet
    python extract_pilot_articles.py proxies        # Step 2 -> proxies.parquet
    python extract_pilot_articles.py strata         # Step 3 -> proxy flag + allocation plan
    python extract_pilot_articles.py wave K         # Step 4+5: draw wave K, write blind annotation files
    python extract_pilot_articles.py catchup K name # re-deal wave-K items whose only rater (name) is absent
    python rebalance.py --after-wave K              # after wave K is annotated: quota for wave K+1
"""

import sys
import os
import csv
import re
import random
from collections import defaultdict, Counter
from itertools import combinations

# ----------------------------------------------------------------------------
# CONFIG — change these two lines if only four of you annotate
# ----------------------------------------------------------------------------
ANNOTATORS = ["vladimir", "martin", "ksenia", "raphael", "baptiste"]
ARTICLES_EACH = 8          # articles per person
OVERLAP_EACH = 3           # how many of those are shared with others
RATERS_PER_SHARED = 3      # how many people see each shared article
SOLO_B_SHARE = 0.5         # fraction of solo articles drawn from stratum B (rest from A)

YEAR = "2020"              # 2020 = US election, COVID policy, BLM
N_CANDIDATES = 200
MAX_SCANNED = 60_000
SEED = 20260927

POOL = "pilot_candidates.csv"
OUTDIR = "pilot_annotation"

POLITICAL = re.compile(
    r"\b(abortion|immigration|immigrant|migrant|asylum|gun control|firearm|"
    r"second amendment|climate|emission|vaccine|vaccination|mandate|lockdown|"
    r"election|ballot|voter|senate|congress|parliament|president|prime minister|"
    r"supreme court|legislation|bill|policy|tax|tariff|healthcare|welfare|"
    r"minimum wage|pension|brexit|referendum|protest|impeach|sanction|"
    r"border|deportation|refugee|racism|police reform|defund)\b",
    re.I,
)

# Columns the annotator sees. Deliberately NOT source/domain/url/stratum:
# seeing the outlet or the expected stratum would bias the judgment.
ANNOTATOR_COLS = ["item_id", "title", "lead_150w", "label", "confidence", "reason"]


# ----------------------------------------------------------------------------
def inspect():
    from datasets import load_dataset, get_dataset_config_names, get_dataset_split_names
    for fn, name in ((get_dataset_config_names, "configs"), (get_dataset_split_names, "splits")):
        try:
            print(name + ":", fn("ioverho/misinfo-general"))
        except Exception as e:
            print(name + ": could not list ->", e)

    ds = load_dataset("ioverho/misinfo-general", split=YEAR, streaming=True)
    for i, row in enumerate(ds):
        print({k: (str(v)[:120] + "...") if isinstance(v, str) and len(str(v)) > 120 else v
               for k, v in row.items()})
        if i >= 1:
            break


def labels():
    """Download ONLY metadata.db (small) and show the MBFC publisher labels."""
    import subprocess
    import duckdb

    subprocess.run([
        "hf", "download", "ioverho/misinfo-general", "--repo-type", "dataset",
        "--include", "metadata.db", "--local-dir", "./misinfo-general",
    ], check=True)

    con = duckdb.connect("./misinfo-general/metadata.db", read_only=True)
    tables = con.execute("SHOW TABLES").fetchall()
    print("tables:", tables)
    for (t,) in tables:
        print(f"\n--- {t}")
        print(con.execute(f"DESCRIBE {t}").fetchdf())
        print(con.execute(f"SELECT * FROM {t} LIMIT 5").fetchdf())


def extract():
    """Stream one year, keep political candidates, write the candidate pool."""
    from datasets import load_dataset

    ds = load_dataset("ioverho/misinfo-general", split=YEAR, streaming=True)

    rows, seen, scanned = [], set(), 0
    for row in ds:
        scanned += 1
        if scanned > MAX_SCANNED or len(rows) >= N_CANDIDATES:
            break

        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        if len(title) < 20 or len(content) < 400:
            continue
        if not POLITICAL.search(title):
            continue

        key = title.lower()[:80]
        if key in seen:                       # syndicated near-duplicates
            continue
        seen.add(key)

        rows.append({
            "article_id": row.get("article_id"),
            "source": row.get("source"),
            "domain": row.get("domain"),
            "date": row.get("publication_date"),
            "title": title,
            "lead_150w": " ".join(content.split()[:150]),
            "url": row.get("raw_url"),
            "stratum": "",     # <-- YOU FILL THIS: A / B / C
        })

    with open(POOL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    need_shared = len(ANNOTATORS) * OVERLAP_EACH // RATERS_PER_SHARED
    need_solo = len(ANNOTATORS) * (ARTICLES_EACH - OVERLAP_EACH)
    print(f"scanned {scanned} rows -> wrote {len(rows)} candidates to {POOL}")
    print(f"\nNEXT (manual, ~15 min): open {POOL} and fill the `stratum` column:")
    print(f"  C = boundary/hard cases   -> mark at least {need_shared} of these")
    print(f"  A = expected controversial, B = expected not controversial")
    print(f"  -> mark about {need_solo} rows across A and B combined")
    print(f"Then run:  python {os.path.basename(__file__)} assign")


# ----------------------------------------------------------------------------
def _build_design(n_shared, annotators=None, raters=None, overlap_each=None, load=None):
    """Greedy balanced assignment: each shared item -> RATERS_PER_SHARED people,
    each person ends with exactly OVERLAP_EACH shared items.
    The gold-sample waves pass their own annotators/raters, overlap_each=None
    (loads may differ by 1 when the totals don't divide) and a running `load`
    so balance carries over from one wave to the next."""
    annotators = annotators or ANNOTATORS
    raters = raters or RATERS_PER_SHARED
    load = load if load is not None else Counter({a: 0 for a in annotators})
    start = dict(load)
    design = []
    for _ in range(n_shared):
        # pick the least-loaded annotators, tie-broken deterministically
        picks = sorted(annotators, key=lambda a: (load[a], a))[:raters]
        for a in picks:
            load[a] += 1
        design.append(picks)

    if annotators is not ANNOTATORS and overlap_each is None:
        got = [load[a] - start.get(a, 0) for a in annotators]
        if max(got) - min(got) > 1:
            raise SystemExit(f"Unbalanced overlap design {dict(zip(annotators, got))}")
        return design
    overlap_each = OVERLAP_EACH if overlap_each is None else overlap_each
    bad = {a: c for a, c in load.items() if c != overlap_each}
    if bad:
        raise SystemExit(
            f"Unbalanced design {bad}. Check that "
            f"len(ANNOTATORS)*OVERLAP_EACH is divisible by RATERS_PER_SHARED."
        )
    return design


def _stratum(r):
    return (r.get("stratum") or "").strip().upper()


def _pick_solo(others, n_solo):
    """Quota draw: SOLO_B_SHARE of the solo items from B, the rest from A.
    Without it the pool's A/B imbalance leaks straight into the pilot, and a
    near-all-YES sample makes alpha/kappa meaningless (prevalence paradox)."""
    n_b = round(n_solo * SOLO_B_SHARE)
    quota = {"A": n_solo - n_b, "B": n_b}
    picked, rest = [], []
    for r in others:
        s = _stratum(r)
        if quota.get(s, 0) > 0:
            quota[s] -= 1
            picked.append(r)
        else:
            rest.append(r)
    if len(picked) < n_solo:
        print(f"WARNING: quota short by {quota} — topping up from remaining rows.")
        picked += rest[:n_solo - len(picked)]
    # B first, so the round-robin deal in assign() spreads B evenly across people
    picked.sort(key=lambda r: _stratum(r) != "B")
    return picked


def assign():
    rng = random.Random(SEED)

    with open(POOL, encoding="utf-8") as f:
        pool = list(csv.DictReader(f))

    n_shared = len(ANNOTATORS) * OVERLAP_EACH // RATERS_PER_SHARED
    n_solo = len(ANNOTATORS) * (ARTICLES_EACH - OVERLAP_EACH)

    boundary = [r for r in pool if (r.get("stratum") or "").strip().upper() == "C"]
    others = [r for r in pool if (r.get("stratum") or "").strip().upper() in ("A", "B")]

    if len(boundary) < n_shared or len(others) < n_solo:
        print(f"WARNING: strata not fully filled "
              f"(C={len(boundary)} need {n_shared}, A+B={len(others)} need {n_solo}). "
              f"Falling back to unlabelled rows — fill `stratum` for a proper pilot.")
        rest = [r for r in pool if r not in boundary and r not in others]
        rng.shuffle(rest)
        while len(boundary) < n_shared and rest:
            boundary.append(rest.pop())
        while len(others) < n_solo and rest:
            others.append(rest.pop())

    rng.shuffle(boundary)
    rng.shuffle(others)
    shared = boundary[:n_shared]          # hard cases get the multiple eyes
    solo = _pick_solo(others, n_solo)

    # stable anonymous ids, shuffled so shared items aren't guessable by position
    allrows = shared + solo
    rng.shuffle(allrows)
    for i, r in enumerate(allrows, 1):
        r["item_id"] = f"P{i:03d}"

    design = _build_design(n_shared)
    assignment = defaultdict(list)
    raters_of = {}

    for item, picks in zip(shared, design):
        raters_of[item["item_id"]] = picks
        for a in picks:
            assignment[a].append(item)

    # deal solo items round-robin so each person gets a similar A/B mix
    for k, item in enumerate(solo):
        a = ANNOTATORS[k % len(ANNOTATORS)]
        raters_of[item["item_id"]] = [a]
        assignment[a].append(item)

    # ---- write per-person folders -------------------------------------------
    os.makedirs(OUTDIR, exist_ok=True)
    for a in ANNOTATORS:
        d = os.path.join(OUTDIR, a)
        os.makedirs(d, exist_ok=True)
        items = assignment[a][:]
        rng.shuffle(items)                 # hide which ones are shared
        with open(os.path.join(d, "articles.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ANNOTATOR_COLS)
            w.writeheader()
            for r in items:
                w.writerow({"item_id": r["item_id"], "title": r["title"],
                            "lead_150w": r["lead_150w"],
                            "label": "", "confidence": "", "reason": ""})
        with open(os.path.join(d, "GUIDELINE.md"), "w", encoding="utf-8") as f:
            f.write(GUIDELINE)

    # ---- admin key (NOT for annotators) -------------------------------------
    admin = os.path.join(OUTDIR, "_admin")
    os.makedirs(admin, exist_ok=True)
    with open(os.path.join(admin, "assignment_key.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "n_raters", "raters", "stratum", "source", "domain", "date", "url", "title"])
        for r in allrows:
            rs = raters_of[r["item_id"]]
            w.writerow([r["item_id"], len(rs), "|".join(rs), r.get("stratum", ""),
                        r.get("source", ""), r.get("domain", ""), r.get("date", ""),
                        r.get("url", ""), r["title"]])

    # ---- report -------------------------------------------------------------
    print(f"{len(allrows)} unique articles -> {len(ANNOTATORS)}x{ARTICLES_EACH} = "
          f"{len(ANNOTATORS)*ARTICLES_EACH} judgments")
    print(f"  shared: {n_shared} articles x {RATERS_PER_SHARED} raters")
    print(f"  solo:   {n_solo} articles x 1 rater  {dict(Counter(_stratum(r) for r in solo))}")
    for a in ANNOTATORS:
        print(f"    {a:10s} strata {dict(Counter(_stratum(r) for r in assignment[a]))}")
    print(f"\nwrote {OUTDIR}/<name>/articles.csv for: {', '.join(ANNOTATORS)}")
    print(f"admin key (do NOT share): {admin}/assignment_key.csv")

    pair_cov = Counter()
    for picks in design:
        for p in combinations(sorted(picks), 2):
            pair_cov[p] += 1
    missing = [p for p in combinations(sorted(ANNOTATORS), 2) if p not in pair_cov]
    print(f"\npairwise coverage: {len(pair_cov)}/{len(list(combinations(ANNOTATORS,2)))} pairs share >=1 article")
    if missing:
        print("  pairs with NO shared article (cannot compute their agreement):")
        for p in missing:
            print("   ", p)


# ----------------------------------------------------------------------------
def merge():
    rows = []
    for a in ANNOTATORS:
        p = os.path.join(OUTDIR, a, "articles.csv")
        if not os.path.exists(p):
            print(f"missing {p} — skipping")
            continue
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab = (r.get("label") or "").strip().upper()
                if lab:
                    rows.append({"annotator": a, "item_id": r["item_id"],
                                 "label": lab, "conf": (r.get("confidence") or "").strip(),
                                 "reason": (r.get("reason") or "").strip()})

    by_item = defaultdict(dict)
    for r in rows:
        by_item[r["item_id"]][r["annotator"]] = r["label"]

    shared = {k: v for k, v in by_item.items() if len(v) > 1}
    print(f"{len(rows)} judgments, {len(by_item)} items, {len(shared)} with >1 rater\n")

    print("=== shared items ===")
    split_items = []
    for item, labs in sorted(shared.items()):
        c = Counter(labs.values())
        flag = "UNANIMOUS" if len(c) == 1 else "*** SPLIT ***"
        print(f"{item}  {dict(c)}  {flag}   {labs}")
        if len(c) > 1:
            split_items.append(item)

    print("\n=== pairwise agreement (co-annotated items only) ===")
    for a, b in combinations(ANNOTATORS, 2):
        both = [(labs[a], labs[b]) for labs in by_item.values() if a in labs and b in labs]
        if both:
            agree = sum(x == y for x, y in both)
            print(f"{a:10s} vs {b:10s}  {agree}/{len(both)}")

    print("\n=== prevalence (all judgments, purposive sample — NOT a corpus estimate) ===")
    print(dict(Counter(r["label"] for r in rows)))

    try:
        import numpy as np
        import krippendorff
        items = sorted(by_item)
        idx = {k: i for i, k in enumerate(items)}
        m = np.full((len(ANNOTATORS), len(items)), np.nan)
        for r in rows:
            m[ANNOTATORS.index(r["annotator"]), idx[r["item_id"]]] = 1.0 if r["label"] == "YES" else 0.0
        alpha = krippendorff.alpha(reliability_data=m, level_of_measurement="nominal")
        print(f"\nKrippendorff alpha = {alpha:.3f}")
        print(f"  CAVEAT: computed from only {len(shared)} multiply-annotated items.")
        print("  Report it as a smoke test with that n stated, never as an estimate.")
    except ImportError:
        print("\n(pip install krippendorff numpy  -> to also get alpha)")

    if split_items:
        print(f"\nTALK ABOUT THESE TOMORROW: {', '.join(split_items)}")
        print("Pull their titles from _admin/assignment_key.csv and the free-text")
        print("`reason` fields from each annotator — that is your best slide.")


GUIDELINE = """# Annotation guideline — pilot

**Question:** Does this article concern an issue on which the relevant public
substantively disagrees?

- **label**: YES or NO
- **confidence**: low / medium / high
- **reason**: one short line. REQUIRED, especially when you hesitate.

## Reminders
- Controversy is a property of the **issue and its reception**, not of the
  article's wording.
- A neutrally written article on a contested issue -> **YES**.
- A strongly worded, biased article on a settled issue -> **NO**.
- You are not judging whether the article is good, true, or fair.
- "Relevant public" = the audience the outlet writes for.

## Rules
- Read the **title + the 150-word lead only**. Nothing else. Do not look the
  article up online.
- Work **alone**. No discussion until everyone has submitted.
- If you are stuck for more than ~90 seconds, pick your best guess, set
  confidence to low, and say why in `reason`. The hesitation is the data.
"""


def _gold(module):
    """Lazy entry point into the gold-sample pipeline (gold/<module>.py)."""
    def run():
        import importlib
        importlib.import_module(f"gold.{module}").main()
    return run


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    {"inspect": inspect, "labels": labels, "extract": extract,
     "assign": assign, "merge": merge,
     "content-stats": _gold("content_stats"), "reddit-fetch": _gold("reddit_fetch"),
     "wiki": _gold("wiki"), "candidates": _gold("candidates"), "proxies": _gold("proxies"),
     "strata": _gold("strata"), "wave": _gold("waves"),
     "catchup": _gold("assign")}[cmd]()
