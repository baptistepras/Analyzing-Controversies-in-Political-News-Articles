"""Step 3: proxy flag, strata and the (not yet drawn) allocation plan.

flag  = H (likely controversial / high signal) if any rule in config.proxy_flag
        fires, else L. Reddit rules can only fire for matched articles.
stratum = flag | year | bias3 | pool | matched   (e.g. "H|2020|Right|unreliable|m1")

Allocation of the stratified part (config.allocation):
  flag share -> equal over year x bias cells -> pool split -> Reddit-matched
  share (non-2018 only; 2018 has no URLs so its cells are all m0).
Fractional targets are rounded with largest remainders (seeded tie-break),
then any stratum whose frame is too small gives its shortfall to the nearest
stratum (matched -> pool -> nearest year -> bias -> flag); every move is logged.
The plan is split into waves by dealing the sorted stratum units round-robin,
so every wave carries the same flag / year / bias / pool / matched mix.
Unknown-bias publishers are not stratified: they enter only through the
random part.
"""

import itertools
import os

import numpy as np
import pandas as pd

from .common import load_config, p, start_run, sub_seed

DIMS = ["flag", "year", "bias3", "reliability_pool", "matched"]


def proxy_flag(df, fc):
    """Return (flag Series of 'H'/'L', DataFrame of the individual rules)."""
    r = pd.DataFrame(index=df.index)
    r["wiki_topic"] = df.wiki_topic_n_issue >= fc["wiki_topic_min_pages"]
    r["wiki_title"] = df.wiki_title_n_issue >= fc["wiki_title_min_pages"]
    m = df.reddit_matched.fillna(False).astype(bool)
    r["reddit_low_ratio"] = m & (df.reddit_upvote_ratio <= fc["reddit_max_upvote_ratio"]) & \
        (df.reddit_comments_max >= fc["reddit_min_comments"])
    r["reddit_disagreement"] = m & (df.reddit_disagreement >= fc["reddit_min_disagreement"])
    r["reddit_discussion"] = m & (df.reddit_comment_ratio >= fc["reddit_min_comment_ratio"]) & \
        (df.reddit_comments_max >= fc["reddit_min_comments"])
    use = [k for k in r.columns if k in fc["rules"]]
    flag = np.where(r[use].any(axis=1), "H", "L")
    return pd.Series(flag, index=df.index), r


def build_frame(cfg, log):
    cand = pd.read_parquet(p(cfg, "candidates"))
    prox = pd.read_parquet(p(cfg, "proxies"))
    df = cand.merge(prox, on="article_id", how="left", validate="1:1")
    df["flag"], rules = proxy_flag(df, cfg["proxy_flag"])
    for c in rules:
        df["rule_" + c] = rules[c]
    df["matched"] = np.where(df.reddit_matched.fillna(False), "m1", "m0")
    df["stratum"] = df[DIMS].astype(str).agg("|".join, axis=1)
    ac = cfg["allocation"]
    df["stratifiable"] = df.bias3.isin(ac["bias_levels"]) & df.year.isin(ac["years"])
    log.info(f"flag: {df.flag.value_counts().to_dict()}; rules fired: "
             f"{ {c: int(rules[c].sum()) for c in rules} }; stratifiable {df.stratifiable.mean():.1%}")
    return df


def sampling_pool(cfg, frame):
    """The frame every draw comes from (config allocation.sampling_pool):
    reddit_matched = the distant-supervision pool (articles linked from the
    subreddits, m1); all = the whole candidate universe."""
    pool = cfg["allocation"].get("sampling_pool", "all")
    if pool == "all":
        return frame
    if pool == "reddit_matched":
        return frame[frame.matched == "m1"]
    raise SystemExit(f"unknown allocation.sampling_pool {pool!r}")


ROUND_ORDER =["flag", "reliability_pool", "bias3", "year", "matched"]


def _round(targets, total, rng):
    """Systematic (cumulative) rounding with a seeded random start, over cells
    sorted flag -> pool -> bias -> year -> matched: every nested margin (flag,
    flag x pool, flag x pool x bias, ...) stays within +-1 of its target."""
    keys = pd.Series(targets.index, index=targets.index).str.split("|", expand=True)
    keys.columns = DIMS
    order = keys.sort_values(ROUND_ORDER).index
    t = targets[order] * total / targets.sum()
    cum = np.cumsum(t.to_numpy()) + rng.random()
    ints = np.diff(np.concatenate([[0], np.floor(cum)])).astype(int)
    out = pd.Series(ints, index=order)
    out.iloc[-1] += int(total - out.sum())
    return out.reindex(targets.index)


def targets(cfg, n=None, high_share=None, cap=None, iters=500):
    """Fractional target per stratum by iterative proportional fitting.

    Margins (what "balanced" means here):
      flag x year, flag x bias3, flag x pool : flag share split equally
      year x matched                          : matched_share of each URL year, 0 for 2018
    Strata with an empty frame (cap == 0, e.g. Center x unreliable, which MBFC
    never produces) are structural zeros; IPF lets the other cells absorb their
    share, so the margins stay balanced instead of piling onto a neighbour."""
    ac = cfg["allocation"]
    n = ac["n_total"] - ac["n_random"] if n is None else n
    hs = ac["high_share"] if high_share is None else high_share
    t = pd.DataFrame(list(itertools.product(["H", "L"], ac["years"], ac["bias_levels"], ac["pools"], ["m1", "m0"])),
                     columns=DIMS)
    t.index = t.astype(str).agg("|".join, axis=1)
    t["x"] = 1.0
    t.loc[t.year.isin(ac["no_url_years"]) & (t.matched == "m1"), "x"] = 0.0
    if cap is not None:
        t.loc[cap.reindex(t.index, fill_value=0) == 0, "x"] = 0.0
    fshare = {"H": hs, "L": 1 - hs}
    ny, nb, npool = len(ac["years"]), len(ac["bias_levels"]), len(ac["pools"])

    def year_matched(r):
        ms = 0.0 if r.year in ac["no_url_years"] else ac["matched_share"]
        return n / ny * (ms if r.matched == "m1" else 1 - ms)

    margins = [
        (["flag", "year"], lambda r: n * fshare[r.flag] / ny),
        (["flag", "bias3"], lambda r: n * fshare[r.flag] / nb),
        (["flag", "reliability_pool"], lambda r: n * fshare[r.flag] / npool),
        (["year", "matched"], year_matched),
    ]
    goals = []
    for g, f in margins:
        keys = t[g].astype(str).agg("|".join, axis=1)
        want = np.array([f(r) for r in t[g].itertuples(index=False)], dtype=float)
        goals.append((keys, want))
    for _ in range(iters):
        worst = 0.0
        for keys, want in goals:
            cur = t.x.groupby(keys).transform("sum").to_numpy()
            ok = cur > 0
            worst = max(worst, float(np.abs(cur[ok] - want[ok]).max()))
            x = t.x.to_numpy()
            x[ok] *= want[ok] / cur[ok]
            t["x"] = x
        if worst < 1e-6:
            break
    return t.x * n / t.x.sum()


def _neighbours(key, ac):
    """Strata ordered by distance: matched -> pool -> nearest year -> bias -> flag."""
    flag, year, bias, pool, m = key.split("|")
    year = int(year)
    years = sorted(ac["years"], key=lambda y: (abs(y - year), y))
    biases = sorted(ac["bias_levels"], key=lambda b: (abs(ac["bias_levels"].index(b) -
                                                          ac["bias_levels"].index(bias)), b))
    pools = [pool] + [x for x in ac["pools"] if x != pool]
    ms = [m] + [x for x in ("m1", "m0") if x != m]
    flags = [flag] + [x for x in ("H", "L") if x != flag]
    for f in flags:
        for b in biases:
            for y in years:
                for pl in pools:
                    for mm in ms:
                        k = f"{f}|{y}|{b}|{pl}|{mm}"
                        if k != key:
                            yield k


def plan(cfg, log, frame, already=None, n=None, high_share=None, seed_name="allocation"):
    """Integer allocation per stratum, after shortfall redistribution."""
    ac = cfg["allocation"]
    rng = np.random.default_rng(sub_seed(cfg, seed_name))
    elig = frame[frame.stratifiable]
    if already is not None:
        elig = elig[~elig.article_id.isin(already)]
    cap = elig.stratum.value_counts()
    t = targets(cfg, n, high_share, cap=cap)
    alloc = _round(t, round(t.sum()), rng)
    cap = cap.reindex(alloc.index, fill_value=0)
    moves = []
    for k in alloc.index:
        short = alloc[k] - cap[k]
        if short <= 0:
            continue
        alloc[k] = cap[k]
        for nb in _neighbours(k, ac):
            room = cap.get(nb, 0) - alloc.get(nb, 0)
            if room <= 0:
                continue
            give = min(room, short)
            alloc[nb] = alloc.get(nb, 0) + give
            moves.append({"from_stratum": k, "to_stratum": nb, "n": int(give), "reason": f"frame has {int(cap[k])}"})
            short -= give
            if short == 0:
                break
        if short:
            raise SystemExit(f"cannot place {short} units from {k}")
    mv = pd.DataFrame(moves, columns=["from_stratum", "to_stratum", "n", "reason"])
    for r in mv.itertuples():
        log.info(f"redistributed {r.n} from {r.from_stratum} -> {r.to_stratum} ({r.reason})")
    res = pd.DataFrame({"target": t, "n_alloc": alloc, "N_frame": cap})
    res = res[(res.target > 0) | (res.n_alloc > 0)]
    res[DIMS] = res.index.to_series().str.split("|", expand=True).values
    return res, mv


def waves(cfg, res):
    """Deal stratum units round-robin over waves -> DataFrame[stratum, wave, n]."""
    ac = cfg["allocation"]
    W = ac["n_waves"]
    order = res.sort_values(ROUND_ORDER).index
    units = [k for k in order for _ in range(int(res.n_alloc[k]))]
    size = [len(units) // W + (1 if w < len(units) % W else 0) for w in range(W)]
    # greedy: each unit goes to the (non-full) wave holding the fewest units that
    # share its flag, flag x pool, bias, year and matched values -> every wave
    # keeps the same proportions on all design margins
    # flag balance matters most, then flag x pool, then the rest
    weight = {"f": 100, "fp": 10, "b": 1, "y": 1, "m": 1}
    feats = lambda k: (lambda f, y, b, pl, m: [("f", f), ("fp", f, pl), ("b", b), ("y", y), ("m", m)])(*k.split("|"))
    cnt = [dict() for _ in range(W)]
    out = []
    for k in units:
        fs = feats(k)
        cands = [w for w in range(W) if sum(1 for _, wv in out if wv == w) < size[w]]
        w = min(cands, key=lambda w: (sum(weight[x[0]] * cnt[w].get(x, 0) for x in fs), w))
        for x in fs:
            cnt[w][x] = cnt[w].get(x, 0) + 1
        out.append((k, w))
    df = pd.DataFrame(out, columns=["stratum", "wave"])
    df["wave"] += 1
    return df.groupby(["stratum", "wave"]).size().rename("n").reset_index()


def main():
    cfg = load_config()
    log = start_run(cfg, "strata")
    frame = build_frame(cfg, log)
    out = os.path.join(p(cfg, "data_dir"), "strata_frame.parquet")
    frame[["article_id", "stratum", "flag", "matched", "stratifiable"] +
          [c for c in frame.columns if c.startswith("rule_")]].to_parquet(out, index=False)
    pool = sampling_pool(cfg, frame)
    log.info(f"sampling pool {cfg['allocation'].get('sampling_pool', 'all')}: {len(pool):,} of {len(frame):,} "
             f"candidates, {pool.source.nunique()} publishers")
    res, mv = plan(cfg, log, pool)
    res.to_csv(os.path.join(p(cfg, "data_dir"), "allocation_plan.csv"))
    mv.to_csv(os.path.join(p(cfg, "data_dir"), "allocation_moves.csv"), index=False)
    wv = waves(cfg, res)
    wv.to_csv(os.path.join(p(cfg, "data_dir"), "allocation_waves.csv"), index=False)
    ac = cfg["allocation"]
    log.info(f"stratified units {res.n_alloc.sum()} + random {ac['n_random']}; "
             f"per wave {wv.groupby('wave').n.sum().to_dict()} (+{ac['n_random'] // ac['n_waves']} random each)")
    for d in DIMS:
        log.info(f"by {d}: {res.groupby(d).n_alloc.sum().to_dict()}")


if __name__ == "__main__":
    main()
