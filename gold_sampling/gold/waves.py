"""Step 4: draw one wave -> gold_sample/sample_wave{k}.csv (internal, with weights).

Random part: n_random articles fixed once by a seeded permutation of the whole
universe (random_reserve), n_random / n_waves per wave, never rebalanced and
never skipped by a cap (an exact SRS). Stratified part: the wave's quota from
  wave 1   data/allocation_waves.csv (the Step 3 plan)
  wave k>1 data/allocation_wave{k}.csv written by rebalance.py, else the plan.
Every frame here is the sampling pool (config allocation.sampling_pool, e.g.
only Reddit-matched articles), so N and pi below refer to that pool.
Articles drawn in earlier waves are excluded from every frame, so nothing is
re-drawn or replaced. Caps (whole sample): <= publisher_cap_share * n_total per
publisher, <= max_per_topic_month_publisher per (topic, month, publisher). A
stratum that runs dry passes its shortfall to the nearest stratum (logged).

Inclusion probabilities:
  pi_random      = n_random / N                     (design level)
  pi_strat_wave  = n_hk / N_hk  for the article's stratum h in wave k
  pi_strat       = sum over the waves drawn so far   (recomputed every wave)
  pi_total       = 1 - (1 - pi_random) * (1 - pi_strat);  weight = 1 / pi_total
"""

import glob
import os
from collections import Counter

import numpy as np
import pandas as pd

from .common import load_config, p, start_run, sub_seed
from .strata import _neighbours, sampling_pool

SAMPLE_COLS = ["article_id", "wave", "selection", "stratum", "stratum_drawn_for", "flag", "matched",
               "year", "bias3", "reliability_pool", "source", "topic_id", "month",
               "pi_random", "pi_strat_wave", "pi_strat", "pi_total", "weight"]


def load_frame(cfg):
    """Candidates x strata, restricted to the sampling pool (config)."""
    cand = pd.read_parquet(p(cfg, "candidates"))
    fr = pd.read_parquet(os.path.join(p(cfg, "data_dir"), "strata_frame.parquet"))
    return sampling_pool(cfg, cand.merge(fr, on="article_id", validate="1:1"))


def sample_path(cfg, k):
    return os.path.join(p(cfg, "sample_dir"), f"sample_wave{k}.csv")


def previous(cfg, k):
    fs = [sample_path(cfg, j) for j in range(1, k) if os.path.exists(sample_path(cfg, j))]
    if len(fs) != k - 1:
        raise SystemExit(f"wave {k} needs waves 1..{k - 1} drawn first")
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame(columns=SAMPLE_COLS)


def wave_allocation(cfg, k):
    d = p(cfg, "data_dir")
    reb = os.path.join(d, f"allocation_wave{k}.csv")
    if os.path.exists(reb):
        return pd.read_csv(reb), "rebalanced"
    plan = pd.read_csv(os.path.join(d, "allocation_waves.csv"))
    return plan[plan.wave == k][["stratum", "n"]], "plan"


def random_reserve(cfg, frame):
    """The whole random part (n_random articles), fixed once from one seeded
    permutation of the universe; wave k takes its slice in order. Fixing it up
    front keeps it an exact SRS (no cap ever skips a random pick), while the
    stratified draws exclude these ids and count them against the caps.
    The seeded draw has `random_draw_size` articles (default n_random) and the
    reserve is its first n_random: `choice` returns them in random order, so a
    prefix is still an SRS, and shrinking n_random keeps the earlier picks."""
    ac = cfg["allocation"]
    ids = frame.article_id.sort_values().to_numpy()
    size = max(ac.get("random_draw_size", ac["n_random"]), ac["n_random"])
    pick = np.random.default_rng(sub_seed(cfg, "random_part")).choice(len(ids), size, replace=False)
    reserved = list(ids[pick[:ac["n_random"]]])
    W, n = ac["n_waves"], ac["n_random"]
    sizes = [n // W + (1 if w < n % W else 0) for w in range(W)]
    starts = np.cumsum([0] + sizes)
    return reserved, {w + 1: reserved[starts[w]:starts[w + 1]] for w in range(W)}


def draw(cfg, frame, alloc, k, prev, log):
    """Pure function: returns the wave-k sample rows (no IO)."""
    ac = cfg["allocation"]
    pub_cap = int(np.floor(ac["publisher_cap_share"] * ac["n_total"]))
    tmp_cap = ac["max_per_topic_month_publisher"]
    reserved, by_wave = random_reserve(cfg, frame)
    for j, d in prev[prev.selection == "random"].groupby("wave"):
        if set(d.article_id) != set(by_wave.get(int(j), [])):
            raise SystemExit(f"wave {j}: drawn random articles differ from the seeded reserve; "
                             f"check allocation.n_random / random_draw_size / n_waves")
    res_rows = frame[frame.article_id.isin(reserved)]
    # caps count the whole random reserve from wave 1 on, plus everything drawn
    pubs = Counter(prev[prev.selection != "random"].source) + Counter(res_rows.source)
    tmp = Counter(zip(prev[prev.selection != "random"].topic_id, prev[prev.selection != "random"].month,
                      prev[prev.selection != "random"].source)) + \
        Counter(zip(res_rows.topic_id, res_rows.month, res_rows.source))
    taken = set(prev.article_id) | set(reserved)
    avail = frame[~frame.article_id.isin(taken)]
    rows = []

    def ok(r):
        return pubs[r.source] < pub_cap and tmp[(r.topic_id, r.month, r.source)] < tmp_cap

    def take(r, selection, drawn_for, count=True):
        if count:
            pubs[r.source] += 1
            tmp[(r.topic_id, r.month, r.source)] += 1
        taken.add(r.article_id)
        rows.append({**{c: getattr(r, c) for c in ["article_id", "stratum", "flag", "matched", "year", "bias3",
                                                    "reliability_pool", "source", "topic_id", "month"]},
                     "wave": k, "selection": selection, "stratum_drawn_for": drawn_for})

    # ---- random part: this wave's slice of the reserve (already counted) ---
    for _, r in res_rows.set_index("article_id", drop=False).loc[by_wave[k]].iterrows():
        take(r, "random", "", count=False)
    over = {s: c for s, c in Counter(res_rows.source).items() if c > pub_cap}
    if over:
        log.warning(f"random reserve alone exceeds the publisher cap: {over}")

    # ---- stratified part ----------------------------------------------------
    strat = avail[avail.stratifiable]
    groups = {s: g for s, g in strat.groupby("stratum")}
    N_h = {s: len(g) for s, g in groups.items()}
    queue = [(s, int(n), s) for s, n in alloc.sort_values("stratum").itertuples(index=False) if n > 0]
    perms = {}
    n_drawn = Counter()
    while queue:
        s, n, origin = queue.pop(0)
        g = groups.get(s)
        if g is not None and s not in perms:
            perms[s] = list(np.random.default_rng(sub_seed(cfg, f"wave{k}:{s}")).permutation(len(g)))
        while n and g is not None and perms[s]:
            r = g.iloc[perms[s].pop(0)]
            if r.article_id in taken or not ok(r):
                continue
            take(r, "stratified", origin)
            n_drawn[s] += 1
            n -= 1
        if n:
            nxt = next((nb for nb in _neighbours(s, ac) if nb in groups and
                        (nb not in perms or perms[nb])), None)
            if nxt is None:
                raise SystemExit(f"wave {k}: cannot place {n} units from {s}")
            log.info(f"wave {k}: stratum {s} short by {n} -> {nxt}")
            queue.insert(0, (nxt, n, origin))

    out = pd.DataFrame(rows)
    out["pi_random"] = ac["n_random"] / len(frame)
    out["pi_strat_wave"] = [n_drawn[s] / N_h[s] if s in N_h else 0.0 for s in out.stratum]
    return out


def recompute_weights(cfg, frame, samples):
    """Cumulative stratified inclusion probability over all waves drawn so far.
    For wave j, stratum h: n_hj / N_hj, with N_hj the stratum frame at wave j."""
    done = pd.concat(samples, ignore_index=True)
    reserved = set(random_reserve(cfg, frame)[0])
    st = done[done.selection == "stratified"].groupby(["wave", "stratum"]).size().rename("n").reset_index()
    pi = Counter()
    for j, sj in st.groupby("wave"):
        before = set(done.article_id[done.wave < j]) | reserved
        fr = frame[frame.stratifiable & ~frame.article_id.isin(before)]
        N = fr.stratum.value_counts()
        for r in sj.itertuples():
            pi[r.stratum] += r.n / N.get(r.stratum, np.inf)
    out = []
    for s in samples:
        s = s.copy()
        s["pi_random"] = cfg["allocation"]["n_random"] / len(frame)
        s["pi_strat"] = s.stratum.map(lambda h: pi.get(h, 0.0))
        s.loc[~s.stratum.isin(frame.stratum[frame.stratifiable].unique()), "pi_strat"] = 0.0
        s["pi_total"] = 1 - (1 - s.pi_random) * (1 - s.pi_strat)
        s["weight"] = 1 / s.pi_total
        out.append(s[SAMPLE_COLS])
    return out


def run(cfg, log, k, frame=None):
    frame = load_frame(cfg) if frame is None else frame
    if os.path.exists(sample_path(cfg, k)):
        raise SystemExit(f"{sample_path(cfg, k)} exists; waves are never re-drawn")
    prev = previous(cfg, k)
    alloc, src = wave_allocation(cfg, k)
    log.info(f"wave {k}: allocation from {src}: {int(alloc.n.sum())} stratified units")
    new = draw(cfg, frame, alloc, k, prev, log)
    samples = [pd.read_csv(sample_path(cfg, j)) for j in range(1, k)] + [new]
    samples = recompute_weights(cfg, frame, samples)
    for j, s in enumerate(samples, 1):
        s.to_csv(sample_path(cfg, j), index=False)
    log.info(f"wave {k}: {len(new)} articles ({new.selection.value_counts().to_dict()}), "
             f"flag {new.flag.value_counts().to_dict()}, matched {new.matched.value_counts().to_dict()}")
    return samples[-1]


def main():
    import sys
    cfg = load_config()
    k = int(sys.argv[2] if sys.argv[1] == "wave" else sys.argv[1])
    log = start_run(cfg, f"wave{k}")
    s = run(cfg, log, k)
    from . import assign
    assign.run(cfg, log, k, s)


if __name__ == "__main__":
    main()
