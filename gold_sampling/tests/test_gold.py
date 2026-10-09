"""Tests for the gold-sample pipeline.

Draw / assignment logic runs on a synthetic frame (no network, no real draw);
checks on the real outputs run only when those files exist.
"""

import copy
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gold import assign, strata, waves  # noqa: E402
from gold.common import load_config  # noqa: E402
from gold.urlnorm import normalize_url  # noqa: E402

LOG = logging.getLogger("test")
PROXY_WORDS = ("stratum", "flag", "reddit", "wiki", "proxy", "weight", "pi_", "matched", "source",
               "domain", "url", "bias", "reliab", "selection")


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["paths"]["sample_dir"] = str(tmp_path / "gold_sample")
    c["paths"]["data_dir"] = str(tmp_path / "data")
    c["paths"]["returned_dir"] = str(tmp_path / "gold_sample" / "returnedW{k}")
    os.makedirs(c["paths"]["sample_dir"])
    os.makedirs(c["paths"]["data_dir"])
    return c


@pytest.fixture
def frame(cfg, universe):
    """What the draw sees: the universe restricted to the sampling pool (as waves.load_frame does)."""
    return strata.sampling_pool(cfg, universe)


@pytest.fixture
def universe(cfg):
    rng = np.random.default_rng(0)
    n = 6000
    years = rng.choice(cfg["allocation"]["years"], n)
    df = pd.DataFrame({
        "article_id": [f"{y}-{i:07d}" for i, y in enumerate(years)],
        "year": years,
        "flag": rng.choice(["H", "L"], n, p=[0.3, 0.7]),
        "bias3": rng.choice(["Left", "Center", "Right", "Unknown"], n, p=[0.4, 0.15, 0.35, 0.1]),
        "reliability_pool": rng.choice(["reliable", "unreliable"], n, p=[0.7, 0.3]),
        "source": [f"pub{i}" for i in rng.integers(0, 60, n)],
        "topic_id": rng.integers(0, 40, n),
        "month": rng.choice(["01", "06"], n),
    })
    # one publisher owns 80% of Center (~12% overall): the ~53 stratified Center units
    # would put ~40 articles on it without the cap, while a pure random 40 expects ~5
    df.loc[(df.bias3 == "Center") & (rng.random(n) < 0.8), "source"] = "bigpub"
    df["matched"] = np.where((df.year != 2018) & (rng.random(n) < 0.3), "m1", "m0")
    df["stratum"] = df[strata.DIMS].astype(str).agg("|".join, axis=1)
    df["stratifiable"] = df.bias3 != "Unknown"
    return df


def draw_all(cfg, frame):
    res, _ = strata.plan(cfg, LOG, frame)
    wv = strata.waves(cfg, res)
    samples = []
    for k in range(1, cfg["allocation"]["n_waves"] + 1):
        prev = pd.concat(samples, ignore_index=True) if samples else pd.DataFrame(columns=waves.SAMPLE_COLS)
        alloc = wv[wv.wave == k][["stratum", "n"]]
        samples.append(waves.draw(cfg, frame, alloc, k, prev, LOG))
    return waves.recompute_weights(cfg, frame, samples)


# ---------------------------------------------------------------- draw logic
def test_no_duplicate_article_ids(cfg, frame):
    s = pd.concat(draw_all(cfg, frame))
    assert len(s) == cfg["allocation"]["n_total"]
    assert not s.article_id.duplicated().any()


def test_publisher_and_topic_caps(cfg, frame):
    s = pd.concat(draw_all(cfg, frame))
    ac = cfg["allocation"]
    assert s.source.value_counts().max() <= int(ac["publisher_cap_share"] * ac["n_total"])
    assert s.groupby(["topic_id", "month", "source"]).size().max() <= ac["max_per_topic_month_publisher"]


def test_seed_reproducibility(cfg, frame):
    a = pd.concat(draw_all(cfg, frame)).reset_index(drop=True)
    b = pd.concat(draw_all(cfg, frame)).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    cfg2 = copy.deepcopy(cfg)
    cfg2["seed"] += 1
    c = pd.concat(draw_all(cfg2, frame))
    assert set(c.article_id) != set(a.article_id)


def test_sampling_pool_is_reddit_matched(cfg, universe, frame):
    """Every drawn article (random part included) has a Reddit link, and the
    random inclusion probability is taken over the pool, not the universe."""
    assert cfg["allocation"]["sampling_pool"] == "reddit_matched"
    assert (universe.matched == "m0").any() and (frame.matched == "m1").all()
    s = pd.concat(draw_all(cfg, frame))
    assert (s.matched == "m1").all()
    assert np.allclose(s.pi_random, cfg["allocation"]["n_random"] / len(frame))


def test_weights_and_parts(cfg, frame):
    s = pd.concat(draw_all(cfg, frame))
    assert (s.selection == "random").sum() == cfg["allocation"]["n_random"]
    assert ((s.pi_total > 0) & (s.pi_total <= 1)).all()
    assert np.allclose(s.weight, 1 / s.pi_total)
    # unknown-bias articles can only come from the random part
    assert (s[s.bias3 == "Unknown"].selection == "random").all()


# ---------------------------------------------------------------- annotation files
def test_annotation_files_have_no_proxy_columns(cfg, frame):
    s = draw_all(cfg, frame)[0]
    texts = pd.DataFrame({"article_id": s.article_id, "title": "t", "lead_150w": "lead"})
    files, key = assign.assign(cfg, LOG, 1, s, texts)
    for a, df in files.items():
        assert list(df.columns) == assign.BLIND_COLS
        assert not any(w in c.lower() for c in df.columns for w in PROXY_WORDS)
        assert not df.isin(set(s.stratum)).any().any()
    # overlap items reach several raters, every article reaches at least one
    assert key.n_raters.min() >= 1 and key[key.overlap].n_raters.min() >= 2
    assert key.article_id.nunique() == len(s)


# ---------------------------------------------------------------- real outputs (when present)
REAL_SAMPLES = sorted(glob.glob(os.path.join(ROOT, "gold_sample", "sample_wave*.csv")))
REAL_ANN = sorted(glob.glob(os.path.join(ROOT, "gold_sample", "annotation", "annotation_wave*_*.csv")))


@pytest.mark.skipif(not REAL_SAMPLES, reason="no wave drawn yet")
def test_real_samples():
    cfg = load_config()
    ac = cfg["allocation"]
    s = pd.concat([pd.read_csv(f) for f in REAL_SAMPLES])
    assert not s.article_id.duplicated().any()
    assert s.source.value_counts().max() <= int(ac["publisher_cap_share"] * ac["n_total"])
    assert s.groupby(["topic_id", "month", "source"]).size().max() <= ac["max_per_topic_month_publisher"]


@pytest.mark.skipif(not REAL_ANN, reason="no annotation files yet")
def test_real_annotation_files_blind():
    for f in REAL_ANN:
        cols = list(pd.read_csv(f, nrows=0).columns)
        assert cols == assign.BLIND_COLS, f


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "candidates.parquet")), reason="no candidates yet")
def test_candidates_table():
    cfg = load_config()
    c = pd.read_parquet(os.path.join(ROOT, "candidates.parquet"))
    assert not c.article_id.duplicated().any()
    assert "content" not in c.columns and "lead_150w" not in c.columns
    assert c.n_words_text.between(cfg["filters"]["min_words"], cfg["filters"]["max_words"]).all()
    assert (c[c.year == 2018].url_status == "missing_2018").all()
    assert c[c.year == 2018].raw_url.isna().all()


# ---------------------------------------------------------------- helpers
@pytest.mark.parametrize("raw,key", [
    ("https://www.cbsnews.com/news/x/?ftag=CNM-00", "cbsnews.com/news/x"),
    ("https://amp.theguardian.com/us-news/2020/x", "theguardian.com/us-news/2020/x"),
    ("https://www.google.com/amp/s/www.cnn.com/2020/x/index.html", "cnn.com/2020/x"),
    ("https://example.com/?p=123&utm_source=tw", "example.com?p=123"),
    ("N/A", None),
])
def test_urlnorm(raw, key):
    assert normalize_url(raw) == key


# ---------------------------------------------------------------- rebalance.py end to end
def test_rebalance_shifts_quota_toward_minority(cfg, frame, tmp_path):
    import subprocess
    import yaml
    d = cfg["paths"]["data_dir"]
    for k in ("candidates", "proxies", "report"):
        cfg["paths"][k] = str(tmp_path / f"{k}.x")
    cfg["paths"]["logs_dir"] = str(tmp_path / "logs")
    frame.drop(columns=["stratum", "flag", "matched", "stratifiable"]).to_parquet(cfg["paths"]["candidates"])
    frame[["article_id", "stratum", "flag", "matched", "stratifiable"]].to_parquet(
        os.path.join(d, "strata_frame.parquet"))
    res, _ = strata.plan(cfg, LOG, frame)
    strata.waves(cfg, res).to_csv(os.path.join(d, "allocation_waves.csv"), index=False)
    s1 = waves.recompute_weights(cfg, frame, [waves.draw(cfg, frame, strata.waves(cfg, res).query("wave == 1")[["stratum", "n"]], 1,
                                                        pd.DataFrame(columns=waves.SAMPLE_COLS), LOG)])[0]
    s1.to_csv(waves.sample_path(cfg, 1), index=False)
    texts = pd.DataFrame({"article_id": s1.article_id, "title": "t", "lead_150w": "l"})
    files, key = assign.assign(cfg, LOG, 1, s1, texts)
    os.makedirs(assign.ann_dir(cfg)), os.makedirs(assign.admin_dir(cfg))
    key.to_csv(assign.key_path(cfg, 1), index=False)
    flag = dict(zip(key.item_id, key.flag))
    for a, df in files.items():
        # H mostly YES, L often NO -> NO is the minority and is more frequent in L
        df["label"] = ["YES" if flag[i] == "H" or j % 3 else "NO" for j, i in enumerate(df.item_id)]
        df.to_csv(assign.ann_path(cfg, 1, a), index=False)
    cpath = tmp_path / "config.yaml"
    cpath.write_text(yaml.safe_dump(cfg))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "rebalance.py"), "--after-wave", "1"],
                       cwd=ROOT, env={**os.environ, "GOLD_CONFIG": str(cpath)}, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "minority class = NO" in r.stdout
    nxt = pd.read_csv(os.path.join(d, "allocation_wave2.csv"))
    assert nxt.n.sum() == (cfg["allocation"]["n_total"] - cfg["allocation"]["n_random"]) // cfg["allocation"]["n_waves"]
    assert nxt[nxt.stratum.str[0] == "H"].n.sum() < nxt.n.sum() / 2


def test_pilot_design_unchanged():
    """The pilot's `assign` path (defaults) still yields the exact 5x3 design."""
    import extract_pilot_articles as pilot
    from collections import Counter
    d = pilot._build_design(5)
    assert d == [["baptiste", "ksenia", "martin"], ["raphael", "vladimir", "baptiste"],
                 ["ksenia", "martin", "raphael"], ["vladimir", "baptiste", "ksenia"],
                 ["martin", "raphael", "vladimir"]]
    assert set(Counter(a for p in d for a in p).values()) == {pilot.OVERLAP_EACH}


def test_random_part_is_the_seeded_reserve(cfg, frame):
    """The 40 random articles are exactly the seeded reserve: no cap or
    stratified pick can displace them, so the random part stays an exact SRS."""
    s = pd.concat(draw_all(cfg, frame))
    reserved, by_wave = waves.random_reserve(cfg, frame)
    assert set(s[s.selection == "random"].article_id) == set(reserved)
    for k, ids in by_wave.items():
        assert set(s[(s.selection == "random") & (s.wave == k)].article_id) == set(ids)


def test_cap_binds_in_fixture(cfg, frame):
    """Guard against a vacuous cap test: the dominant publisher must hit the cap exactly."""
    s = pd.concat(draw_all(cfg, frame))
    ac = cfg["allocation"]
    assert (s.source == "bigpub").sum() == int(ac["publisher_cap_share"] * ac["n_total"])


# ---------------------------------------------------------------- 100-article redesign, absent annotator
def test_random_reserve_prefix_consistent(cfg, frame):
    """Shrinking n_random keeps the earlier picks: with random_draw_size fixed,
    the smaller reserve is a prefix of the larger one, wave slices included."""
    big, small = copy.deepcopy(cfg), copy.deepcopy(cfg)
    for c, n, w in ((big, 40, 4), (small, 20, 2)):
        c["allocation"].update(random_draw_size=40, n_random=n, n_waves=w)
    rb, wb = waves.random_reserve(big, frame)
    rs, ws = waves.random_reserve(small, frame)
    assert rs == rb[:20] and ws[1] == wb[1] and ws[2] == wb[2]


def test_returned_path_tolerates_renames(cfg, tmp_path):
    d = cfg["paths"]["returned_dir"].format(k=1)
    os.makedirs(d)
    for n in ("annotation_wave1_raphael1.csv", "annotation_wave1_raphael1.csv:Zone.Identifier"):
        open(os.path.join(d, n), "w").write("item_id\n")
    assert assign.returned_path(cfg, 1, "raphael").endswith("annotation_wave1_raphael1.csv")
    assert assign.returned_path(cfg, 1, "nobody") is None
    open(os.path.join(d, "annotation_wave1_raphael2.csv"), "w").write("item_id\n")
    with pytest.raises(SystemExit):
        assign.returned_path(cfg, 1, "raphael")


def test_catchup_redeals_orphans(cfg, frame):
    s = draw_all(cfg, frame)[0]
    texts = pd.DataFrame({"article_id": s.article_id, "title": "t", "lead_150w": "lead"})
    c5 = copy.deepcopy(cfg)
    c5["annotators"] = cfg["annotators"] + ["gone"]
    files, key = assign.assign(c5, LOG, 1, s, texts)
    os.makedirs(assign.ann_dir(cfg)), os.makedirs(assign.admin_dir(cfg))
    key.to_csv(assign.key_path(cfg, 1), index=False)
    for a, df in files.items():
        df.to_csv(assign.ann_path(cfg, 1, a), index=False)
    orphans = set(key.item_id[key.raters == "gone"])
    assert orphans
    cf, ck = assign.catchup(c5, LOG, 1, ["gone"])
    assert set(ck.item_id) == orphans and set(ck.new_rater) <= set(cfg["annotators"])
    assert set().union(*[set(df.item_id) for df in cf.values()]) == orphans
    for df in cf.values():
        assert list(df.columns) == assign.BLIND_COLS
