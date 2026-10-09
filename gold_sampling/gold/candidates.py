"""Step 1: candidate universe -> candidates.parquet (article_id + metadata, no text).

Filters, applied in this order (counts logged and written to
data/candidates_filter_summary.csv):
  0. all articles in metadata.db
  1. publisher MBFC label not excluded (Satire)
  2. political topic (topic-level lexicon rule, see gold/political.py)
  3. title length >= min_title_chars
  4. min_words <= words (mask tokens not counted) <= max_words
  5. mask_share <= max_mask_share
  6. exact content duplicates (content_hash), keep earliest
  7. near-duplicate titles, keep earliest
"""

import glob
import os
import re
import unicodedata

import duckdb
import numpy as np
import pandas as pd

from .common import load_config, p, start_run
from .political import PATTERN, rep_hits

STOP = set("""a an the and or of to in on for at by with from as is are was were be been
it its this that these those after over into about than but not no new says said say
will would can could may might has have had he she they we you i his her their our
""".split())


def political_topics(cfg, log):
    fc = cfg["filters"]["political"]
    con = duckdb.connect(p(cfg, "metadata_db"), read_only=True)
    pat = PATTERN.replace("'", "''")
    t = con.execute(f"""
        SELECT a.topic_id, any_value(t.year) AS year, any_value(t.representation) AS rep,
               count(*) AS n_articles,
               avg(regexp_matches(a.title, '{pat}')::INT) AS title_share
        FROM articles a JOIN topics t USING (topic_id) GROUP BY 1""").fetchdf()
    t["rep_hits"] = t.rep.map(rep_hits)
    by_title = t.title_share >= fc["min_title_share"]
    by_kw = (t.rep_hits >= fc["alt_min_rep_hits"]) & (t.title_share >= fc["alt_min_title_share"])
    t["political"] = by_title | by_kw
    t["reason"] = np.select([by_title, by_kw],
                            [f"title_share>={fc['min_title_share']}",
                             f"rep_hits>={fc['alt_min_rep_hits']} & title_share>={fc['alt_min_title_share']}"],
                            "dropped")
    t = t.sort_values(["political", "n_articles"], ascending=False)
    t.to_csv(os.path.join(p(cfg, "data_dir"), "political_topics.csv"), index=False)
    k = t[t.political]
    log.info(f"political topics: {len(k)}/{len(t)} topics, {k.n_articles.sum()}/{t.n_articles.sum()} articles")
    return t


def _title_key(title):
    s = unicodedata.normalize("NFKC", title or "").lower()
    s = re.sub(r"[‘’“”]", "'", s)
    # drop a short trailing " - Publisher" / " | Publisher" suffix
    s = re.sub(r"\s+[-|–—]\s+[^-|–—]{1,40}$", "", s)
    toks = re.findall(r"[a-z0-9]+", s)
    content = sorted(set(t for t in toks if t not in STOP))
    # token-set key catches reordering / punctuation / stopword edits; fall back to
    # the exact token sequence when too few content words to be safe
    return " ".join(content) if len(content) >= 4 else "=" + " ".join(toks)


def load_articles(cfg):
    con = duckdb.connect(p(cfg, "metadata_db"), read_only=True)
    df = con.execute("""
        SELECT a.article_id, a.year, a.source, a.title, a.content_hash, a.publication_date,
               a.topic_id, a.url AS meta_url, a.covid_article,
               t.representation AS topic_rep,
               s.label AS mbfc_label, s.bias AS mbfc_bias, s.factuality AS mbfc_factuality,
               s.credibility AS mbfc_credibility, s.country AS publisher_country
        FROM articles a JOIN topics t USING (topic_id) LEFT JOIN sources s USING (source)""").fetchdf()
    for c in ["mbfc_label", "mbfc_bias", "mbfc_factuality", "mbfc_credibility"]:
        df[c] = df[c].astype("string")
    return df


def load_content_stats(cfg):
    files = sorted(glob.glob(os.path.join(p(cfg, "data_dir"), "content_stats", "*.parquet")))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), len(files)


LEN_BINS = [0, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, np.inf]
MASK_BINS = [0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]


def _distributions(cfg, log, d):
    """Length / mask-share distribution of the political universe BEFORE the
    length and mask filters, so the bounds can be judged (written for the report)."""
    out = p(cfg, "data_dir")
    q = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    qs = pd.DataFrame({"n_words_text": d.n_words_text.quantile(q), "n_words": d.n_words.quantile(q),
                       "mask_share": d.mask_share.quantile(q)})
    qs.index.name = "quantile"
    qs.to_csv(os.path.join(out, "length_quantiles.csv"))
    h = pd.cut(d.n_words_text, LEN_BINS, right=False).value_counts(sort=False)
    by_year = pd.crosstab(pd.cut(d.n_words_text, LEN_BINS, right=False), d.year)
    by_year.insert(0, "all", h)
    by_year.to_csv(os.path.join(out, "length_histogram.csv"))
    pd.cut(d.mask_share, MASK_BINS, include_lowest=True).value_counts(sort=False).to_csv(
        os.path.join(out, "mask_histogram.csv"))
    log.info(f"length quantiles (words excl. mask tokens), political universe:\n{qs.round(3).to_string()}")


def main():
    cfg = load_config()
    log = start_run(cfg, "candidates")
    fc = cfg["filters"]

    topics = political_topics(cfg, log)
    df = load_articles(cfg)
    cs, nfiles = load_content_stats(cfg)
    if nfiles != 36:
        raise SystemExit(f"content stats incomplete: {nfiles}/36 files; run `python -m gold.content_stats`")
    cs["n_words_text"] = cs.n_words - cs.n_mask
    df = df.merge(cs.drop(columns=[c for c in cs.columns if c.startswith("n_") and c not in
                                   ("n_words", "n_mask", "n_words_text")]),
                  on="article_id", how="left")
    miss = df.n_words.isna().sum()
    log.info(f"{len(df)} metadata articles; {miss} without content stats (not in HF parquet)")

    # ---- URL / domain: 2018 has none, handled explicitly ------------------
    for c in ["domain", "raw_url"]:
        df.loc[df[c].isin(["N/A", ""]), c] = None
    df["url_status"] = np.where(df.year == 2018, "missing_2018",
                                np.where(df.raw_url.isna(), "missing_other", "ok"))
    n_diff = ((df.raw_url != df.meta_url) & df.raw_url.notna()).sum()
    log.info(f"url_status: {df.url_status.value_counts().to_dict()}; raw_url != metadata url: {n_diff}")

    # ---- MBFC derived columns ---------------------------------------------
    b3 = {v: k for k, vs in cfg["bias3"].items() for v in vs}
    df["bias3"] = df.mbfc_bias.map(b3).fillna("Unknown")
    df["reliability_pool"] = np.where(df.mbfc_label.isin(cfg["reliability_pool"]["unreliable"]),
                                      "unreliable", "reliable")
    df["month"] = pd.to_datetime(df.publication_date).dt.strftime("%Y-%m")

    # ---- filters ----------------------------------------------------------
    steps = [("0_all_articles", None)]
    keep = pd.Series(True, index=df.index)
    pol = set(topics.topic_id[topics.political])

    def apply(name, mask):
        nonlocal keep
        before = int(keep.sum())
        keep &= mask
        steps.append((name, before - int(keep.sum())))
        log.info(f"{name}: removed {before - int(keep.sum())}, remaining {int(keep.sum())}")

    apply("1_excluded_mbfc_label", ~df.mbfc_label.isin(fc["exclude_mbfc_labels"]))
    apply("2_non_political_topic", df.topic_id.isin(pol))
    apply("3_short_title", df.title.fillna("").str.len() >= fc["min_title_chars"])
    apply("3b_no_content_stats", df.n_words.notna())
    _distributions(cfg, log, df[keep])
    apply("4_length_out_of_bounds", df.n_words_text.between(fc["min_words"], fc["max_words"]))
    apply("5_mostly_masked", df.mask_share <= fc["max_mask_share"])

    elig = df[keep].sort_values(["publication_date", "article_id"])
    dup_hash = elig.index[elig.duplicated("content_hash")]
    apply("6_exact_content_duplicate", ~df.index.isin(dup_hash))

    elig = df[keep].copy()
    elig["title_key"] = elig.title.map(_title_key)
    elig = elig.sort_values(["publication_date", "article_id"])
    elig["n_title_dups"] = elig.groupby("title_key").article_id.transform("size")
    firsts = elig[~elig.duplicated("title_key")]
    apply("7_near_duplicate_title", df.index.isin(firsts.index))

    cand = df[keep].merge(firsts[["article_id", "n_title_dups"]], on="article_id", how="left")
    cand = cand.merge(topics[["topic_id", "title_share", "rep_hits"]].rename(
        columns={"title_share": "topic_political_share", "rep_hits": "topic_political_kw"}),
        on="topic_id", how="left")
    cols = ["article_id", "year", "publication_date", "month", "source", "domain", "raw_url",
            "url_status", "title", "n_words", "n_words_text", "n_mask", "mask_share",
            "mbfc_label", "mbfc_bias", "bias3", "mbfc_factuality", "mbfc_credibility",
            "reliability_pool", "publisher_country", "topic_id", "topic_rep",
            "topic_political_share", "topic_political_kw", "covid_article", "n_title_dups"]
    cand = cand[cols].sort_values("article_id").reset_index(drop=True)
    cand.to_parquet(p(cfg, "candidates"), index=False)

    summ = pd.DataFrame(steps, columns=["filter", "removed"])
    summ.loc[0, "removed"] = 0
    summ["remaining"] = len(df) - summ.removed.cumsum()
    summ.to_csv(os.path.join(p(cfg, "data_dir"), "candidates_filter_summary.csv"), index=False)
    log.info("\n" + summ.to_string(index=False))
    log.info(f"wrote {p(cfg, 'candidates')}: {len(cand)} articles")


if __name__ == "__main__":
    main()
