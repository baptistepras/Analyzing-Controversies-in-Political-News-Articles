#!/usr/bin/env python3
"""Export the labelled gold sample with its Reddit distant-supervision links.

    python export_gold_annotations.py

Reads every returned file in paths.returned_dir (annotationsW{k}/; renames such
as `raphael1` or `kseniia_wave2.csv` are tolerated), joins the admin keys, the
sample design, candidates/proxies/strata metadata, rebuilds the article <-> post
matches of gold/proxies.py for the labelled articles, and fetches the full
submission records + comments of those posts (cached in data/reddit/gold/).

Writes gold_sample/export/:
    annotations_long.csv  one row per (item, annotator)
    articles.csv          one row per article, majority label + agreement
    reddit_posts.csv      one row per matched post (permalink, title, ...)
    reddit_comments.csv   top-N top-level comments per post (reddit.gold_top_comments)
Article text is never written (CC BY-NC-SA); the export dir is git-ignored.
"""

import glob
import os
import re

import numpy as np
import pandas as pd

from gold import assign, waves
from gold.common import load_config, p, start_run
from gold.proxies import load_posts
from gold.reddit_fetch import fetch_gold_comments, fetch_gold_posts
from gold.urlnorm import normalize_url
from rebalance import norm_label

ALIASES = {"kseniia": "ksenia"}
REDDIT = "https://www.reddit.com"
CONF = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
BOTS = {"AutoModerator", "autotldr", "PoliticsModeratorBot", "PublishDateBot"}


def rater_of(path, names):
    """Annotator name from a returned file name: annotation_wave1_raphael1.csv,
    kseniia_wave2.csv -> raphael, ksenia."""
    stem = re.sub(r"^annotation_wave\d+_|_?wave\d+|\d+$", "", os.path.basename(path)[:-4].lower())
    stem = ALIASES.get(stem, stem)
    return stem if stem in names else None


def read_annotations(cfg, log, key):
    rows = []
    names = set(cfg["annotators"]) | set(ALIASES.values())
    for k in sorted(key.wave.unique()):
        for f in sorted(glob.glob(os.path.join(assign.returned_dir(cfg, k), "*.csv"))):
            a = rater_of(f, names)
            if a is None:
                log.warning(f"{f}: cannot tell the annotator, skipped")
                continue
            d = pd.read_csv(f, dtype=str)[["item_id", "label", "confidence", "reason"]]
            rows.append(d.assign(wave=k, annotator=a, label_raw=d.label, label=d.label.map(norm_label),
                                 file=os.path.relpath(f, p(cfg, "sample_dir"))))
    ann = pd.concat(rows, ignore_index=True)
    ann["confidence"] = ann.confidence.str.strip().str.upper()
    ann["confidence_score"] = ann.confidence.map(CONF)          # LOW 1, MEDIUM 2, HIGH 3
    ann["reason"] = ann.reason.str.strip()
    for r in ann[ann.confidence_score.isna()].itertuples():
        log.warning(f"{r.file}: item {r.item_id} has unreadable confidence {r.confidence!r}")
    bad = ann[ann.label.isna()]
    for r in bad.itertuples():
        log.warning(f"{r.file}: item {r.item_id} has no usable label ({r.label_raw!r})")

    # every assigned (item, annotator) should come back exactly once
    exp = key.assign(annotator=key.raters.str.split("|")).explode("annotator")[["item_id", "annotator"]]
    got = ann[["item_id", "annotator"]]
    miss = exp.merge(got, how="left", indicator=True).query("_merge == 'left_only'")
    extra = got.merge(exp, how="left", indicator=True).query("_merge == 'left_only'")
    dup = got[got.duplicated(keep=False)]
    for name, d in [("assigned but not returned", miss), ("returned but not assigned", extra),
                    ("returned twice", dup)]:
        if len(d):
            log.warning(f"{len(d)} (item, annotator) {name}: {d.values.tolist()}")
    log.info(f"{len(ann)} labels from {ann.file.nunique()} files; expected {len(exp)}")
    return ann.drop(columns="file")


def article_frame(cfg, key):
    samp = pd.concat([pd.read_csv(waves.sample_path(cfg, k)) for k in sorted(key.wave.unique())])
    ids = set(key.article_id)
    filt = [("article_id", "in", list(ids))]
    cand = pd.read_parquet(p(cfg, "candidates"), filters=filt)
    prox = pd.read_parquet(p(cfg, "proxies"), filters=filt)
    strat = pd.read_parquet(os.path.join(p(cfg, "data_dir"), "strata_frame.parquet"), filters=filt)
    strat = strat.drop(columns=["stratum", "flag", "matched"])        # already in the sample file
    art = (key[["item_id", "article_id", "wave", "overlap", "n_raters", "raters"]]
           .merge(samp.drop(columns=["wave", "year", "bias3", "reliability_pool", "source", "topic_id",
                                     "month"]), on="article_id", how="left")
           .merge(cand, on="article_id", how="left")
           .merge(prox, on="article_id", how="left")
           .merge(strat, on="article_id", how="left"))
    assert len(art) == len(key) and art.source.notna().all(), "article metadata did not join"
    return art


def match_posts(cfg, log, art):
    """Same matching rule as gold/proxies.reddit_features, for these articles only."""
    rc = cfg["reddit"]
    a = art[["article_id", "publication_date", "raw_url"]].copy()
    a["url_key"] = a.raw_url.map(normalize_url)
    posts, _ = load_posts(cfg, log, set(a.url_key.dropna().str.split("/", n=1).str[0]))
    m = a.merge(posts[["id", "url_key", "url", "subreddit", "score", "num_comments", "created_utc",
                       "retrieved_on", "crosspost_parent"]], on="url_key")
    pub = (pd.to_datetime(m.publication_date) - pd.Timestamp("1970-01-01")).dt.total_seconds()
    m["lag_days"] = (m.created_utc - pub) / 86400
    m = m[m.lag_days.between(-2, rc["max_days_after_pub"] + 1)].drop(columns=["publication_date", "url_key"])
    full = pd.read_parquet(os.path.join(p(cfg, "data_dir"), "reddit", "matched_full.parquet"))
    m = m.merge(full, on="id", how="left")
    valid_from = pd.Timestamp(rc["upvote_ratio_valid_from"]).timestamp()
    m["upvote_ratio"] = pd.to_numeric(m.upvote_ratio, errors="coerce").where(m.created_utc >= valid_from)
    n = m.groupby("article_id").size().reindex(art.article_id, fill_value=0)
    diff = n[n.values != art.set_index("article_id").reddit_n_posts.values]
    if len(diff):
        log.warning(f"post count differs from proxies.parquet for {len(diff)} article(s): {diff.to_dict()}")
    return m.rename(columns={"id": "post_id", "url": "post_url", "score": "score_ingest",
                             "num_comments": "num_comments_ingest", "retrieved_on": "retrieved_on_ingest"})


def main():
    cfg = load_config()
    log = start_run(cfg, "export_gold")
    out = os.path.join(p(cfg, "sample_dir"), "export")
    os.makedirs(out, exist_ok=True)

    key = pd.concat([pd.read_csv(assign.key_path(cfg, k)).assign(wave=k)
                     for k in range(1, cfg["allocation"]["n_waves"] + 1)
                     if os.path.exists(assign.key_path(cfg, k))], ignore_index=True)
    ann = read_annotations(cfg, log, key)
    art = article_frame(cfg, key)

    # ---- Reddit: matched posts + full records + comments -------------------
    m = match_posts(cfg, log, art)
    ids = sorted(m.post_id.unique())
    full = fetch_gold_posts(cfg, log, ids)
    full = full[["post_id", "title", "author", "permalink", "link_flair_text", "over_18",
                 "removed_by_category", "num_crossposts", "upvote_ratio", "score", "num_comments"]]
    # upvote_ratio comes from matched_full (pre-2020 placeholder already NaN);
    # score/num_comments here are the latest snapshot, *_ingest the first one
    full = full.drop(columns="upvote_ratio").rename(columns={"title": "post_title"})
    m = m.drop(columns=["score_full", "num_comments_full", "retrieved_on_full"])   # same snapshot as /posts/ids
    posts = m.merge(full, on="post_id", how="left")
    posts["permalink"] = np.where(posts.permalink.notna(), REDDIT + posts.permalink.fillna(""),
                                  REDDIT + "/r/" + posts.subreddit + "/comments/" + posts.post_id + "/")
    posts["created"] = pd.to_datetime(posts.created_utc, unit="s", utc=True)
    if (miss := posts.post_title.isna().sum()):
        log.warning(f"{miss} post(s) not returned by /posts/ids (permalink built from id)")
    posts = posts.sort_values(["article_id", "num_comments_ingest"], ascending=[True, False])
    lead = ["article_id", "post_id", "permalink", "subreddit", "post_title", "author", "created", "lag_days"]
    posts = posts[lead + [c for c in posts.columns if c not in lead]]

    com = fetch_gold_comments(cfg, log, ids).dropna(subset=["id"])
    com = com[com.parent_id.str.startswith("t3_")
              & ~com.body.isin(["[deleted]", "[removed]"]) & ~com.author.isin(BOTS)]
    top = (com.sort_values(["post_id", "score"], ascending=[True, False])
              .groupby("post_id").head(cfg["reddit"]["gold_top_comments"]))
    top = top.merge(posts[["post_id", "article_id", "subreddit"]], on="post_id")
    top["comment_link"] = REDDIT + "/r/" + top.subreddit + "/comments/" + top.post_id + "/_/" + top.id + "/"
    top = top.rename(columns={"id": "comment_id"})[
        ["article_id", "post_id", "comment_id", "comment_link", "author", "score", "created_utc", "body"]]
    log.info(f"comments: {len(com)} top-level archived, {len(top)} exported "
             f"(top {cfg['reddit']['gold_top_comments']} per post; bots and [deleted]/[removed] dropped)")

    links = posts.groupby("article_id").permalink.agg("|".join).rename("reddit_links")
    art = art.merge(links, on="article_id", how="left")

    # ---- annotations: long + per-article majority ---------------------------
    long = ann.merge(art, on=["item_id", "wave"], how="left")
    assert long.article_id.notna().all(), "item_id without article"

    def summary(g):
        s = g.label.dropna()
        c = s.value_counts()
        maj = None if c.empty or (len(c) > 1 and c.iloc[0] == c.iloc[1]) else c.index[0]
        return pd.Series({"n_labels": len(s), "n_yes": int((s == "YES").sum()), "majority_label": maj,
                          "agreement": c.iloc[0] / len(s) if len(s) else np.nan,
                          "mean_confidence": g.confidence_score.mean(),
                          "annotators": "|".join(sorted(g.annotator))})

    votes = long.groupby("article_id")[["label", "confidence_score", "annotator"]].apply(summary).reset_index()
    arts = art.merge(votes, on="article_id", how="left")
    first = ["item_id", "article_id", "wave", "majority_label", "agreement", "n_labels", "n_yes",
             "title", "source", "publication_date", "raw_url", "reddit_links"]
    arts = arts[first + [c for c in arts.columns if c not in first]]
    first = ["item_id", "article_id", "wave", "annotator", "label", "confidence", "confidence_score", "reason", "label_raw",
             "title", "source", "raw_url", "reddit_links"]
    long = long[first + [c for c in long.columns if c not in first]]

    for name, d in [("annotations_long", long), ("articles", arts), ("reddit_posts", posts),
                    ("reddit_comments", top)]:
        d.to_csv(os.path.join(out, f"{name}.csv"), index=False)
        log.info(f"{name}.csv: {len(d)} rows x {d.shape[1]} columns")
    log.info(f"majority labels: {arts.majority_label.value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
