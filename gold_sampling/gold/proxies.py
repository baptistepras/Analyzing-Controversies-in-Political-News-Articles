"""Step 2: distant-supervision proxies per candidate article -> proxies.parquet.

(a) Reddit: candidate raw_url and submission url are normalised with
    gold/urlnorm.py and joined exactly. A post counts if it was created between
    2 days before and `max_days_after_pub` days after publication (dates in
    the dataset are day-level, posts are UTC). upvote_ratio is not served by
    the bulk search, so it is fetched by id for matched posts only.
    upvote_ratio before `upvote_ratio_valid_from` (2020) is a constant 1 in the
    archive and is set to NaN, so ratio-based features exist for 2020-2022 only.
(b) Wikipedia: topic-level and title-level hits on controversial pages
    (gold/wiki.py).
reddit_status explains every unmatched article:
    missing_2018  - dataset has no URL for 2018
    no_url        - URL missing / unparsable for another reason
    not_posted    - URL fine, no submission in the 4 subreddits
    out_of_window - only posts outside the time window
    matched
"""

import glob
import os
import time

import numpy as np
import pandas as pd
import requests

from .common import load_config, p, start_run
from .urlnorm import normalize_url
from . import wiki


def load_posts(cfg, log, hosts):
    files = sorted(glob.glob(os.path.join(p(cfg, "data_dir"), "reddit", "raw", "*", "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cov = pd.Series([f.split(os.sep)[-2] for f in files]).value_counts().to_dict()
    log.info(f"{len(df)} Reddit posts from {len(files)} subreddit-months {cov}")
    df = df.drop_duplicates("id")
    df["url_key"] = df.url.map(normalize_url)
    df["host"] = df.url_key.str.split("/", n=1).str[0]
    df = df[df.host.isin(hosts)].copy()
    log.info(f"{len(df)} posts link to a candidate publisher host")
    return df, cov


def fetch_upvote_ratio(cfg, log, ids):
    path = os.path.join(p(cfg, "data_dir"), "reddit", "matched_full.parquet")
    old = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(
        columns=["id", "upvote_ratio", "score_full", "num_comments_full", "retrieved_on_full"])
    todo = sorted(set(ids) - set(old.id))
    log.info(f"upvote_ratio: {len(todo)} matched posts to fetch ({len(old)} cached)")
    rows, s = [], requests.Session()
    s.headers["User-Agent"] = cfg["wiki"]["user_agent"]
    for i in range(0, len(todo), 500):
        for attempt in range(8):
            try:
                js = s.get(cfg["reddit"]["api"] + "/posts/ids",
                           params={"ids": ",".join(todo[i:i + 500])}, timeout=180).json()
                if js.get("error"):
                    raise RuntimeError(js["error"])
                break
            except Exception as e:
                log.warning(f"ids batch {i}: {e!r}")
                time.sleep(min(300, 5 * 2 ** attempt))
        else:
            raise RuntimeError("upvote_ratio fetch failed")
        rows += [{"id": x["id"], "upvote_ratio": x.get("upvote_ratio"), "score_full": x.get("score"),
                  "num_comments_full": x.get("num_comments"), "retrieved_on_full": x.get("retrieved_on")}
                 for x in js["data"] or []]
        time.sleep(cfg["reddit"]["sleep_s"])
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if rows else old
    df.to_parquet(path, index=False)
    return df


def reddit_features(cfg, log, cand):
    rc = cfg["reddit"]
    cand = cand[["article_id", "year", "publication_date", "raw_url", "url_status"]].copy()
    cand["url_key"] = cand.raw_url.map(normalize_url)
    hosts = set(cand.url_key.dropna().str.split("/", n=1).str[0])
    posts, cov = load_posts(cfg, log, hosts)

    m = cand.dropna(subset=["url_key"]).merge(
        posts[["id", "url_key", "subreddit", "score", "num_comments", "created_utc", "retrieved_on"]],
        on="url_key")
    # unit-safe (DuckDB DATE arrives as datetime64[us], not [ns])
    pub = (pd.to_datetime(m.publication_date) - pd.Timestamp("1970-01-01")).dt.total_seconds()
    lag_days = (m.created_utc - pub) / 86400
    m["in_window"] = lag_days.between(-2, rc["max_days_after_pub"] + 1)
    m["ingest_lag_h"] = (m.retrieved_on - m.created_utc) / 3600
    log.info(f"{len(m)} url matches ({m.article_id.nunique()} articles); in window: {m.in_window.mean():.1%}; "
             f"ingest lag hours (score snapshot) median {m.ingest_lag_h.median():.1f}")
    out_of_window = set(m.article_id[~m.in_window]) - set(m.article_id[m.in_window])
    m = m[m.in_window]

    full = fetch_upvote_ratio(cfg, log, m.id.tolist())
    m = m.merge(full[["id", "upvote_ratio"]], on="id", how="left")
    m["upvote_ratio"] = pd.to_numeric(m.upvote_ratio, errors="coerce")
    # the archive stores a constant 1 placeholder before 2020: treat as missing
    valid_from = pd.Timestamp(rc["upvote_ratio_valid_from"]).timestamp()
    n_ph = int((m.created_utc < valid_from).sum())
    m.loc[m.created_utc < valid_from, "upvote_ratio"] = np.nan
    log.info(f"upvote_ratio set to NaN for {n_ph} matched posts before {rc['upvote_ratio_valid_from']} "
             f"(placeholder value 1); remaining values: {m.upvote_ratio.describe().round(3).to_dict()}")

    g = m.groupby("article_id")
    f = pd.DataFrame({
        "reddit_n_posts": g.size(),
        "reddit_subreddits": g.subreddit.apply(lambda s: "|".join(sorted(set(s)))),
        "reddit_n_subreddits": g.subreddit.nunique(),
        "reddit_score_max": g.score.max(),
        "reddit_score_sum": g.score.sum(),
        "reddit_comments_max": g.num_comments.max(),
        "reddit_comments_sum": g.num_comments.sum(),
    })
    # upvote_ratio of the most-commented post, and per subreddit (for disagreement)
    top = m.sort_values(["num_comments", "score"], ascending=False).drop_duplicates("article_id")
    f["reddit_upvote_ratio"] = top.set_index("article_id").upvote_ratio
    per_sub = (m.sort_values("num_comments", ascending=False)
                .drop_duplicates(["article_id", "subreddit"])
                .pivot(index="article_id", columns="subreddit", values="upvote_ratio"))
    for s in rc["subreddits"]:
        f[f"reddit_ur_{s}"] = per_sub[s] if s in per_sub else np.nan
    ur = f[[f"reddit_ur_{s}" for s in rc["subreddits"]]]
    f["reddit_disagreement"] = np.where(ur.notna().sum(axis=1) >= 2, ur.max(axis=1) - ur.min(axis=1), np.nan)
    f["reddit_comment_ratio"] = f.reddit_comments_sum / (f.reddit_score_sum.clip(lower=0) + 1)
    f = f.reset_index()

    cand = cand.merge(f, on="article_id", how="left")
    cand["reddit_matched"] = cand.reddit_n_posts.notna()
    cand["reddit_status"] = np.select(
        [cand.url_status == "missing_2018", cand.url_key.isna(), cand.reddit_matched,
         cand.article_id.isin(out_of_window)],
        ["missing_2018", "no_url", "matched", "out_of_window"], "not_posted")
    for c in ["reddit_n_posts", "reddit_n_subreddits"]:
        cand[c] = cand[c].fillna(0).astype(int)
    return cand.drop(columns=["year", "publication_date", "raw_url", "url_status", "url_key"]), cov


def main():
    cfg = load_config()
    log = start_run(cfg, "proxies")
    cand = pd.read_parquet(p(cfg, "candidates"))
    rf, cov = reddit_features(cfg, log, cand)

    wf = wiki.article_features(cfg, log, cand)
    prox = cand[["article_id"]].merge(rf, on="article_id", how="left").merge(wf, on="article_id", how="left")
    prox.to_parquet(p(cfg, "proxies"), index=False)
    log.info(f"wrote {p(cfg, 'proxies')}: {len(prox)} rows")

    # ---- match rate per year / bias (checkpoint before going further) -----
    x = cand[["article_id", "year", "bias3", "reliability_pool"]].merge(prox, on="article_id")
    for by in ["year", "bias3", "reliability_pool"]:
        t = pd.crosstab(x[by], x.reddit_status, margins=True)
        t["match_rate"] = (t.get("matched", 0) / t["All"]).round(4)
        log.info(f"\nReddit status by {by}:\n{t.to_string()}")
    num = ["reddit_n_posts", "reddit_score_max", "reddit_comments_max", "reddit_upvote_ratio",
           "reddit_comment_ratio", "reddit_disagreement", "wiki_topic_n_issue", "wiki_title_n_issue",
           "wiki_topic_n_tag", "wiki_title_n_tag"]
    log.info("\nSpearman correlations (matched articles only for reddit_*):\n"
             + x[x.reddit_matched][num].corr("spearman").round(2).to_string())


if __name__ == "__main__":
    main()
