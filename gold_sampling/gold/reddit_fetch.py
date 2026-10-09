"""Step 2a (download): per-subreddit submissions from the Arctic Shift API.

The Watchful1 per-subreddit torrent is no longer served, so we page through
the Arctic Shift search API instead (same Pushshift lineage), keeping only the
fields we need. A shared queue of (subreddit, month) jobs feeds `workers` threads; results are written one parquet file
per (subreddit, month), so an interrupted run resumes at month granularity.

Caveat for the report: score / num_comments are snapshots taken at ingest
time (`retrieved_on`), not final values.
"""

import calendar
import datetime as dt
import os
import queue
import threading
import time

import pandas as pd
import requests

from .common import load_config, p, start_run


def _months(start, end):
    d = dt.date.fromisoformat(start)
    stop = dt.date.fromisoformat(end)
    while d < stop:
        nxt = d.replace(day=1) + dt.timedelta(days=calendar.monthrange(d.year, d.month)[1])
        yield d, min(nxt, stop)
        d = nxt


def _epoch(d):
    return calendar.timegm(d.timetuple())


class Fetcher:
    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        self.rc = cfg["reddit"]
        self.local = threading.local()          # one requests.Session per worker

    @property
    def sess(self):
        if not hasattr(self.local, "sess"):
            self.local.sess = requests.Session()
            self.local.sess.headers["User-Agent"] = self.cfg["wiki"]["user_agent"]
        return self.local.sess

    def _get(self, params, endpoint="/posts/search"):
        url = self.rc["api"] + endpoint
        for attempt in range(self.rc["max_retries"]):
            try:
                r = self.sess.get(url, params=params, timeout=120)
                if r.status_code == 429:
                    wait = float(r.headers.get("x-ratelimit-reset", 30)) + 1
                    self.log.warning(f"429, sleeping {wait:.0f}s")
                    time.sleep(wait)
                    continue
                js = r.json()
                if js.get("error"):
                    raise RuntimeError(js["error"])
                r.raise_for_status()
                return js["data"] or []
            except Exception as e:  # network error, timeout, "Timeout. Maybe slow down"
                wait = min(300, 5 * 2 ** attempt)
                self.log.warning(f"{endpoint} {params.get('subreddit') or params.get('link_id') or ''} "
                                 f"after={params.get('after')}: {e!r}; retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"giving up on {params}")

    def month(self, sub, start, end, out):
        lo, hi = _epoch(start), _epoch(end)
        after, seen, rows, stalled = lo - 1, set(), [], False
        while True:
            page = self._get({"subreddit": sub, "after": after, "before": hi, "sort": "asc",
                              "limit": "auto", "fields": ",".join(self.rc["fields"])})
            if not page:
                break
            last = max(x["created_utc"] for x in page)
            new = [x for x in page if x["id"] not in seen]
            if not new:
                # whole page already seen: step strictly past `last` once, then stop
                if stalled or after >= last:
                    break
                stalled, after = True, last
                continue
            stalled = False
            for x in new:
                seen.add(x["id"])
            rows += new
            # resume from last-1 so posts sharing the boundary second are not
            # skipped; the re-fetched duplicates are dropped through `seen`
            after = last - 1
            time.sleep(self.rc["sleep_s"])
        df = pd.DataFrame(rows, columns=self.rc["fields"])
        tmp = out + ".tmp"
        df.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return len(df)

    def jobs(self):
        """All missing (subreddit, month) jobs, months interleaved across subreddits."""
        out = []
        for a, b in self.rc["windows"]:
            for start, end in _months(a, b):
                for sub in self.rc["subreddits"]:
                    d = os.path.join(p(self.cfg, "data_dir"), "reddit", "raw", sub)
                    os.makedirs(d, exist_ok=True)
                    out_path = os.path.join(d, f"{start:%Y-%m}.parquet")
                    if not os.path.exists(out_path):
                        out.append((sub, start, end, out_path))
        return out

    def worker(self, q):
        while True:
            try:
                sub, start, end, out = q.get_nowait()
            except queue.Empty:
                return
            t = time.time()
            n = self.month(sub, start, end, out)
            self.log.info(f"{sub} {start:%Y-%m}: {n} posts ({time.time()-t:.0f}s, {q.qsize()} jobs left)")


GOLD_POST_FIELDS = ["id", "title", "author", "permalink", "url", "subreddit", "created_utc", "score",
                    "num_comments", "upvote_ratio", "link_flair_text", "over_18", "removed_by_category",
                    "num_crossposts", "retrieved_on"]
GOLD_COMMENT_FIELDS = ["id", "link_id", "parent_id", "author", "body", "score", "created_utc"]


def _cached(path, ids, fetch):
    """Rows of the parquet cache at `path`, after fetching the ids it lacks
    (`fetch(todo) -> DataFrame` must carry a `post_id` column)."""
    old = pd.read_parquet(path) if os.path.exists(path) else None
    done = set(old.post_id) if old is not None else set()
    todo = sorted(set(ids) - done)
    if todo:
        new = fetch(todo)
        df = new if old is None else pd.concat([old, new], ignore_index=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        old = df
    return old[old.post_id.isin(set(ids))]


def fetch_gold_posts(cfg, log, ids):
    """Full submission records (title, author, permalink, flair...) for the
    gold-sample posts, by id; cached in data/reddit/gold/posts.parquet."""
    f = Fetcher(cfg, log)

    def fetch(todo):
        rows = []
        for i in range(0, len(todo), 500):
            rows += f._get({"ids": ",".join(todo[i:i + 500])}, endpoint="/posts/ids")
            time.sleep(f.rc["sleep_s"])
        df = pd.DataFrame([{k: x.get(k) for k in GOLD_POST_FIELDS} for x in rows], columns=GOLD_POST_FIELDS)
        log.info(f"posts/ids: {len(df)}/{len(todo)} posts returned")
        return df.assign(post_id=df.id)

    return _cached(os.path.join(p(cfg, "data_dir"), "reddit", "gold", "posts.parquet"), ids, fetch)


def fetch_gold_comments(cfg, log, ids):
    """Every archived comment of the gold-sample posts (paged by created_utc,
    like Fetcher.month); cached in data/reddit/gold/comments.parquet. A post
    with no comment gets one all-NaN row so it is not fetched again."""
    f = Fetcher(cfg, log)

    def one(pid):
        after, seen, rows = "2005-06-01", set(), []      # the API rejects after=0 (Reddit start)
        while True:
            page = f._get({"link_id": pid, "after": after, "sort": "asc", "limit": "auto",
                           "fields": ",".join(GOLD_COMMENT_FIELDS)}, endpoint="/comments/search")
            new = [x for x in page if x["id"] not in seen]
            if not new:
                break
            seen.update(x["id"] for x in new)
            rows += new
            after = max(x["created_utc"] for x in page) - 1
            time.sleep(f.rc["sleep_s"])
        return rows

    def fetch(todo):
        out = []
        for i, pid in enumerate(todo):
            rows = one(pid)
            out.append(pd.DataFrame(rows, columns=GOLD_COMMENT_FIELDS).assign(post_id=pid) if rows
                       else pd.DataFrame({"post_id": [pid]}))
            log.info(f"comments {i + 1}/{len(todo)} {pid}: {len(rows)}")
        return pd.concat(out, ignore_index=True)

    return _cached(os.path.join(p(cfg, "data_dir"), "reddit", "gold", "comments.parquet"), ids, fetch)


def main():
    cfg = load_config()
    log = start_run(cfg, "reddit_fetch")
    f = Fetcher(cfg, log)
    q = queue.Queue()
    for j in f.jobs():
        q.put(j)
    log.info(f"{q.qsize()} subreddit-month jobs, {cfg['reddit']['workers']} workers")
    threads = [threading.Thread(target=f.worker, args=(q,)) for _ in range(cfg["reddit"]["workers"])]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.info("done")


if __name__ == "__main__":
    main()
