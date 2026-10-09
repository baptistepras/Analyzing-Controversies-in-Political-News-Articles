"""Step 2b: Wikipedia-based topic controversy (cheap, list-based).

Sources (MediaWiki API, cached under data/wiki/):
  LCI : pages linked from "Wikipedia:List of controversial issues"
  TAG : subject pages whose talk page carries {{Controversial}}
        (Category:Wikipedia controversial topics)
Each page is canonicalised through redirects, and every redirect pointing to
it is kept as an alias, so "Rittenhouse shooting" or "Roe v Wade" still hit.
Matching is local and exact on lower-cased word n-grams:
  topic level   : the topic's 5 BERTopic keywords (underscores -> spaces)
  article level : the article title
Only LCI pages from issue sections (not People / Entertainment / Sports /
Languages) count as "issue" hits; TAG hits are recorded but are mostly people
and places. Issue pages that hit too large a share of the corpus (Trump-like
context entities) are ignored, see article_features().
Anonymous API clients are limited to 10 requests/min, so the crawl is paced
(~200 requests total) and everything is cached after the first run.
"""

import json
import os
import re
import time

import pandas as pd
import requests

from .common import load_config, p, start_run

MIN_ALIAS_CHARS = 4


class Wiki:
    def __init__(self, cfg, log):
        self.cfg, self.log = cfg, log
        wc = cfg["wiki"]
        self.api, self.gap = wc["api"], wc["min_interval_s"]
        self.s = requests.Session()
        self.s.headers["User-Agent"] = wc["user_agent"]
        self.dir = os.path.join(p(cfg, "data_dir"), "wiki")
        os.makedirs(self.dir, exist_ok=True)
        self.last = 0.0

    def _get(self, params):
        for attempt in range(8):
            time.sleep(max(0.0, self.gap - (time.time() - self.last)))
            self.last = time.time()
            try:
                r = self.s.get(self.api, params=params, timeout=60)
                if r.status_code in (429, 503):
                    wait = float(r.headers.get("retry-after") or 60)
                    self.log.warning(f"wiki api {r.status_code}; sleeping {wait:.0f}s")
                    time.sleep(wait)
                    continue
                js = r.json()
                if "error" in js:
                    raise RuntimeError(js["error"])
                return js
            except Exception as e:
                wait = 10 * 2 ** attempt
                self.log.warning(f"wiki api: {e!r}; retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError(f"wiki api gave up on {params}")

    def query(self, params):
        params = {**params, "action": "query", "format": "json", "formatversion": 2}
        while True:
            js = self._get(params)
            yield js
            if "continue" not in js:
                return
            params.update(js["continue"])

    def _cached(self, name, fn):
        path = os.path.join(self.dir, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        out = fn()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        return out

    def lists(self):
        def fetch():
            lci, tag = [], []
            for js in self.query({"prop": "links", "titles": "Wikipedia:List of controversial issues",
                                  "plnamespace": 0, "pllimit": "max"}):
                for pg in js["query"]["pages"]:
                    lci += [l["title"] for l in pg.get("links", [])]
            for js in self.query({"list": "categorymembers", "cmlimit": "max",
                                  "cmtitle": "Category:Wikipedia controversial topics"}):
                tag += [m["title"][5:].split("/")[0] for m in js["query"]["categorymembers"]
                        if m["title"].startswith("Talk:")]
            self.log.info(f"LCI links {len(lci)}, TAG talk pages {len(tag)}")
            return {"LCI": sorted(set(lci)), "TAG": sorted(set(tag))}
        return self._cached("lists.json", fetch)

    def lci_sections(self):
        """LCI page title -> its level-3 section (Politics and economics, People, ...)."""
        def fetch():
            js = self._get({"action": "parse", "page": "Wikipedia:List of controversial issues",
                            "prop": "wikitext", "format": "json", "formatversion": 2})
            sec, out = None, {}
            for line in js["parse"]["wikitext"].splitlines():
                h = re.match(r"^(=+)\s*(.*?)\s*\1\s*$", line)
                if h:
                    sec = h.group(2) if len(h.group(1)) == 3 else None
                    continue
                for link in re.findall(r"\[\[([^\]|#]+)", line):
                    if sec and ":" not in link:
                        out.setdefault(link.strip(), sec)
            return out
        return self._cached("lci_sections.json", fetch)

    def aliases(self, titles):
        """title -> canonical page + all redirect titles pointing to it."""
        def fetch():
            out = {}
            for i in range(0, len(titles), 50):
                batch = titles[i:i + 50]
                for js in self.query({"titles": "|".join(batch), "redirects": 1, "prop": "redirects",
                                      "rdnamespace": 0, "rdlimit": "max"}):
                    q = js["query"]
                    norm = {n["from"]: n["to"] for n in q.get("normalized", [])}
                    redir = {r["from"]: r["to"] for r in q.get("redirects", [])}
                    pages = {pg["title"]: pg for pg in q.get("pages", [])}
                    for t in batch:
                        c = redir.get(norm.get(t, t), norm.get(t, t))
                        pg = pages.get(c)
                        if pg is None or pg.get("missing"):
                            continue
                        e = out.setdefault(t, {"page": c, "redirects": []})
                        e["redirects"] += [r["title"] for r in pg.get("redirects", [])]
                if i % 1000 == 0:
                    self.log.info(f"aliases: {i + len(batch)}/{len(titles)} titles")
            return out
        return self._cached("aliases.json", fetch)


def _norm(s):
    s = re.sub(r"\s*\([^)]*\)$", "", s)          # drop "(disambiguator)"
    return " ".join(re.findall(r"[a-z0-9]+(?:['\-][a-z0-9]+)*", s.lower()))


def alias_table(cfg, log):
    """alias -> page, src in {LCI_issue, LCI_other, TAG}.
    LCI_issue = LCI pages outside the excluded sections (People, Entertainment, ...).
    Aliases: the page title, the listed title, and multi-word redirects only;
    single-word redirects ("Union" -> Trade union, "Christ" -> ...) were the
    main source of spurious hits."""
    w = Wiki(cfg, log)
    ls, secs = w.lists(), w.lci_sections()
    excl = set(cfg["wiki"]["issue_exclude_sections"])
    al = w.aliases(sorted(set(ls["LCI"]) | set(ls["TAG"])))
    lci, tag = set(ls["LCI"]), set(ls["TAG"])
    rows = []
    for t, e in al.items():
        srcs = []
        if t in lci:
            sec = secs.get(t) or secs.get(e["page"])
            srcs.append("LCI_other" if sec in excl else "LCI_issue")
        if t in tag:
            srcs.append("TAG")
        names = [e["page"], t] + [r for r in e["redirects"] if len(_norm(r).split()) >= 2]
        for name in names:
            a = _norm(name)
            if len(a) >= MIN_ALIAS_CHARS and not a.isdigit():
                rows += [{"alias": a, "page": e["page"], "src": s} for s in srcs]
    df = pd.DataFrame(rows).drop_duplicates()
    stop = set(cfg["wiki"]["stop_aliases"])
    log.info(f"stop_aliases removed {int(df.alias.isin(stop).sum())} alias rows")
    df = df[~df.alias.isin(stop)]
    df.to_csv(os.path.join(w.dir, "controversial_aliases.csv"), index=False)
    log.info(f"aliases: {df.alias.nunique()} distinct, {df.page.nunique()} pages "
             f"({df.groupby('src').page.nunique().to_dict()})")
    return df


def _ngrams(text, n_max):
    toks = _norm(text).split()
    return {" ".join(toks[i:i + n]) for n in range(1, n_max + 1) for i in range(len(toks) - n + 1)}


MAX_NGRAM = 8   # longer aliases essentially never occur verbatim in a title


class Matcher:
    """alias n-gram -> set of controversial pages; built once, reused for every text."""

    def __init__(self, aliases):
        self.amap = aliases.groupby("alias").page.apply(set).to_dict()
        self.n_max = min(MAX_NGRAM, max(len(a.split()) for a in self.amap))

    def __call__(self, texts):
        out = []
        for t in texts:
            pages = set()
            for g in _ngrams(t or "", self.n_max) & self.amap.keys():
                pages |= self.amap[g]
            out.append(pages)
        return out


def article_features(cfg, log, cand):
    """Per candidate: issue pages hit by its topic keywords / title, TAG hits.
    Issue pages hit by more than `max_page_df` of all candidates are treated as
    corpus-generic context (logged to data/wiki/generic_pages.csv) and ignored."""
    al = alias_table(cfg, log)
    issue = Matcher(al[al.src == "LCI_issue"])
    tag = Matcher(al[al.src == "TAG"])
    topics = cand[["topic_id", "topic_rep"]].drop_duplicates("topic_id")
    kws = [[k.strip().replace("_", " ") for k in r.split(",") if k.strip()] for r in topics.topic_rep.fillna("")]
    t_issue = {t: set().union(*issue(k)) if k else set() for t, k in zip(topics.topic_id, kws)}
    t_tag = {t: set().union(*tag(k)) if k else set() for t, k in zip(topics.topic_id, kws)}
    ti_issue = issue(cand.title.tolist())
    ti_tag = tag(cand.title.tolist())
    a_topic = [t_issue[t] for t in cand.topic_id]
    df = pd.Series([pg for a, b in zip(a_topic, ti_issue) for pg in a | b]).value_counts() / len(cand)
    cap = cfg["wiki"]["max_page_df"]
    generic = df[df > cap]
    generic.rename("doc_freq").to_csv(os.path.join(p(cfg, "data_dir"), "wiki", "generic_pages.csv"))
    log.info(f"generic issue pages (> {cap:.0%} of candidates, ignored): {generic.round(3).to_dict()}")
    g = set(generic.index)
    a_topic = [x - g for x in a_topic]
    ti_issue = [x - g for x in ti_issue]
    return pd.DataFrame({
        "article_id": cand.article_id.values,
        "wiki_topic_n_issue": [len(x) for x in a_topic],
        "wiki_title_n_issue": [len(x) for x in ti_issue],
        "wiki_issue_pages": ["|".join(sorted(a | b)) for a, b in zip(a_topic, ti_issue)],
        "wiki_topic_n_tag": [len(t_tag[t]) for t in cand.topic_id],
        "wiki_title_n_tag": [len(x) for x in ti_tag],
    })


def main():
    """Warm the cache (the slow, rate-limited part) and print coverage."""
    cfg = load_config()
    log = start_run(cfg, "wiki")
    al = alias_table(cfg, log)
    log.info(f"examples: {al.sample(10, random_state=0).to_dict('records')}")


if __name__ == "__main__":
    main()
