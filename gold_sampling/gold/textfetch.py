"""Fetch title + 150-word lead for a handful of drawn articles.

Parquet files are sorted by article_id in 1000-row groups, so only the row
groups containing the requested ids are read (a few MB each). The text goes
straight into the annotation files; nothing else is cached.
"""

import os

import pandas as pd
import pyarrow.parquet as pq

from .common import p

LEAD_WORDS = 150


def _rowgroup_index(cfg, fs, files):
    path = os.path.join(p(cfg, "data_dir"), "rowgroup_index.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    rows = []
    for f in files:
        with fs.open(f, "rb") as fh:
            md = pq.ParquetFile(fh).metadata
            i = md.schema.to_arrow_schema().names.index("article_id")
            for g in range(md.num_row_groups):
                st = md.row_group(g).column(i).statistics
                rows.append({"file": f, "rg": g, "min": st.min, "max": st.max})
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def fetch(cfg, article_ids):
    from huggingface_hub import HfApi, HfFileSystem
    repo = cfg["paths"]["hf_dataset"]
    fs = HfFileSystem()
    files = [f"datasets/{repo}/{f}" for f in HfApi().list_repo_files(repo, repo_type="dataset")
             if f.startswith("data/") and f.endswith(".parquet")]
    idx = _rowgroup_index(cfg, fs, sorted(files))
    want = sorted(set(article_ids))
    need = {}
    for a in want:
        hit = idx[(idx["min"] <= a) & (idx["max"] >= a)]
        if hit.empty:
            raise KeyError(f"{a} not found in any row group")
        r = hit.iloc[0]
        need.setdefault((r.file, int(r.rg)), []).append(a)
    out = []
    for (f, g), ids in sorted(need.items()):
        with fs.open(f, "rb") as fh:
            t = pq.ParquetFile(fh).read_row_group(g, columns=["article_id", "title", "content"]).to_pandas()
        t = t[t.article_id.isin(ids)]
        t["lead_150w"] = t.content.fillna("").str.split().str[:LEAD_WORDS].str.join(" ")
        out.append(t[["article_id", "title", "lead_150w"]])
    res = pd.concat(out, ignore_index=True)
    missing = set(want) - set(res.article_id)
    if missing:
        raise KeyError(f"text not found for {sorted(missing)[:5]}")
    return res
