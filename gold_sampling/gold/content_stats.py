"""Step 1 (content pass): stream every parquet file of the HF dataset once and
keep only per-article numbers. No article text is written to disk.

Output: data/content_stats/<file>.parquet with
  article_id, domain, raw_url, n_words, n_chars, n_mask, mask_share, n_<token>
One output per input file, so an interrupted run resumes where it stopped.
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .common import load_config, p, start_run

COLUMNS = ["article_id", "domain", "raw_url", "content"]


def _files(cfg):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(cfg["paths"]["hf_dataset"], repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/") and f.endswith(".parquet"))


def _stats(tbl, tokens):
    content = pc.fill_null(tbl["content"], "")
    n_words = pc.list_value_length(pc.utf8_split_whitespace(content))
    cols = {
        "article_id": tbl["article_id"],
        "domain": tbl["domain"],
        "raw_url": tbl["raw_url"],
        "n_words": n_words,
        "n_chars": pc.utf8_length(content),
    }
    n_mask = None
    for t in tokens:
        c = pc.count_substring(content, t)
        cols["n_" + t.strip("<>")] = c
        n_mask = c if n_mask is None else pc.add(n_mask, c)
    cols["n_mask"] = n_mask
    cols["mask_share"] = pc.divide(pc.cast(n_mask, pa.float64()),
                                   pc.max_element_wise(pc.cast(n_words, pa.float64()), 1.0))
    return pa.table(cols)


def one_file(repo, fname, out, tokens, batch_size):
    from datasets import load_dataset
    ds = load_dataset("parquet", data_files={"train": f"hf://datasets/{repo}/{fname}"},
                      split="train", streaming=True, columns=COLUMNS).with_format("arrow")
    parts = [_stats(b, tokens) for b in ds.iter(batch_size=batch_size)]
    tbl = pa.concat_tables(parts)
    pq.write_table(tbl, out + ".tmp")
    os.replace(out + ".tmp", out)
    return tbl.num_rows


def main():
    cfg = load_config()
    log = start_run(cfg, "content_stats")
    cc = cfg["content"]
    outdir = os.path.join(p(cfg, "data_dir"), "content_stats")
    os.makedirs(outdir, exist_ok=True)
    todo = []
    for f in _files(cfg):
        out = os.path.join(outdir, os.path.basename(f))
        if not os.path.exists(out):
            todo.append((f, out))
    log.info(f"{len(todo)} parquet files to stream (content column read, never stored)")
    t0 = time.time()
    with ProcessPoolExecutor(cc["workers"]) as ex:
        futs = {ex.submit(one_file, cfg["paths"]["hf_dataset"], f, out,
                          cc["mask_tokens"], cc["batch_size"]): f for f, out in todo}
        for fut in as_completed(futs):
            log.info(f"{futs[fut]}: {fut.result()} rows ({time.time()-t0:.0f}s elapsed)")
    log.info("done")


if __name__ == "__main__":
    main()
