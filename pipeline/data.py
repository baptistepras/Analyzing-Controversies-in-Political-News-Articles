"""Loading the label files (gold, silver, distant) and building the article table."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

REGIMES = ["gold", "silver", "distant"]
TEXT_COLUMNS = ["title", "lead_150w"]
LABELS = {"YES": 1, "Y": 1, "1": 1, "TRUE": 1, "NO": 0, "N": 0, "0": 0, "FALSE": 0}
PLACEHOLDER_RE = re.compile(r"(?:<copyright>\s*)+|<url>", re.IGNORECASE)


def read_label_file(path: Path, regime: str) -> pd.DataFrame:
    """Read one label file; an empty label is allowed (the article is scored but not evaluated)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    if "item_id" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{regime} file {path} needs the columns item_id and label; found {list(df.columns)}")
    df["item_id"] = df["item_id"].str.strip()
    dups = df.loc[df["item_id"].duplicated(), "item_id"].tolist()
    if dups:
        raise ValueError(f"{regime} file {path}: duplicate item_id values {dups[:10]}")
    raw = df["label"].str.strip().str.upper()
    mapped = raw.map(LABELS)
    bad = df.loc[(raw != "") & mapped.isna(), "item_id"].tolist()
    if bad:
        raise ValueError(f"{regime} file {path}: unreadable label (need YES or NO) for items {bad[:10]}")
    df["y"] = mapped  # float with NaN for the empty labels
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def load_regimes(
    paths: dict[str, Path], mask_placeholders: bool = False, max_articles: int | None = None
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    """Read the label files that were passed and return (articles, labeled frame per regime, notes).

    The article table is the union of the files, in the order gold, silver, distant. Every article is
    scored by the LLM systems once, whatever the regime. Each regime is evaluated only on its own labeled
    articles.
    """
    raw = {name: read_label_file(Path(paths[name]), name) for name in REGIMES if name in paths}
    notes: list[str] = []
    text: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for name, df in raw.items():
        for row in df.itertuples():
            title, lead = _norm(row.title), _norm(row.lead_150w)
            if not title or not lead:
                continue
            if row.item_id not in text:
                text[row.item_id] = {"title": row.title, "lead_150w": row.lead_150w}
            elif (_norm(text[row.item_id]["title"]), _norm(text[row.item_id]["lead_150w"])) != (title, lead):
                notes.append(f"{row.item_id}: the text in the {name} file differs from the first file; the first one is used")
        for item in df["item_id"]:
            if item not in order:
                order.append(item)
    no_text = [i for i in order if i not in text]
    if no_text:
        raise ValueError(f"no title and lead for {len(no_text)} articles (first: {no_text[:5]}). "
                         "At least one of the files must contain title and lead_150w for every article.")
    articles = pd.DataFrame({"item_id": order,
                             "title": [text[i]["title"] for i in order],
                             "lead_150w": [text[i]["lead_150w"] for i in order]})
    if mask_placeholders:
        for col in TEXT_COLUMNS:
            articles[col] = articles[col].str.replace(PLACEHOLDER_RE, "[...]", regex=True)
    if max_articles:
        articles = articles.head(max_articles).reset_index(drop=True)
    window = set(articles["item_id"])

    regimes: dict[str, pd.DataFrame] = {}
    for name, df in raw.items():
        inside = df[df["item_id"].isin(window)]
        labeled = inside.dropna(subset=["y"]).copy()
        if len(inside) - len(labeled):
            notes.append(f"{name}: {len(inside) - len(labeled)} articles have no label yet; they are scored but not evaluated")
        if labeled.empty:
            notes.append(f"{name}: no labeled article in the selected window, this regime is skipped")
            continue
        labeled["y"] = labeled["y"].astype(int)
        labeled = labeled.drop(columns=TEXT_COLUMNS).merge(articles, on="item_id", how="left")
        regimes[name] = labeled.reset_index(drop=True)
    return articles, regimes, notes


def _hash_order(ids: list[str]) -> list[str]:
    """A fixed pseudo-random order that does not depend on the file order or on the other articles."""
    return sorted(ids, key=lambda i: hashlib.sha1(i.encode()).hexdigest())


def pick_ablation_subset(
    articles: pd.DataFrame, primary: pd.DataFrame | None, n: int | None, mode: str = "minority_first"
) -> list[str]:
    """Choose the articles used by the ablation.

    minority_first keeps every article of the rarer class (as labeled in the primary regime), then fills
    with the other articles in a fixed hash order. random uses the hash order only. The choice is stable
    when more articles are added later, so earlier ablation calls stay useful.
    """
    ids = articles["item_id"].tolist()
    if n is None or n >= len(ids):
        return ids
    chosen: list[str] = []
    if mode == "minority_first" and primary is not None and len(primary):
        pos = 1 if primary["y"].mean() <= 0.5 else 0
        chosen = _hash_order(primary.loc[primary["y"] == pos, "item_id"].tolist())[:n]
    rest = _hash_order([i for i in ids if i not in set(chosen)])
    chosen += rest[: n - len(chosen)]
    keep = set(chosen)
    return [i for i in ids if i in keep]
