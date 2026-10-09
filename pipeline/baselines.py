"""Supervised baselines trained by cross-validation, plus the DA-RoBERTa zero-shot transfer test."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold



def scores_file(regime: str) -> str:
    """Name of the baseline score file for one label regime."""
    return f"baseline_scores_{regime}.csv"


def runs_file(regime: str) -> str:
    """Name of the baseline timing file for one label regime."""
    return f"baseline_runs_{regime}.csv"


def article_texts(df: pd.DataFrame) -> list[str]:
    """Title and lead as one string, for the bag-of-words model."""
    return (df["title"] + ". " + df["lead_150w"]).tolist()


def topic_groups(df: pd.DataFrame, seed: int) -> np.ndarray:
    """Cluster articles by vocabulary so that similar topics stay in the same CV fold."""
    X = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(article_texts(df))
    k = min(15, max(3, len(df) // 12))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


def make_folds(df: pd.DataFrame, n_splits: int, seed: int, mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified train/test index pairs; every article is tested exactly once."""
    y = df["y"].to_numpy()
    n_splits = min(n_splits, int(np.bincount(y, minlength=2).min()))
    if n_splits < 2:
        raise ValueError("fewer than 2 articles in the rarer class, cross-validation is impossible")
    if mode == "topic_grouped":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(df, y, topic_groups(df, seed)))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(df, y))


def tfidf_oof(df: pd.DataFrame, folds: list) -> np.ndarray:
    """Out-of-fold P(controversial) from TF-IDF features and logistic regression."""
    texts, y = np.array(article_texts(df), dtype=object), df["y"].to_numpy()
    oof = np.zeros(len(df))
    for train, test in folds:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                              max_features=20000, strip_accents="unicode")
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        clf.fit(vec.fit_transform(texts[train]), y[train])
        oof[test] = clf.predict_proba(vec.transform(texts[test]))[:, 1]
    return oof


def torch_device():
    """Apple GPU if available, else CPU."""
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def gpu_memory_gb() -> float | None:
    """Memory held by the Apple GPU allocator, if torch can report it."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / 1e9
    except Exception:
        return None
    return None


def encode_articles(tok, df: pd.DataFrame, max_len: int) -> list[dict]:
    """Tokenize title and lead as a sentence pair, truncated to max_len."""
    enc = tok(df["title"].tolist(), df["lead_150w"].tolist(), truncation="longest_first", max_length=max_len)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]} for i in range(len(df))]


def predict_proba(model, tok, items: list[dict], batch_size: int, device) -> np.ndarray:
    """P(class 1) for every encoded item."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = tok.pad(items[i:i + batch_size], return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def finetune_fold(model_name: str, tok, items: list[dict], y: np.ndarray, train: np.ndarray,
                  test: np.ndarray, cfg: dict, seed: int, device) -> np.ndarray:
    """Fine-tune a fresh classifier on the train indices and return P(controversial) on the test ones."""
    import torch
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)
    counts = np.maximum(np.bincount(y[train], minlength=2), 1)
    weights = torch.tensor(len(train) / (2.0 * counts), dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    bs, epochs = cfg["batch_size"], cfg["epochs"]
    steps = epochs * math.ceil(len(train) / bs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train), generator=generator).tolist()
        for i in range(0, len(order), bs):
            idx = [train[j] for j in order[i:i + bs]]
            batch = tok.pad([items[j] for j in idx], return_tensors="pt")
            labels = torch.tensor(y[idx], device=device)
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    probs = predict_proba(model, tok, [items[j] for j in test], bs, device)
    del model, optimizer
    if device.type == "mps":
        torch.mps.empty_cache()
    return probs


def finetune_oof(df: pd.DataFrame, folds: list, model_name: str, cfg: dict, seed: int) -> np.ndarray:
    """Out-of-fold P(controversial) from a fine-tuned encoder."""
    from transformers import AutoTokenizer

    device = torch_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    items = encode_articles(tok, df, cfg["max_len"])
    y = df["y"].to_numpy()
    oof = np.zeros(len(df))
    for fold, (train, test) in enumerate(folds):
        print(f"    fold {fold + 1}/{len(folds)}")
        oof[test] = finetune_fold(model_name, tok, items, y, train, test, cfg, seed * 100 + fold, device)
    return oof


def resolve_positive_label(id2label: dict, wanted: int | None) -> int:
    """Index of the 'biased' output of the BABE checkpoint (config value, else a guess from the labels)."""
    if wanted is not None:
        return int(wanted)
    for idx, name in id2label.items():
        low = str(name).lower()
        if "bias" in low and not any(w in low for w in ("non", "un", "neutral", "no_")):
            return int(idx)
    print("    the label names are generic; using output 1 (BABE convention: 1 = biased), checked by the probe below")
    return 1


def transfer_scores(df: pd.DataFrame, model_name: str, positive: int | None, cfg: dict) -> np.ndarray:
    """P(biased) from the BABE-fine-tuned DA-RoBERTa, used as is as a controversy score."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    print(f"    checkpoint labels: {model.config.id2label}")
    idx = resolve_positive_label(model.config.id2label, positive)
    print(f"    using output {idx} ({model.config.id2label.get(idx)}) as the score")
    items = encode_articles(tok, df, cfg["max_len"])
    model.eval()
    probes = ["The city council met on Tuesday to discuss next year's budget.",
              "The radical left is shamelessly wrecking the country with its disgraceful, reckless agenda."]
    with torch.no_grad():
        enc = tok(probes, return_tensors="pt", padding=True, truncation=True, max_length=cfg["max_len"])
        p_neutral, p_loaded = torch.softmax(model(**{k: v.to(device) for k, v in enc.items()}).logits, dim=-1)[:, idx].tolist()
    print(f"    probe: P(output {idx}) is {p_neutral:.2f} for a neutral sentence and {p_loaded:.2f} for a loaded one")
    if p_loaded <= p_neutral:
        print("    WARNING: the loaded sentence does not score higher, so the index may be inverted. "
              "Set da_positive_label in config.yaml and rerun with --force-baselines.")
    out = []
    with torch.no_grad():
        for i in range(0, len(items), cfg["batch_size"]):
            batch = tok.pad(items[i:i + cfg["batch_size"]], return_tensors="pt")
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            out.append(torch.softmax(logits, dim=-1)[:, idx].cpu().numpy())
    return np.concatenate(out)


def run_baselines(df: pd.DataFrame, cfg: dict, prof: dict, out_dir: Path, regime: str, force: bool = False) -> None:
    """Compute out-of-fold scores for every enabled baseline on one label regime and save them."""
    bcfg = cfg["baselines"]
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path, runs_path = out_dir / scores_file(regime), out_dir / runs_file(regime)
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame(columns=["system", "run", "item_id", "score"])
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame(columns=["system", "run", "seconds", "gpu_gb"])
    prof_cfg = {**bcfg, "epochs": prof.get("roberta_epochs", bcfg["epochs"])}

    for system in bcfg["enabled"]:
        if not force and system in set(runs["system"]):
            print(f"  {system} ({regime}): already computed, skipping (use --force-baselines to redo)")
            continue
        print(f"  {system} ({regime} labels)")
        scores = scores[scores["system"] != system]
        runs = runs[runs["system"] != system]
        new_scores, new_runs = [], []
        try:
            if system == "da_roberta_transfer":
                start = time.perf_counter()
                s = transfer_scores(df, bcfg["da_roberta_model"], bcfg.get("da_positive_label"), prof_cfg)
                jobs = [(0, s, time.perf_counter() - start)]
            else:
                jobs = []
                for seed in prof["cv_seeds"]:
                    folds = make_folds(df, prof["cv_folds"], seed, bcfg["cv_mode"])
                    start = time.perf_counter()
                    if system == "tfidf":
                        s = tfidf_oof(df, folds)
                    elif system == "roberta_ft":
                        s = finetune_oof(df, folds, bcfg["roberta_model"], prof_cfg, seed)
                    elif system == "da_roberta_ft":
                        s = finetune_oof(df, folds, bcfg["da_roberta_model"], prof_cfg, seed)
                    else:
                        raise ValueError(f"unknown baseline {system}")
                    jobs.append((seed, s, time.perf_counter() - start))
        except Exception as exc:  # one broken baseline must not block the long LLM runs
            print(f"    skipped: {type(exc).__name__}: {exc}")
            continue
        for run_id, s, seconds in jobs:
            new_scores.append(pd.DataFrame({"system": system, "run": run_id, "item_id": df["item_id"], "score": s}))
            new_runs.append({"system": system, "run": run_id, "seconds": seconds, "gpu_gb": gpu_memory_gb()})
        scores = pd.concat([scores, *new_scores], ignore_index=True)
        runs = pd.concat([runs, pd.DataFrame(new_runs)], ignore_index=True)
        scores.to_csv(scores_path, index=False)
        runs.to_csv(runs_path, index=False)
