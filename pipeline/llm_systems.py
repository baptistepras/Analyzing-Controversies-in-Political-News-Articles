"""LLM systems: single agent, audience simulation, direct aggregator, deliberation, ablations."""

from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import prompts
from .backends import Backend, CallStore

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional
    tqdm = None

MAIN_SYSTEMS = ["llm_single", "audience", "audience_direct"]
ABLATION_SYSTEMS = ["audience_deliberative", "ablation_homogeneous", "ablation_selfconsistency"]
ABLATION_ALIASES = {
    "homogeneous": "ablation_homogeneous",
    "deliberative": "audience_deliberative",
    "selfconsistency": "ablation_selfconsistency",
}


def resolve_ablations(names: list[str]) -> list[str]:
    """Turn short names (homogeneous, ...) or full system ids into the list of ablation system ids."""
    out = []
    for name in names:
        system = ABLATION_ALIASES.get(name, name)
        if system not in ABLATION_SYSTEMS:
            raise ValueError(f"unknown ablation {name!r}; choose from {sorted(ABLATION_ALIASES)}")
        out.append(system)
    return out


@dataclass(frozen=True)
class Article:
    item_id: str
    title: str
    lead: str


@dataclass(frozen=True)
class Run:
    paraphrase: int
    seed: int


@dataclass
class LLMContext:
    backbone: str
    backend: Backend
    store: CallStore
    settings: dict


def derive_seed(*parts: object) -> int:
    """Stable per-call seed so reruns reproduce the same sampling."""
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def extract_json(text: str) -> dict | None:
    """Parse the first {...} block of a model answer, or None."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


STANCE_RE = re.compile(r'"stance"\s*:\s*"?(-?\d)"?')
# Qwen often writes "controversity"; accept any key that starts with "controvers".
SCORE_RE = re.compile(r'"controvers\w*"\s*:\s*"?([0-9]*\.?[0-9]+)', re.IGNORECASE)


def parse_stance(text: str) -> dict | None:
    """Read {reaction, stance} from an answer; stance is clamped to -1, 0 or 1."""
    obj = extract_json(text)
    if obj is not None and "stance" in obj:
        raw, reaction = obj["stance"], str(obj.get("reaction", "")).strip()
    else:
        match = STANCE_RE.search(text)
        if not match:
            return None
        raw, reaction = match.group(1), text.strip().replace("\n", " ")[:160]
    try:
        value = int(round(float(raw)))
    except (TypeError, ValueError):
        return None
    return {"stance": max(-1, min(1, value)), "reaction": reaction}


def parse_score(text: str) -> dict | None:
    """Read a controversy score in [0, 1] plus the extra fields (summary or reason)."""
    obj = extract_json(text)
    extra: dict[str, str] = {}
    key = next((k for k in obj if str(k).lower().startswith("controvers")), None) if obj else None
    if key is not None:
        try:
            value = float(obj[key])
        except (TypeError, ValueError):
            return None
        extra = {k: str(v).strip() for k, v in obj.items() if k != key}
    else:
        match = SCORE_RE.search(text)
        if not match:
            return None
        value = float(match.group(1))
    if value > 1:  # the model answered on a 0 to 100 scale
        value = value / 100 if value <= 100 else 1.0
    return {"controversy": max(0.0, min(1.0, value)), **extra}


def entropy_score(stances: list[int | None], min_valid: int) -> float:
    """C(a) = H(pi_hat) / log 3 over the valid stances; NaN if too few answers were readable."""
    valid = [s for s in stances if s is not None]
    if len(valid) < min_valid:
        return float("nan")
    counts = np.bincount([s + 1 for s in valid], minlength=3)
    p = counts[counts > 0] / len(valid)
    return float(-(p * np.log(p)).sum() / np.log(3))


def ask(
    ctx: LLMContext,
    system: str | None,
    user: str,
    parse: Callable[[str], dict | None],
    kind: str,
    temp: float,
    seed_parts: tuple,
    meta: dict,
) -> tuple[dict | None, list[dict], int]:
    """Call the model, retrying with a fresh seed when the answer cannot be parsed."""
    max_tokens = ctx.settings["max_tokens"][kind]
    records: list[dict] = []
    failures = 0
    parsed = None
    for attempt in range(ctx.settings["max_retries"] + 1):
        seed = derive_seed(*seed_parts, attempt)
        record = ctx.store.call(
            ctx.backend, system, user, max_tokens, temp, seed, {**meta, "kind": kind, "attempt": attempt}
        )
        records.append(record)
        parsed = parse(record["response"])
        if parsed is not None:
            break
        failures += 1
    return parsed, records, failures


def collect_stances(ctx: LLMContext, art: Article, run: Run, agents: list[dict], role: str) -> list[dict]:
    """One stance call per agent entry; an entry may repeat the same persona for independent draws."""
    out = []
    for draw, agent in enumerate(agents):
        user = prompts.stance_prompt(run.paraphrase, art.title, art.lead)
        meta = {"item_id": art.item_id, "role": role, "agent": agent["name"], "draw": draw,
                "paraphrase": run.paraphrase, "run_seed": run.seed}
        parsed, records, failures = ask(
            ctx, agent["system"], user, parse_stance, "stance", ctx.settings["temp_persona"],
            (run.seed, art.item_id, role, agent["name"], draw), meta,
        )
        out.append({
            "name": agent["name"], "short": agent["short"], "system": agent["system"],
            "stance": parsed["stance"] if parsed else None,
            "reaction": parsed["reaction"] if parsed else "",
            "records": records, "failures": failures,
        })
    return out


def deliberate(ctx: LLMContext, art: Article, run: Run, first: list[dict]) -> list[dict]:
    """Second round: each persona sees the others' reactions and may revise its stance."""
    out = []
    for i, me in enumerate(first):
        if me["stance"] is None:
            out.append({"stance": None, "records": [], "failures": 0})
            continue
        others = [x for j, x in enumerate(first) if j != i and x["stance"] is not None]
        user = prompts.deliberation_prompt(run.paraphrase, art.title, art.lead, me, others)
        meta = {"item_id": art.item_id, "role": "delib", "agent": me["name"],
                "paraphrase": run.paraphrase, "run_seed": run.seed}
        parsed, records, failures = ask(
            ctx, me["system"], user, parse_stance, "delib", ctx.settings["temp_persona"],
            (run.seed, art.item_id, "delib", me["name"]), meta,
        )
        out.append({"stance": parsed["stance"] if parsed else me["stance"],
                    "records": records, "failures": failures})
    return out


def run_article(ctx: LLMContext, art: Article, run: Run, systems: set[str]) -> list[dict]:
    """Run the requested systems on one article for one (paraphrase, seed) run."""
    s = ctx.settings
    min_valid = math.ceil(s["n_draws"] / 2)
    results: list[dict] = []

    def finish(system: str, score: float | None, records: list[dict], failures: int, extra: dict) -> None:
        if score is not None and math.isnan(score):
            score = None
        results.append({
            "backbone": ctx.backbone, "system": system, "paraphrase": run.paraphrase,
            "seed": run.seed, "item_id": art.item_id, "score": score,
            "n_calls": len(records), "n_parse_fail": failures,
            "latency_s": round(sum(r["latency_s"] for r in records), 4), **extra,
        })

    base_meta = {"item_id": art.item_id, "paraphrase": run.paraphrase, "run_seed": run.seed}

    if "llm_single" in systems:
        user = prompts.baseline_prompt(run.paraphrase, art.title, art.lead)
        parsed, records, failures = ask(
            ctx, None, user, parse_score, "judge", s["temp_judge"],
            (run.seed, art.item_id, "baseline"), {**base_meta, "role": "baseline"},
        )
        finish("llm_single", parsed["controversy"] if parsed else None, records, failures,
               {"reason": parsed.get("reason", "") if parsed else ""})

    if systems & {"audience", "audience_direct", "audience_deliberative"}:
        first = collect_stances(ctx, art, run, prompts.PERSONAS, "persona")
        r1_records = [r for x in first for r in x["records"]]
        r1_failures = sum(x["failures"] for x in first)
        stances = {x["name"]: x["stance"] for x in first}
        reactions = {x["name"]: x["reaction"] for x in first}

        if "audience" in systems:
            finish("audience", entropy_score(list(stances.values()), min_valid), r1_records,
                   r1_failures, {"stances": stances, "reactions": reactions})

        if "audience_direct" in systems:
            readers = [x for x in first if x["stance"] is not None]
            score, summary, records, failures = None, "", [], 0
            if readers:
                user = prompts.aggregate_prompt(run.paraphrase, art.title, readers)
                parsed, records, failures = ask(
                    ctx, None, user, parse_score, "judge", s["temp_judge"],
                    (run.seed, art.item_id, "aggregate"), {**base_meta, "role": "aggregate"},
                )
                if parsed:
                    score, summary = parsed["controversy"], parsed.get("summary", "")
            finish("audience_direct", score, r1_records + records, r1_failures + failures,
                   {"summary": summary, "stances": stances, "reactions": reactions})

        if "audience_deliberative" in systems:
            second = deliberate(ctx, art, run, first)
            final = {x["name"]: y["stance"] for x, y in zip(first, second)}
            changed = sum(1 for x, y in zip(first, second)
                          if x["stance"] is not None and y["stance"] != x["stance"])
            d_records = [r for y in second for r in y["records"]]
            finish("audience_deliberative", entropy_score(list(final.values()), min_valid),
                   r1_records + d_records, r1_failures + sum(y["failures"] for y in second),
                   {"stances": final, "first_stances": stances, "n_changed": changed})

    if "ablation_homogeneous" in systems:
        persona = next(p for p in prompts.PERSONAS if p["name"] == s["homogeneous_persona"])
        draws = collect_stances(ctx, art, run, [persona] * s["n_draws"], "homog")
        finish("ablation_homogeneous", entropy_score([x["stance"] for x in draws], min_valid),
               [r for x in draws for r in x["records"]], sum(x["failures"] for x in draws),
               {"stances": [x["stance"] for x in draws]})

    if "ablation_selfconsistency" in systems:
        draws = collect_stances(ctx, art, run, [prompts.GENERIC_READER] * s["n_draws"], "selfcons")
        finish("ablation_selfconsistency", entropy_score([x["stance"] for x in draws], min_valid),
               [r for x in draws for r in x["records"]], sum(x["failures"] for x in draws),
               {"stances": [x["stance"] for x in draws]})

    return results


def load_done(path: Path) -> set[tuple]:
    """Keys (system, paraphrase, seed, item_id) already present in a results file."""
    done: set[tuple] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["system"], r["paraphrase"], r["seed"], r["item_id"]))
    return done


def run_llm_stage(
    ctx: LLMContext,
    articles: list[Article],
    runs: list[Run],
    systems: list[str],
    results_path: Path,
    workers: int,
    label: str,
) -> None:
    """Run every (run, article) pair for the given systems, skipping what is already done."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(results_path)
    bar = tqdm(total=len(runs) * len(articles), desc=label, unit="article-run") if tqdm else None

    def tick() -> None:
        if bar:
            bar.update(1)

    with results_path.open("a", encoding="utf-8") as out:

        def write(rows: list[dict]) -> None:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                done.add((row["system"], row["paraphrase"], row["seed"], row["item_id"]))
            out.flush()

        for run in runs:
            pending = []
            for art in articles:
                todo = {s for s in systems if (s, run.paraphrase, run.seed, art.item_id) not in done}
                if todo:
                    pending.append((art, todo))
                else:
                    tick()
            if workers <= 1:
                for art, todo in pending:
                    write(run_article(ctx, art, run, todo))
                    tick()
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(run_article, ctx, art, run, todo) for art, todo in pending]
                    for future in as_completed(futures):
                        write(future.result())
                        tick()
    if bar:
        bar.close()
