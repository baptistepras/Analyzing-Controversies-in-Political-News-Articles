"""Model backends (MLX, Ollama, mock) and the call store that logs and caches every call."""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    gen_tokens: int | None = None


class Backend(ABC):
    name: str = "base"
    model_id: str = ""

    @abstractmethod
    def generate(
        self, system: str | None, user: str, max_tokens: int, temp: float, seed: int
    ) -> Generation:
        """Run one chat turn and return the text with its timing."""

    def memory_gb(self) -> float | None:
        """Best-effort memory footprint of the loaded model, in GB."""
        return None

    def preflight(self) -> None:
        """Fail early with a readable message if the backend cannot run."""


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model_id: str):
        try:
            import mlx.core as mx
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError("mlx-lm is not installed here. Run: pip install mlx-lm") from exc
        self.model_id = model_id
        self._mx, self._generate, self._make_sampler = mx, generate, make_sampler
        print(f"Loading {model_id} via MLX ...")
        self.model, self.tokenizer = load(model_id)
        print("Model loaded.")

    def _count(self, text: str) -> int | None:
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return None

    def generate(self, system, user, max_tokens, temp, seed):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self._mx.random.seed(seed)
        start = time.perf_counter()
        text = self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=self._make_sampler(temp=temp),
            verbose=False,
        )
        latency = time.perf_counter() - start
        return Generation(text, latency, self._count(prompt), self._count(text))

    def memory_gb(self) -> float | None:
        # The peak getter moved between MLX versions.
        for getter in (
            lambda: self._mx.get_peak_memory(),
            lambda: self._mx.metal.get_peak_memory(),
        ):
            try:
                return getter() / 1e9
            except Exception:
                continue
        return None


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model_id: str, host: str = "http://localhost:11434", keep_alive: str = "60m"):
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive

    def _request(self, path: str, payload: dict | None = None, timeout: int = 600) -> dict:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data else {}
        last: Exception | None = None
        for attempt in range(3):
            req = urllib.request.Request(f"{self.host}{path}", data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                if exc.code < 500:
                    raise RuntimeError(f"Ollama answered {exc.code} on {path}: {body}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(
            f"Could not reach Ollama at {self.host}{path} ({last}). Is the server running?"
        )

    def preflight(self) -> None:
        tags = self._request("/api/tags", timeout=15)
        names = [m.get("name", "") for m in tags.get("models", [])]
        if self.model_id not in names and f"{self.model_id}:latest" not in names:
            raise RuntimeError(
                f"Model {self.model_id} is not installed in Ollama (found: {names}). "
                f"Run: ollama pull {self.model_id}"
            )
        print(f"Ollama OK at {self.host}, model {self.model_id} found.")

    def generate(self, system, user, max_tokens, temp, seed):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {"num_predict": max_tokens, "temperature": temp, "seed": seed},
        }
        start = time.perf_counter()
        data = self._request("/api/chat", payload)
        latency = time.perf_counter() - start
        return Generation(
            data["message"]["content"], latency, data.get("prompt_eval_count"), data.get("eval_count")
        )

    def memory_gb(self) -> float | None:
        try:
            info = self._request("/api/ps", timeout=10)
            for model in info.get("models", []):
                if model.get("name", "").split(":")[0] == self.model_id.split(":")[0]:
                    size = model.get("size_vram") or model.get("size") or 0
                    return size / 1e9 or None
        except Exception:
            return None
        return None


class MockBackend(Backend):
    """Fake model for tests: plausible JSON answers, sometimes malformed, no GPU needed."""

    name = "mock"

    def __init__(self, model_id: str = "mock", fail_rate: float = 0.05):
        self.model_id = model_id
        self.fail_rate = fail_rate

    def generate(self, system, user, max_tokens, temp, seed):
        digest = hashlib.sha256(f"{system}|{user}|{seed}".encode()).hexdigest()
        rng = random.Random(int(digest[:12], 16))
        if rng.random() < self.fail_rate:
            text = "Sorry, I would rather answer in plain words."
        elif '"label"' in user:
            hot = any(w in user.lower() for w in ("impeachment", "senate", "gun", "immigration", "abortion", "climate", "election"))
            text = json.dumps({"label": "YES" if rng.random() < (0.85 if hot else 0.15) else "NO",
                               "confidence": "medium", "reason": "mock reason"})
        elif '"summary"' in user:
            text = json.dumps({"summary": "mock summary", "controversy": round(rng.random(), 2)})
        elif '"stance"' in user:
            lean = 0
            for key, value in (("far left", -2), ("far right", 2), ("Republican", 1), ("Democratic", -1)):
                if system and key in system:
                    lean = value
            topic = int(hashlib.sha256(user[:80].encode()).hexdigest()[:4], 16) % 3 - 1
            stance = max(-1, min(1, round(0.5 * lean * topic + rng.gauss(0, 0.7))))
            text = json.dumps({"reaction": "mock reaction", "stance": stance})
        else:
            text = json.dumps({"controversy": round(rng.random(), 2), "reason": "mock reason"})
        return Generation(text, 0.001, 100, 30)


def make_backend(name: str, bcfg: dict, mock: bool = False) -> Backend:
    """Build the backend described in the config for one model."""
    if mock:
        return MockBackend(model_id=f"mock-{name}")
    if bcfg["backend"] == "mlx":
        return MLXBackend(bcfg["model"])
    if bcfg["backend"] == "ollama":
        return OllamaBackend(bcfg["model"], bcfg.get("host", "http://localhost:11434"))
    raise ValueError(f"unknown backend {bcfg['backend']!r}")


def read_records(path: Path) -> list[dict]:
    """Read a call log without opening it for writing; unreadable lines are skipped."""
    records: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def role_stats(records: list[dict], model_id: str) -> dict[str, dict]:
    """Per call role: calls, mean latency, mean prompt and generated tokens, share of retried calls."""
    groups: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("model") == model_id:
            groups.setdefault(rec.get("meta", {}).get("role", "?"), []).append(rec)
    out: dict[str, dict] = {}
    for role, rows in groups.items():

        def mean(key: str) -> float | None:
            values = [r[key] for r in rows if r.get(key) is not None]
            return sum(values) / len(values) if values else None

        retried = sum(1 for r in rows if r.get("meta", {}).get("attempt", 0) > 0)
        out[role] = {"n": len(rows), "latency_s": mean("latency_s"), "prompt_tokens": mean("prompt_tokens"),
                     "gen_tokens": mean("gen_tokens"), "retry_share": retried / len(rows)}
    return out


class CallStore:
    """Append-only JSONL log of every model call, which doubles as the resume cache."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict] = {}
        self._lock = threading.Lock()
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._records[record["key"]] = record
        self._handle = path.open("a", encoding="utf-8")

    @staticmethod
    def make_key(model_id: str, system: str | None, user: str, max_tokens: int, temp: float, seed: int) -> str:
        """Stable hash of everything that determines a model answer."""
        raw = json.dumps([model_id, system, user, max_tokens, temp, seed], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def call(
        self,
        backend: Backend,
        system: str | None,
        user: str,
        max_tokens: int,
        temp: float,
        seed: int,
        meta: dict,
    ) -> dict:
        """Return the cached answer for this exact call, or run it and log it."""
        key = self.make_key(backend.model_id, system, user, max_tokens, temp, seed)
        with self._lock:
            hit = self._records.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        gen = backend.generate(system, user, max_tokens, temp, seed)
        record = {
            "key": key,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": backend.name,
            "model": backend.model_id,
            "system": system,
            "user": user,
            "max_tokens": max_tokens,
            "temp": temp,
            "seed": seed,
            "response": gen.text,
            "latency_s": round(gen.latency_s, 4),
            "prompt_tokens": gen.prompt_tokens,
            "gen_tokens": gen.gen_tokens,
            "meta": meta,
        }
        with self._lock:
            self._records[key] = record
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._handle.flush()
        return {**record, "cached": False}

    def mean_latency(self, model_id: str) -> tuple[float | None, int]:
        """Mean logged latency for a model and the number of calls behind it."""
        values = [r["latency_s"] for r in self._records.values() if r["model"] == model_id]
        return (sum(values) / len(values) if values else None), len(values)

    def role_stats(self, model_id: str) -> dict[str, dict]:
        """Per call role statistics for one model, from the calls logged so far."""
        return role_stats(list(self._records.values()), model_id)

    def close(self) -> None:
        self._handle.close()
