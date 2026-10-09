"""Config loading, seeding and run logging shared by every step."""

import hashlib
import json
import logging
import os
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config.yaml")


def load_config(path=None):
    """config.yaml, or the file named by $GOLD_CONFIG (used by the tests)."""
    path = path or os.environ.get("GOLD_CONFIG", CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("data_dir", "logs_dir", "sample_dir"):
        os.makedirs(p(cfg, k), exist_ok=True)
    return cfg


def p(cfg, key):
    """Absolute path for cfg['paths'][key]."""
    return os.path.join(ROOT, cfg["paths"][key])


def sub_seed(cfg, name):
    """Deterministic per-step seed derived from the master seed, so adding a
    step never shifts the random stream of another one."""
    h = hashlib.sha256(f"{cfg['seed']}:{name}".encode()).hexdigest()
    return int(h[:8], 16)


def start_run(cfg, step):
    """Log to logs/<step>_<timestamp>.log and snapshot the config + seeds."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    logs = p(cfg, "logs_dir")
    log_path = os.path.join(logs, f"{step}_{ts}.log")
    logger = logging.getLogger(step)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    with open(os.path.join(logs, f"{step}_{ts}.config.json"), "w", encoding="utf-8") as f:
        json.dump({"step": step, "seed": cfg["seed"], "sub_seed": sub_seed(cfg, step),
                   "argv": sys.argv, "config": cfg}, f, indent=2, default=str)
    return logger
