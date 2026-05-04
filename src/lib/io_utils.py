"""
Incremental persistence (JSONL checkpoint) and logging utilities.
Enables resume-on-interrupt for long classification runs.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHECKPOINT_PATH = DATA_DIR / "classification_checkpoint.jsonl"
LOG_PATH = DATA_DIR / "classification_run.log"


def setup_logging():
    """Configure dual logging: console (INFO) + file (DEBUG)."""
    logger = logging.getLogger("classify")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(LOG_PATH, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    return logger


def load_checkpoint():
    """Load completed (uid, model) pairs from checkpoint file."""
    completed = set()
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    completed.add((entry["uid"], entry["model"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def append_checkpoint(uid, model, label, raw_response, latency_ms, backend):
    """Append a single classification result to the checkpoint file."""
    entry = {
        "uid": uid,
        "model": model,
        "label": label,
        "raw": raw_response,
        "latency_ms": latency_ms,
        "backend": backend,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_all_checkpoint_entries():
    """Load all checkpoint entries, grouped by uid."""
    entries_by_uid = {}
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    uid = entry["uid"]
                    if uid not in entries_by_uid:
                        entries_by_uid[uid] = {}
                    entries_by_uid[uid][entry["model"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return entries_by_uid
