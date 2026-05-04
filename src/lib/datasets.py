"""
Unified prompt loader across 4 datasets.
Normalizes all prompts into a common schema.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq


DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_rmcbench():
    """Load 473 code-generation prompts from RMCBench."""
    path = DATA_DIR / "rmcbench" / "prompts_code_only.json"
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for item in data:
        prompts.append({
            "uid": f"rmcbench_{item.get('pid', len(prompts)+1):04d}",
            "prompt": item["prompt"],
            "source_dataset": "rmcbench",
            "category": item.get("malicious_categories", item.get("malicious categories", "unknown")),
            "original_id": item.get("pid"),
            "metadata": {
                "task": item.get("task", ""),
                "level": item.get("level"),
                "category_type": item.get("category", ""),
            },
        })
    return prompts


def _load_malwarebench():
    """Load 320 code-generation prompts from MalwareBench."""
    path = DATA_DIR / "malwarebench" / "prompts_code_only.json"
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for i, item in enumerate(data):
        prompts.append({
            "uid": f"malwarebench_{i+1:04d}",
            "prompt": item["original_question"],
            "source_dataset": "malwarebench",
            "category": item.get("level1", "unknown"),
            "original_id": i + 1,
            "metadata": {
                "level1": item.get("level1", ""),
                "level2": item.get("level2", ""),
                "level3": item.get("level3", ""),
                "type": item.get("type", ""),
            },
        })
    return prompts


def _load_cysecbench():
    """Load 1,820 filtered code-generation prompts from CySecBench."""
    path = DATA_DIR / "cysecbench" / "prompts_code_only.json"
    with open(path) as f:
        data = json.load(f)
    prompts = []
    for i, item in enumerate(data):
        prompts.append({
            "uid": f"cysecbench_{i+1:05d}",
            "prompt": item["prompt"],
            "source_dataset": "cysecbench",
            "category": item.get("category", "unknown"),
            "original_id": i + 1,
            "metadata": {
                "source_file": item.get("source_file", ""),
            },
        })
    return prompts


def _load_harmful_behaviors():
    """Load 520 content-safety prompts from harmful_behaviors."""
    base = DATA_DIR / "harmful_behaviors" / "data"
    train = pq.read_table(base / "train-00000-of-00001.parquet").to_pandas()
    test = pq.read_table(base / "test-00000-of-00001.parquet").to_pandas()

    prompts = []
    for i, row in train.iterrows():
        prompts.append({
            "uid": f"harmful_behaviors_{i+1:04d}",
            "prompt": row["text"],
            "source_dataset": "harmful_behaviors",
            "category": "content_safety",
            "original_id": i + 1,
            "metadata": {"split": "train"},
        })
    for i, row in test.iterrows():
        idx = len(train) + i + 1
        prompts.append({
            "uid": f"harmful_behaviors_{idx:04d}",
            "prompt": row["text"],
            "source_dataset": "harmful_behaviors",
            "category": "content_safety",
            "original_id": idx,
            "metadata": {"split": "test"},
        })
    return prompts


def load_all_prompts():
    """Load all prompts from all 4 datasets. Returns list of unified dicts."""
    rmcbench = _load_rmcbench()
    malwarebench = _load_malwarebench()
    cysecbench = _load_cysecbench()
    harmful = _load_harmful_behaviors()

    all_prompts = rmcbench + malwarebench + cysecbench + harmful

    counts = {
        "rmcbench": len(rmcbench),
        "malwarebench": len(malwarebench),
        "cysecbench": len(cysecbench),
        "harmful_behaviors": len(harmful),
        "total": len(all_prompts),
    }

    return all_prompts, counts
