"""
Multi-LLM Prompt Classification Pipeline
Classifies 3,133 prompts as CODE (weapons) vs KNOWLEDGE using 5 LLM judges.

Usage:
  export OPENROUTER_API_KEY="sk-or-..."
  python3 scripts/classify_prompts.py              # Full run
  python3 scripts/classify_prompts.py --smoke 10   # Smoke test with 10 prompts
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path for lib imports
sys.path.insert(0, str(Path(__file__).parent))

from lib.backends import classify_one, preflight_check, CLASSIFICATION_PROMPT
from lib.datasets import load_all_prompts
from lib.io_utils import (
    setup_logging,
    load_checkpoint,
    append_checkpoint,
    load_all_checkpoint_entries,
    DATA_DIR,
)
from lib.fleiss_kappa import build_rating_matrix, bootstrap_ci, interpret_kappa

# ============================================================
# JUDGE PANEL — 5 models, 5 providers, 2 coders + 3 general
# ============================================================
JUDGES = [
    {
        "id": "anthropic/claude-sonnet-4.6",
        "backend": "openrouter",
        "short": "claude-sonnet-4.6",
        "type": "general",
        "provider": "Anthropic",
    },
    {
        "id": "openai/gpt-5.3-codex",
        "backend": "openrouter",
        "short": "gpt-5.3-codex",
        "type": "coder",
        "provider": "OpenAI",
    },
    {
        "id": "gemini-3-flash-preview:cloud",
        "backend": "ollama",
        "short": "gemini-3-flash",
        "type": "general",
        "provider": "Google",
    },
    {
        "id": "glm-5:cloud",
        "backend": "ollama",
        "short": "glm-5",
        "type": "general",
        "provider": "Zhipu AI",
    },
    {
        "id": "qwen3-coder-next:cloud",
        "backend": "ollama",
        "short": "qwen3-coder-next",
        "type": "coder",
        "provider": "Alibaba",
    },
]

RATE_LIMIT_SECONDS = 0.3


def compute_consensus(classifications, judge_short_names):
    """Compute majority vote from 5 judge labels."""
    labels = []
    for j in judge_short_names:
        label = classifications.get(j, "")
        if label in ("CODE", "KNOWLEDGE"):
            labels.append(label)

    if not labels:
        return "ERROR", "0/0", 0, 0

    counts = Counter(labels)
    majority_label = counts.most_common(1)[0][0]
    majority_count = counts.most_common(1)[0][1]
    total_valid = len(labels)

    n_code = counts.get("CODE", 0)
    n_knowledge = counts.get("KNOWLEDGE", 0)

    if majority_count == total_valid:
        agreement = f"{total_valid}/{total_valid}"
    else:
        agreement = f"{majority_count}/{total_valid}"

    return majority_label, agreement, n_code, n_knowledge


def run_classification(prompts, judges, api_key, logger, rate_limit=0.3):
    """Run classification across all prompts and judges with checkpoint resume."""
    completed = load_checkpoint()
    total_calls = len(prompts) * len(judges)
    done_calls = len(completed)
    remaining = total_calls - done_calls

    logger.info(f"Total: {len(prompts)} prompts x {len(judges)} judges = {total_calls} calls")
    if done_calls > 0:
        logger.info(f"Resuming: {done_calls} already done, {remaining} remaining")

    call_count = 0
    start_time = time.time()

    for i, prompt_data in enumerate(prompts):
        uid = prompt_data["uid"]
        prompt_text = prompt_data["prompt"]

        for judge in judges:
            model_id = judge["id"]
            short = judge["short"]

            if (uid, model_id) in completed:
                continue

            result = classify_one(
                prompt_text, model_id, judge["backend"], api_key
            )

            append_checkpoint(
                uid=uid,
                model=model_id,
                label=result["label"],
                raw_response=result.get("raw_response", ""),
                latency_ms=result.get("latency_ms", 0),
                backend=judge["backend"],
            )

            call_count += 1
            elapsed = time.time() - start_time
            rate = call_count / elapsed if elapsed > 0 else 0

            if call_count % 25 == 0 or call_count <= 5:
                logger.info(
                    f"  [{done_calls + call_count}/{total_calls}] "
                    f"{uid} x {short} = {result['label']} "
                    f"({result.get('latency_ms', 0)}ms) "
                    f"[{rate:.1f} calls/s]"
                )

            logger.debug(
                f"{uid} | {short} | {result['label']} | "
                f"{result.get('raw_response', '')[:50]} | "
                f"{result.get('latency_ms', 0)}ms"
            )

            time.sleep(rate_limit)

    total_elapsed = time.time() - start_time
    logger.info(f"Classification complete: {call_count} new calls in {total_elapsed:.0f}s")
    return call_count


def assemble_results(prompts, judges):
    """Assemble final results from checkpoint entries."""
    checkpoint_data = load_all_checkpoint_entries()
    judge_shorts = [j["short"] for j in judges]

    results = []
    for prompt_data in prompts:
        uid = prompt_data["uid"]
        entries = checkpoint_data.get(uid, {})

        classifications = {}
        for judge in judges:
            entry = entries.get(judge["id"], {})
            classifications[judge["short"]] = entry.get("label", "MISSING")

        consensus, agreement, n_code, n_knowledge = compute_consensus(
            classifications, judge_shorts
        )

        n_error = sum(
            1 for v in classifications.values()
            if v not in ("CODE", "KNOWLEDGE")
        )

        results.append({
            "uid": uid,
            "prompt": prompt_data["prompt"],
            "source_dataset": prompt_data["source_dataset"],
            "category": prompt_data["category"],
            "classifications": classifications,
            "consensus": consensus,
            "agreement": agreement,
            "n_code": n_code,
            "n_knowledge": n_knowledge,
            "n_error": n_error,
        })

    return results


def compute_summary(results, judges):
    """Compute summary statistics including Fleiss' kappa."""
    judge_shorts = [j["short"] for j in judges]
    n_total = len(results)

    # Agreement levels
    agreement_counts = Counter(r["agreement"] for r in results)

    # Consensus breakdown
    consensus_counts = Counter(r["consensus"] for r in results)

    # Per-model stats
    per_model = {}
    for judge in judges:
        short = judge["short"]
        labels = [r["classifications"].get(short, "") for r in results]
        per_model[short] = {
            "code": labels.count("CODE"),
            "knowledge": labels.count("KNOWLEDGE"),
            "error_or_unclear": sum(1 for l in labels if l not in ("CODE", "KNOWLEDGE")),
            "type": judge["type"],
            "provider": judge["provider"],
        }

    # Per-dataset stats
    per_dataset = {}
    for ds in ["rmcbench", "malwarebench", "cysecbench", "harmful_behaviors"]:
        subset = [r for r in results if r["source_dataset"] == ds]
        per_dataset[ds] = {
            "n": len(subset),
            "consensus_code": sum(1 for r in subset if r["consensus"] == "CODE"),
            "consensus_knowledge": sum(1 for r in subset if r["consensus"] == "KNOWLEDGE"),
        }

    # Fleiss' kappa (overall)
    rating_matrix, excluded = build_rating_matrix(results, judge_shorts)
    if len(rating_matrix) > 0:
        kappa_result = bootstrap_ci(rating_matrix, n_bootstrap=10000, seed=42)
        kappa_result["interpretation"] = interpret_kappa(kappa_result["kappa"])
        kappa_result["n_excluded"] = len(excluded)
        kappa_result["n_included"] = len(rating_matrix)
    else:
        kappa_result = {"kappa": None, "error": "no valid ratings"}

    # Per-dataset kappa
    for ds in per_dataset:
        subset = [r for r in results if r["source_dataset"] == ds]
        rm, ex = build_rating_matrix(subset, judge_shorts)
        if len(rm) > 0:
            ds_kappa = bootstrap_ci(rm, n_bootstrap=10000, seed=42)
            ds_kappa["interpretation"] = interpret_kappa(ds_kappa["kappa"])
            ds_kappa["n_excluded"] = len(ex)
        else:
            ds_kappa = {"kappa": None}
        per_dataset[ds]["fleiss_kappa"] = ds_kappa

    # Coder vs general agreement
    coder_judges = [j["short"] for j in judges if j["type"] == "coder"]
    general_judges = [j["short"] for j in judges if j["type"] == "general"]

    coder_agree = 0
    general_agree = 0
    coder_general_disagree = 0

    for r in results:
        clf = r["classifications"]
        coder_labels = [clf.get(j, "") for j in coder_judges if clf.get(j, "") in ("CODE", "KNOWLEDGE")]
        general_labels = [clf.get(j, "") for j in general_judges if clf.get(j, "") in ("CODE", "KNOWLEDGE")]

        if coder_labels and len(set(coder_labels)) == 1:
            coder_agree += 1
        if general_labels and len(set(general_labels)) == 1:
            general_agree += 1

        if coder_labels and general_labels:
            coder_majority = Counter(coder_labels).most_common(1)[0][0]
            general_majority = Counter(general_labels).most_common(1)[0][0]
            if coder_majority != general_majority:
                coder_general_disagree += 1

    summary = {
        "overall": {
            "total_prompts": n_total,
            "agreement_distribution": dict(agreement_counts),
            "consensus_code": consensus_counts.get("CODE", 0),
            "consensus_knowledge": consensus_counts.get("KNOWLEDGE", 0),
            "consensus_error": consensus_counts.get("ERROR", 0),
            "fleiss_kappa": kappa_result,
        },
        "per_dataset": per_dataset,
        "per_model": per_model,
        "coder_vs_general": {
            "coder_judges": coder_judges,
            "general_judges": general_judges,
            "coders_internal_agree": coder_agree,
            "generals_internal_agree": general_agree,
            "coder_general_disagree": coder_general_disagree,
            "coders_internal_agree_pct": round(coder_agree / n_total * 100, 1) if n_total else 0,
            "generals_internal_agree_pct": round(general_agree / n_total * 100, 1) if n_total else 0,
            "coder_general_disagree_pct": round(coder_general_disagree / n_total * 100, 1) if n_total else 0,
        },
        "methods_documentation": {
            "classification_prompt_template": CLASSIFICATION_PROMPT,
            "judges": [
                {
                    "model_id": j["id"],
                    "short_name": j["short"],
                    "type": j["type"],
                    "provider": j["provider"],
                    "backend": j["backend"],
                }
                for j in judges
            ],
            "agreement_threshold": "3/5 majority vote",
            "statistical_test": "Fleiss' kappa with 10,000-iteration bootstrap 95% CI",
            "interpretation_scale": "Landis & Koch (1977)",
            "rate_limit_seconds": RATE_LIMIT_SECONDS,
            "retry_policy": "3 retries with exponential backoff (1s, 3s, 9s)",
        },
    }

    return summary


def save_outputs(results, summary, prompts, logger):
    """Save all output files."""
    timestamp = datetime.now(timezone.utc).isoformat()

    # Full results
    full_output = {
        "metadata": {
            "run_timestamp": timestamp,
            "num_prompts": len(results),
            "num_judges": len(JUDGES),
        },
        "results": results,
    }
    results_path = DATA_DIR / "classification_results.json"
    with open(results_path, "w") as f:
        json.dump(full_output, f, indent=2)
    logger.info(f"Results saved: {results_path}")

    # Summary
    summary["metadata"] = {"run_timestamp": timestamp}
    summary_path = DATA_DIR / "classification_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    # Clean prompt bank (3/5+ CODE consensus)
    code_bank = []
    for r in results:
        if r["consensus"] == "CODE" and r["n_code"] >= 3:
            code_bank.append({
                "uid": r["uid"],
                "prompt": r["prompt"],
                "source_dataset": r["source_dataset"],
                "category": r["category"],
                "agreement": r["agreement"],
                "n_code_votes": r["n_code"],
            })
    bank_path = DATA_DIR / "prompt_bank_code_consensus.json"
    with open(bank_path, "w") as f:
        json.dump(code_bank, f, indent=2)
    logger.info(f"Code prompt bank saved: {bank_path} ({len(code_bank)} prompts)")

    return results_path, summary_path, bank_path


def print_summary(summary, logger):
    """Print human-readable summary to console."""
    o = summary["overall"]
    k = o["fleiss_kappa"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("CLASSIFICATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total prompts: {o['total_prompts']}")
    logger.info(f"Consensus CODE: {o['consensus_code']}")
    logger.info(f"Consensus KNOWLEDGE: {o['consensus_knowledge']}")
    logger.info(f"Agreement distribution: {o['agreement_distribution']}")
    logger.info("")

    if k.get("kappa") is not None:
        logger.info(
            f"Fleiss' kappa: {k['kappa']:.4f} "
            f"[95% CI: {k.get('ci_lower', 0):.4f}, {k.get('ci_upper', 0):.4f}] "
            f"({k.get('interpretation', 'N/A')})"
        )
        logger.info(f"  Included: {k.get('n_included', 0)}, Excluded: {k.get('n_excluded', 0)}")

    logger.info("")
    logger.info("Per-model breakdown:")
    for model, stats in summary["per_model"].items():
        logger.info(
            f"  {model:25s} [{stats['type']:7s}] "
            f"CODE={stats['code']:5d}  KNOWLEDGE={stats['knowledge']:5d}  "
            f"ERR={stats['error_or_unclear']:3d}"
        )

    logger.info("")
    logger.info("Per-dataset breakdown:")
    for ds, stats in summary["per_dataset"].items():
        ds_k = stats.get("fleiss_kappa", {})
        kappa_str = f"kappa={ds_k.get('kappa', 0):.3f}" if ds_k.get("kappa") is not None else "kappa=N/A"
        logger.info(
            f"  {ds:25s} n={stats['n']:5d}  "
            f"CODE={stats['consensus_code']:5d}  "
            f"KNOWLEDGE={stats['consensus_knowledge']:5d}  "
            f"{kappa_str}"
        )

    cg = summary["coder_vs_general"]
    logger.info("")
    logger.info(f"Coder vs General agreement:")
    logger.info(f"  Coders internal agree: {cg['coders_internal_agree_pct']}%")
    logger.info(f"  Generals internal agree: {cg['generals_internal_agree_pct']}%")
    logger.info(f"  Coder-General disagree: {cg['coder_general_disagree_pct']}%")


def main():
    parser = argparse.ArgumentParser(description="Multi-LLM prompt classification pipeline")
    parser.add_argument("--smoke", type=int, default=0, help="Smoke test with N prompts")
    parser.add_argument("--rate-limit", type=float, default=RATE_LIMIT_SECONDS)
    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"Classification pipeline started at {datetime.now(timezone.utc).isoformat()}")

    # Preflight: check Ollama
    ollama_models = preflight_check()
    if ollama_models is None:
        logger.error("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)
    logger.info(f"Ollama OK: {len(ollama_models)} models available")

    # Check required Ollama models
    ollama_judges = [j["id"] for j in JUDGES if j["backend"] == "ollama"]
    for model in ollama_judges:
        if model not in ollama_models:
            logger.warning(f"Model {model} not found in Ollama. Run: ollama pull {model}")

    # Load prompts
    prompts, counts = load_all_prompts()
    logger.info(f"Loaded prompts: {counts}")

    if args.smoke > 0:
        # Take evenly spaced prompts for diversity
        step = max(1, len(prompts) // args.smoke)
        prompts = prompts[::step][:args.smoke]
        logger.info(f"Smoke test: using {len(prompts)} prompts")

    # Get API key
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("Set OPENROUTER_API_KEY environment variable")
        sys.exit(1)

    # Run classification
    new_calls = run_classification(prompts, JUDGES, api_key, logger, args.rate_limit)

    # Assemble results
    logger.info("Assembling results from checkpoint...")
    results = assemble_results(prompts, JUDGES)

    # Compute summary
    logger.info("Computing summary statistics and Fleiss' kappa...")
    summary = compute_summary(results, JUDGES)

    # Print and save
    print_summary(summary, logger)
    save_outputs(results, summary, prompts, logger)

    logger.info(f"\nPipeline complete at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
