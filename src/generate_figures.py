"""
Paper 1 visualization suite.

Produces 4 new figures in paper_1_dataset/figures/:
  fig3_kappa_heatmap           — pairwise Cohen's kappa between the 5 judges
  fig4_per_judge_labels        — CODE/KNOWLEDGE/ERROR label distribution per judge
  fig5_agreement_tier_pyramid  — distribution of agreement tiers across 3,133 prompts
  fig6_per_source_kappa        — per-source Fleiss kappa with 95% bootstrap CIs

Each figure is saved as both PDF (for LaTeX) and SVG.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).parent.parent
CHECKPOINT = PROJECT / "data" / "classification_checkpoint.jsonl"
SUMMARY = PROJECT / "data" / "classification_summary.json"
OUT = PROJECT / "paper_1_dataset" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Color palette consistent with existing figures
COL_CODER = "#924f77"
COL_GENERAL = "#5d7a5d"
COL_ERROR = "#d98c3a"
COL_CODE = "#7d1a1a"
COL_KNOW = "#1b5e20"
COL_TEXT = "#1f2d3d"

JUDGES_ORDER = [
    ("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6",    "general",  "Anthropic"),
    ("openai/gpt-5.3-codex",        "GPT-5.3-Codex",        "coder",    "OpenAI"),
    ("gemini-3-flash-preview:cloud","Gemini 3 Flash",       "general",  "Google"),
    ("glm-5:cloud",                 "GLM-5",                "general",  "Zhipu AI"),
    ("qwen3-coder-next:cloud",      "Qwen3-Coder-Next",     "coder",    "Alibaba"),
]


def load_per_judge_labels():
    """Return dict keyed by judge-model-id -> dict uid -> label (CODE/KNOWLEDGE/ERROR)."""
    per = defaultdict(dict)
    with open(CHECKPOINT) as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            label = r.get("label", "").upper().strip()
            if label not in ("CODE", "KNOWLEDGE"):
                label = "ERROR"
            per[r["model"]][r["uid"]] = label
    return per


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa on aligned label lists; skip any pair where either is ERROR."""
    assert len(labels_a) == len(labels_b)
    valid = [(a, b) for a, b in zip(labels_a, labels_b) if a in ("CODE", "KNOWLEDGE") and b in ("CODE", "KNOWLEDGE")]
    if not valid: return float("nan"), 0
    a_arr = [1 if x[0] == "CODE" else 0 for x in valid]
    b_arr = [1 if x[1] == "CODE" else 0 for x in valid]
    n = len(valid)
    po = sum(1 for a, b in zip(a_arr, b_arr) if a == b) / n
    pa_code = sum(a_arr) / n
    pb_code = sum(b_arr) / n
    pe = pa_code * pb_code + (1 - pa_code) * (1 - pb_code)
    if pe >= 1.0: return 1.0, n
    return (po - pe) / (1 - pe), n


def fig3_kappa_heatmap(per):
    n = len(JUDGES_ORDER)
    mat = np.ones((n, n))
    counts = np.zeros((n, n), dtype=int)
    # Common set of uids (intersection)
    all_uids = set.intersection(*(set(per[mid].keys()) for mid, *_ in JUDGES_ORDER))
    all_uids = sorted(all_uids)
    for i, (mi, *_) in enumerate(JUDGES_ORDER):
        for j, (mj, *_) in enumerate(JUDGES_ORDER):
            if i == j:
                mat[i, j] = 1.0
                counts[i, j] = len(all_uids)
                continue
            a = [per[mi][u] for u in all_uids]
            b = [per[mj][u] for u in all_uids]
            k, nn = cohen_kappa(a, b)
            mat[i, j] = k
            counts[i, j] = nn
    # Render heatmap
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, vmin=0.5, vmax=1.0, cmap="YlGnBu", aspect="equal")
    labels = [f"{d}\n({r})" for _, d, r, _ in JUDGES_ORDER]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            txt_color = "white" if val > 0.85 else "#1f2d3d"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=txt_color, fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label="Cohen's $\\kappa$")
    ax.set_title("Pairwise Cohen's $\\kappa$ between judges\n(same 5-vendor panel, 3,133 prompts)",
                 fontsize=11, fontweight="bold", pad=12)
    # Frame coder pair in a rectangle
    coder_idx = [i for i, (_, _, r, _) in enumerate(JUDGES_ORDER) if r == "coder"]
    if len(coder_idx) == 2:
        ci, cj = sorted(coder_idx)
        ax.add_patch(plt.Rectangle((cj - 0.5, ci - 0.5), 1, 1,
                                   fill=False, edgecolor=COL_CODER, lw=2.2))
        ax.add_patch(plt.Rectangle((ci - 0.5, cj - 0.5), 1, 1,
                                   fill=False, edgecolor=COL_CODER, lw=2.2))
    plt.tight_layout()
    fig.savefig(OUT / "fig3_kappa_heatmap.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig3_kappa_heatmap.svg", bbox_inches="tight")
    plt.close()
    # Print for inspection
    print("Pairwise Cohen's kappa:")
    for i, (_, d1, *_) in enumerate(JUDGES_ORDER):
        for j, (_, d2, *_) in enumerate(JUDGES_ORDER):
            if j > i:
                print(f"  {d1:<22} vs {d2:<22} = {mat[i,j]:.3f}  (n={counts[i,j]})")


def fig4_per_judge_labels():
    summary = json.load(open(SUMMARY))
    judges_stat = summary["per_model"]
    n = len(JUDGES_ORDER)
    key_map = {
        "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
        "openai/gpt-5.3-codex": "gpt-5.3-codex",
        "gemini-3-flash-preview:cloud": "gemini-3-flash",
        "glm-5:cloud": "glm-5",
        "qwen3-coder-next:cloud": "qwen3-coder-next",
    }
    code = [judges_stat[key_map[m]]["code"] for m, *_ in JUDGES_ORDER]
    know = [judges_stat[key_map[m]]["knowledge"] for m, *_ in JUDGES_ORDER]
    err  = [judges_stat[key_map[m]]["error_or_unclear"] for m, *_ in JUDGES_ORDER]

    fig, ax = plt.subplots(figsize=(10.5, 5))
    x = np.arange(n)
    w = 0.27
    b1 = ax.bar(x - w, code, w, color=COL_CODE, label="CODE",        edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x,     know, w, color=COL_KNOW, label="KNOWLEDGE",   edgecolor="white", linewidth=0.5)
    b3 = ax.bar(x + w, err,  w, color=COL_ERROR,label="ERROR / unclear", edgecolor="white", linewidth=0.5)
    for bars, vals in [(b1, code), (b2, know), (b3, err)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{v:,}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}\n({v}, {r})" for _, d, r, v in JUDGES_ORDER], fontsize=9)
    ax.set_ylabel("Prompts labeled", fontsize=11)
    ax.set_ylim(0, 2100)
    ax.set_title("Per-judge label distribution across 3,133 prompts",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(OUT / "fig4_per_judge_labels.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig4_per_judge_labels.svg", bbox_inches="tight")
    plt.close()


def fig5_agreement_tier_pyramid():
    summary = json.load(open(SUMMARY))
    dist = summary["overall"]["agreement_distribution"]
    # Order tiers from strongest to weakest
    tier_order = ["5/5", "4/5", "3/5", "4/4", "3/4", "3/3", "2/4", "2/3"]
    counts = [dist.get(t, 0) for t in tier_order]
    total = sum(counts)
    colors = ["#1b5e20", "#2e7d32", "#43a047", "#66bb6a", "#9ccc65",
              "#d4e157", "#f0ad4e", "#e57373"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(tier_order))
    bars = ax.barh(y, counts, color=colors, edgecolor="white", linewidth=0.8)
    for bar, c in zip(bars, counts):
        pct = 100 * c / total
        ax.text(c + 20, bar.get_y() + bar.get_height()/2,
                f"{c:,}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels([
        "5/5 (unanimous, all judges valid)",
        "4/5 (strong majority, one dissent)",
        "3/5 (simple majority, two dissent)",
        "4/4 (unanimous, 1 judge errored)",
        "3/4 (majority, 1 errored + 1 dissent)",
        "3/3 (unanimous, 2 errored)",
        "2/4 (tied or minority, 1 errored)",
        "2/3 (minority, 2 errored)",
    ], fontsize=9.3)
    ax.invert_yaxis()
    ax.set_xlabel("Number of prompts (of 3,133 total)", fontsize=11)
    ax.set_xlim(0, max(counts) * 1.22)
    ax.set_title("Agreement-tier distribution across the 5-judge panel",
                 fontsize=12, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
    # Annotation for unanimous percentage
    pct_unan = 100 * counts[0] / total
    ax.text(counts[0]*0.5, 0, f"{pct_unan:.1f}%\nUNANIMOUS",
            ha="center", va="center", fontsize=12, fontweight="bold", color="white")
    plt.tight_layout()
    fig.savefig(OUT / "fig5_agreement_tier_pyramid.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig5_agreement_tier_pyramid.svg", bbox_inches="tight")
    plt.close()


def fig6_per_source_kappa():
    summary = json.load(open(SUMMARY))
    per = summary["per_dataset"]
    sources = [
        ("rmcbench",        "RMCBench\n(n=473, all CODE)"),
        ("malwarebench",    "MalwareBench\n(n=320, all CODE)"),
        ("harmful_behaviors", "harmful_behaviors\n(n=520, mixed)"),
        ("cysecbench",      "CySecBench\n(n=1,820, mixed)"),
    ]
    kappas, cis, interp = [], [], []
    for k, _ in sources:
        d = per[k]["fleiss_kappa"]
        kappas.append(d["kappa"])
        cis.append((d["ci_lower"], d["ci_upper"]))
        interp.append(d.get("interpretation", "?"))
    # Also include overall
    overall = summary["overall"]["fleiss_kappa"]
    sources.append(("overall", "Overall\n(n=3,133)"))
    kappas.append(overall["kappa"])
    cis.append((overall["ci_lower"], overall["ci_upper"]))
    interp.append(overall.get("interpretation", "?"))
    n = len(sources)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(n)
    for i, (k, (lo, hi)) in enumerate(zip(kappas, cis)):
        is_overall = (i == n - 1)
        color = "#6a4c93" if is_overall else "#3e7cb1"
        # Clamp CI display to [-0.1, 1.05] so the RMCBench degenerate doesn't break the axis
        lo_d = max(lo, -0.1); hi_d = min(hi, 1.05)
        ax.errorbar([k], [i], xerr=[[k - lo_d], [hi_d - k]],
                    fmt="o", markersize=12, color=color, ecolor=color,
                    elinewidth=2.2, capsize=6, capthick=2.2)
        ax.text(max(hi_d, k) + 0.03, i,
                f"$\\kappa = {k:.3f}$ ({interp[i]})",
                va="center", ha="left", fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels([lbl for _, lbl in sources], fontsize=9.5)
    ax.invert_yaxis()
    ax.axvline(0.81, color="#1b5e20", ls=":", lw=1.2, alpha=0.7,
               label="\"almost perfect\" threshold ($\\kappa \\geq 0.81$)")
    ax.axvline(0.61, color="#d98c3a", ls=":", lw=1.2, alpha=0.7,
               label="\"substantial\" threshold ($\\kappa \\geq 0.61$)")
    ax.set_xlabel("Fleiss' $\\kappa$  (bootstrap 95% CI)", fontsize=11)
    ax.set_xlim(-0.15, 1.6)
    ax.set_title("Per-source inter-rater reliability (5 judges, Fleiss' $\\kappa$)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=8.5, frameon=False, bbox_to_anchor=(1.0, 1.12))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
    plt.tight_layout()
    fig.savefig(OUT / "fig6_per_source_kappa.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig6_per_source_kappa.svg", bbox_inches="tight")
    plt.close()


def main():
    per = load_per_judge_labels()
    print(f"Loaded labels from {len(per)} judges. Sample counts:")
    for mid, d in per.items():
        print(f"  {mid}: {len(d)} labels")
    print()
    fig3_kappa_heatmap(per)
    fig4_per_judge_labels()
    fig5_agreement_tier_pyramid()
    fig6_per_source_kappa()
    print(f"\nSaved to: {OUT}")


if __name__ == "__main__":
    main()
