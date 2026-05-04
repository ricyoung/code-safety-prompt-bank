# Code-Safety Prompt Bank

> A consolidated, consensus-labeled prompt bank for distinguishing **malicious-code-generation requests** from **harmful-knowledge requests** in LLM safety evaluation. This repository hosts **code, consensus labels, agreement-tier metadata, figure-generation scripts, and documentation** — it is **not** the distribution channel for the prompt text itself. Prompt text is hosted as a **gated-access dataset** on the Hugging Face Hub.

**Paper**: *A Validated Prompt Bank for Malicious Code Generation* — Young, R.J. and Moody, G.D. (2026)
**Code + labels + metadata (openly public, MIT)**: this repository.
**Prompt-text dataset (gated access, mixed upstream terms)**: [Hugging Face — richardyoung/code-safety-prompt-bank](https://huggingface.co/datasets/richardyoung/code-safety-prompt-bank). Gating is a dual-use mitigation step; access is granted on request for refusal-rate benchmarking, safety-alignment research, interpretability work, and related defensive applications.

---

## What's in this repository

```
.
├── src/
│   ├── classify_prompts.py      # 5-judge consensus classification orchestrator
│   ├── fleiss_kappa.py          # Fleiss' κ with 10,000-iter bootstrap 95% CI
│   ├── backends.py              # OpenRouter + Ollama dual-backend client
│   ├── response_classifier.py   # Response-classification templates (used by companion papers)
│   └── generate_figures.py      # Regenerates all paper figures from released data
├── docs/
│   └── pipeline.md              # End-to-end reproduction guide
├── LICENSE
├── CITATION.cff
└── README.md
```

## Quickstart — reproduce the consensus labels

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API keys (for the judge panel)
export OPENROUTER_API_KEY="sk-or-..."        # Anthropic, OpenAI
export OLLAMA_HOST="https://ollama.com:443"  # Google, Zhipu AI, Alibaba

# 3. Load the release dataset
python -c "
from datasets import load_dataset
ds = load_dataset('richardyoung/code-safety-prompt-bank', 'code_safety', split='train')
print(ds[0])
"

# 4. Re-run the 5-judge panel on any prompt list
python -m src.classify_prompts \
    --prompts path/to/prompts.jsonl \
    --output path/to/labels.jsonl \
    --majority 3
```

## The judge panel

| # | Judge                         | Provider | Type             | Access        |
|---|-------------------------------|----------|------------------|---------------|
| 1 | `anthropic/claude-sonnet-4.6` | Anthropic | general-purpose  | OpenRouter    |
| 2 | `openai/gpt-5.3-codex`        | OpenAI    | coder-specialized | OpenRouter    |
| 3 | `gemini-3-flash-preview:cloud`| Google    | general-purpose  | Ollama Cloud  |
| 4 | `glm-5:cloud`                 | Zhipu AI  | general-purpose  | Ollama Cloud  |
| 5 | `qwen3-coder-next:cloud`      | Alibaba   | coder-specialized | Ollama Cloud  |

Consensus rule: **3-of-5 majority** on CODE or KNOWLEDGE; prompts below this threshold are dropped as ambiguous.

## Headline numbers

- **3,133 prompts** classified × **5 judges** = **15,665 individual decisions**
- **Fleiss' κ = 0.876** [95% CI: 0.862, 0.888] — *almost perfect* by Landis & Koch (1977)
- **1,254 / 1,554** code-safety prompts received unanimous 5/5 agreement (80.7%)
- **Zero consensus-error prompts**
- Inter-judge pairwise Cohen's κ ranges from 0.71 to 0.94 across the ten judge pairs

## Dataset composition

| Config            | Rows  | Sources                                                        |
|-------------------|-------|----------------------------------------------------------------|
| `code_safety`     | 1,554 | RMCBench 473 · MalwareBench 320 · CySecBench 632 · AdvBench 129 |

The 388-prompt content-safety comparison bank described in the paper is not distributed as a public Hugging Face configuration; it is available from the maintainers on request under the same gated-access review process, or can be reconstructed from AdvBench/harmful\_behaviors using the published consensus labels. Schema, loading code, and upstream licenses are documented on the Hugging Face dataset card.

## Reproducing paper figures

```bash
# Regenerate Figures 1–6 of the companion paper from the released data
python -m src.generate_figures \
    --labels path/to/classification_checkpoint.jsonl \
    --summary path/to/classification_summary.json \
    --out figures/
```

## Citation

```bibtex
@article{young2026promptbank,
  title   = {A Validated Prompt Bank for Malicious Code Generation: Separating Executable Weapons from Security Knowledge in 1,554 Consensus-Labeled Prompts},
  author  = {Young, Richard J. and Moody, Gregory D.},
  note    = {Manuscript in preparation; arXiv identifier to be assigned upon submission},
  year    = {2026}
}
```

## Companion work in the same research program

1. **This paper (dataset)** — the validated prompt bank.
2. **Benchmark paper** — *"Is Code Safety a Separate Direction? A 13-Model Behavioral Benchmark of Malicious-Code Refusal in Coding-Specialized LLMs"* — uses this dataset to measure refusal-rate asymmetries across 13 coder-specialized LLMs.
3. **Mechanism paper** — abliteration-based test of whether code-safety and content-safety refusal directions are geometrically separable in the residual stream.

## License

See [LICENSE](./LICENSE) and [LICENSING_NOTES.md](./LICENSING_NOTES.md). This repository (code, consensus labels, agreement-tier metadata, figure-generation scripts, documentation) is MIT-licensed and openly public. The prompt-text dataset hosted on the Hugging Face Hub is distributed under mixed upstream terms as a **gated-access dataset**; no license claim in this repository extends to the prompt text of the dataset, and users of the prompt text are responsible for honoring each upstream source's terms plus the dataset's gated-access policy.

## Ethics and responsible disclosure

- No new malicious prompts were constructed.
- No functional malicious code is distributed.
- Responsible-disclosure coordination with upstream maintainers is ongoing.
- Users should follow the extreme-risk evaluation framework of Shevlane et al. (2023) and avoid any application of the data toward malware fine-tuning or active attack deployment.

## Contact

- **Richard J. Young** — Department of Information Systems, UNLV · ryoung@unlv.edu
- **Gregory D. Moody** — Department of Information Systems, UNLV · greg.moody@unlv.edu
