# Reproduction Pipeline

End-to-end reproduction of the consensus-labeled prompt bank described in the
companion paper. The repository ships **code, consensus labels, and metadata**;
the **prompt text** is gated on the Hugging Face Hub.

## 0. Prerequisites

```bash
pip install -r requirements.txt
```

API keys for the judge panel:

```bash
export OPENROUTER_API_KEY="sk-or-..."        # Anthropic, OpenAI
export OLLAMA_HOST="https://ollama.com:443"  # Google, Zhipu AI, Alibaba
```

Hugging Face access (gated dataset):

```bash
huggingface-cli login
# Then request access at:
#   https://huggingface.co/datasets/richardyoung/code-safety-prompt-bank
```

## 1. Load the gated dataset

```python
from datasets import load_dataset
ds = load_dataset(
    "richardyoung/code-safety-prompt-bank",
    "code_safety",
    split="train",
)
```

## 2. Re-run the 5-judge consensus pipeline

```bash
python -m src.classify_prompts \
    --prompts path/to/prompts.jsonl \
    --output path/to/labels.jsonl \
    --majority 3
```

Outputs:
- `labels.jsonl` — one row per prompt with all five judge votes and the
  3-of-5 consensus label.
- `classification_summary.json` — overall and per-source Fleiss' kappa with
  10,000-iteration bootstrap 95 % confidence intervals.

## 3. Regenerate paper figures

```bash
python -m src.generate_figures \
    --labels  path/to/classification_checkpoint.jsonl \
    --summary path/to/classification_summary.json \
    --out     figures/
```

## 4. Validate against the released artifact

The released `classification_summary.json` reports:

- Fleiss' kappa = 0.876 [95 % CI: 0.862, 0.888] computed on the 2,488
  prompts that received a valid label from all five judges.
- 645 prompts excluded from the kappa computation due to at least one
  judge returning an error or unparseable response. These prompts retain
  their 3-of-5 consensus label where one is reachable and are flagged in
  the released JSONL with the per-row `n_valid_judges` field.

A clean re-run on a fresh OpenRouter or Ollama Cloud account should
reproduce the kappa point estimate to within bootstrap noise; any larger
deviation is a signal that an upstream judge has been retrained.

## Judge panel

| # | Model                          | Provider | Type             | Backend       |
|---|--------------------------------|----------|------------------|---------------|
| 1 | `anthropic/claude-sonnet-4.6`  | Anthropic | general          | OpenRouter    |
| 2 | `openai/gpt-5.3-codex`         | OpenAI    | coder            | OpenRouter    |
| 3 | `gemini-3-flash-preview:cloud` | Google    | general          | Ollama Cloud  |
| 4 | `glm-5:cloud`                  | Zhipu AI  | general          | Ollama Cloud  |
| 5 | `qwen3-coder-next:cloud`       | Alibaba   | coder            | Ollama Cloud  |

## Repository layout

```
src/
  classify_prompts.py      — orchestrator
  generate_figures.py      — paper figures
  lib/
    backends.py            — OpenRouter + Ollama clients
    datasets.py            — upstream loader
    fleiss_kappa.py        — kappa with bootstrap CI
    io_utils.py            — JSONL checkpointing
    response_classifier.py — response template (used by companion papers)
docs/
  pipeline.md              — this file
```
