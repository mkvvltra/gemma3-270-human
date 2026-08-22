# gemma3-270m-human

Fine-tuning **`google/gemma-3-270m`** (the 270-million-parameter base model) with
[Hugging Face TRL](https://github.com/huggingface/trl) to make it sound like a
**generic human** rather than an AI assistant.

This is **Phase 1** of a two-layer design:

```
gemma-3-270m (pretrained base)
      │  Phase 1 (THIS REPO): full supervised fine-tune (SFT)
      │             on human↔human dialogue
      ▼
"generic-human" base   ──(freeze & distribute)──►  the foundation
      │  Phase 2 (LATER): train a small LoRA adapter per personality
      ▼
+ persona-A.lora  /  + persona-B.lora  /  …   (swapped at inference time)
```

Why this split? A broad "sound human, not like an assistant" shift wants the full
model's capacity, so we **fully fine-tune** the base. A specific personality is a
narrow delta, so later it becomes a cheap, swappable **LoRA** adapter on top of this
frozen base. Keeping them separate means you train the expensive base once and iterate
on personalities cheaply.

---

## The pipeline (read the scripts in order)

Each script is one stage. They are numbered because they run in sequence — the output of
one is the input of the next. Every file starts with a long header comment explaining
*what* it does, *why*, and *how it fits*. Read them top to bottom; they are meant to be
a tutorial as much as working code.

| Stage | Script | What it does |
|------:|--------|--------------|
| 0 | `00_setup_check.py`      | Verify installs, Hugging Face auth (Gemma is gated), and smoke-test loading the base model. |
| 1 | `01_prepare_data.py`     | Download a human-dialogue dataset, clean it, reshape it into chat turns, write `train`/`val` JSONL. |
| 2 | `02_inspect_and_dryrun.py` | *Look* at exactly what the model will see (templated text + token IDs + loss mask), then run a 10-step training dry-run locally to prove the pipeline works. |
| 3 | `03_train_sft.py`        | The real full fine-tune, meant for a cloud GPU. |
| 4 | `04_evaluate.py`         | Qualitative side-by-side generations + held-out perplexity. |
| 5 | `05_export_base.py`      | Save/merge/push the frozen "human base", plus a commented preview of the Phase-2 LoRA setup. |

`config.py` holds the shared constants (model id, paths, the chat template) that every
stage imports, so there is a single source of truth.

---

## Quick start

```bash
# 1. Install (a virtualenv is strongly recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Accept the Gemma license on the model page, then log in:
#    https://huggingface.co/google/gemma-3-270m
huggingface-cli login

# 3. Walk the pipeline
python 00_setup_check.py
python 01_prepare_data.py
python 02_inspect_and_dryrun.py      # runs fine on a Mac (MPS/CPU)
python 03_train_sft.py               # do this on a CUDA GPU (Colab/rented)
python 04_evaluate.py
python 05_export_base.py
```

## Where to run what

- **This Mac (Apple Silicon / MPS):** great for stages 0–2 (setup, data, dry-run).
  MPS is fine for a tiny 10-step sanity run but is slow and lacks some kernels — do
  **not** do the real training here.
- **Cloud GPU (CUDA — Colab L4/A10, or a rented box):** do stage 3 here. A 270M model
  trains in minutes-to-an-hour, so a free/cheap tier is plenty.

## The most important thing in this project

**Data quality decides the outcome, not the code.** The scripts are ~20% of the work.
Whether the model stops saying "As an AI, I can help you with…" and starts sounding like
a person is entirely down to the dialogue data you feed it. Stage 1 starts with
DailyDialog (clean, human, everyday conversation); improving results means improving the
data, then re-running stages 3–4 in a loop.
