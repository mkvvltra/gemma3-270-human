"""
Stage 3 — The real supervised fine-tune (SFT)
==============================================

This is the actual training run that produces your "generic-human" base. Run it on a
CUDA GPU (Colab L4/A10 or a rented box). At 270M params this is minutes-to-an-hour.

What "SFT" means here: we show the model many (context → human reply) examples and nudge
its weights so the replies it produces look like the human replies in the data. Because
ASSISTANT_ONLY_LOSS masks the loss to the `model` turns, the gradient signal is "make the
reply more like this human said it" — not "also memorize the prompt".

We do a FULL fine-tune (all weights update), not LoRA, because turning an assistant-shaped
model into a human-shaped one is a broad change that wants the whole model's capacity.
LoRA is reserved for Phase 2 (personalities), where a small delta is exactly right.

Key knobs and why they're set the way they are are documented inline. The two things you
will most often change to improve results: the *data* (stage 1) and `num_train_epochs`
(watch validation loss — a 270M model overfits a small corpus quickly).

Run (on a GPU):  python 03_train_sft.py
"""

import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from config import (
    MODEL_ID, TRAIN_FILE, VAL_FILE, SFT_MODEL_DIR, CHAT_TEMPLATE,
    ASSISTANT_ONLY_LOSS, SEED,
)
from trl import SFTConfig, SFTTrainer

# Reuse stage 2's tokenizer loader (which attaches our shared chat template) and stage 0's
# device picker, so the whole pipeline stays consistent.
from importlib import import_module
load_tokenizer = import_module("02_inspect_and_dryrun").load_tokenizer
pick_device = import_module("00_setup_check").pick_device


def require_data() -> None:
    """Fail fast with a human-readable message if stage 1's output is missing.

    Stage 3 is compute-self-contained (it downloads the model itself) but it still needs
    the JSONL that stage 1 produces. On a fresh GPU box it's easy to jump straight here and
    hit a confusing traceback deep inside `datasets`; this turns that into one clear line.
    """
    missing = [str(p) for p in (TRAIN_FILE, VAL_FILE) if not p.exists()]
    if missing:
        raise SystemExit(
            "[X] Training data not found:\n"
            + "\n".join(f"    - {m}" for m in missing)
            + "\n    Fix: run `python 01_prepare_data.py` first (or copy the data/ folder "
              "from another machine)."
        )


def main() -> None:
    print("=" * 70)
    print("Stage 3: full supervised fine-tune")
    print("=" * 70)

    require_data()          # stop early + clearly if stage 1 hasn't produced the data
    device = pick_device()
    if device != "cuda":
        # Not a hard stop — but be loud, because a full run on CPU/MPS is painfully slow.
        print(f"[!] Device is '{device}', not CUDA. This will be SLOW. Consider a cloud GPU.")

    # ------------------------------------------------------------------
    # 1. Tokenizer (with our shared chat template) + datasets.
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer()

    # TRL detects the "messages" column and applies the chat template automatically.
    dataset = load_dataset(
        "json",
        data_files={"train": str(TRAIN_FILE), "validation": str(VAL_FILE)},
    )
    print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}")

    # ------------------------------------------------------------------
    # 2. Model. bf16 on CUDA is the sweet spot: half the memory, negligible quality loss.
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,  # `dtype` in transformers 5.x
        # SDPA is PyTorch's built-in fused attention: fast, and it masks each example correctly
        # on its own. We deliberately avoid `packing=True` (see below) because that path uses
        # padding-free flattening, which only Flash Attention implements correctly — and flash-attn
        # has no prebuilt wheel for this Python/torch, so SDPA + unpacked is the correct combo here.
        attn_implementation="sdpa",
    )

    # ------------------------------------------------------------------
    # 3. Training configuration. Every field is annotated.
    # ------------------------------------------------------------------
    # A few knobs are referenced both here and in the warmup calc below, so pull them out
    # as named values to keep the two in sync.
    num_train_epochs = 3               # 2–3 is plenty; more on a small corpus => memorization
    per_device_train_batch_size = 16   # lower this if you hit OOM…
    gradient_accumulation_steps = 2    # …and raise this to keep the effective batch (32) stable
    max_length = 1024                  # dialogues are short; 1024 tokens is plenty

    # transformers 5.x removed `warmup_ratio`, so express the same "warm up over the first ~3%
    # of training" intent as an absolute `warmup_steps`. With `packing=False` (below) each
    # example is its own sequence, so the optimizer-step count is simply examples / effective
    # batch — no packing factor to account for.
    effective_batch = per_device_train_batch_size * gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(dataset["train"]) / effective_batch)
    warmup_steps = max(1, round(0.03 * steps_per_epoch * num_train_epochs))

    args = SFTConfig(
        output_dir=str(SFT_MODEL_DIR),

        # --- How long to train ---
        # Start with 2–3 epochs. More epochs on a small dataset => memorization, not
        # generalization. Trust the validation loss (below) over the training loss.
        num_train_epochs=num_train_epochs,

        # --- Batch size / effective batch ---
        # Effective batch = per_device_batch * grad_accum * num_gpus.
        # 16 * 2 = 32 sequences/step. Lower per_device_train_batch_size if you hit OOM;
        # raise grad_accum to compensate so the effective batch stays stable.
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,

        # --- Optimizer schedule ---
        learning_rate=5e-5,          # modest for a FULL fine-tune (LoRA would use ~1e-4+)
        lr_scheduler_type="cosine",  # decay to ~0 for a smooth landing
        warmup_steps=warmup_steps,   # ~3% ramp (transformers 5.x dropped warmup_ratio; see above)
        weight_decay=0.01,           # mild regularization
        max_grad_norm=1.0,           # gradient clipping — guards against loss spikes

        # --- Precision ---
        bf16=(device == "cuda"),

        # --- Sequence handling ---
        max_length=max_length,       # dialogues are short; 1024 tokens is plenty (set above)
        packing=False,               # keep each dialogue a separate padded sequence so SDPA masks
                                     # them correctly. packing=True's bfd strategy uses padding-free
                                     # flattening, which needs flash-attn (unavailable here) to stop
                                     # examples from attending across each other.
        assistant_only_loss=ASSISTANT_ONLY_LOSS,  # loss only on the human-reply turns

        # --- Validation & checkpoints ---
        eval_strategy="steps",
        eval_steps=100,              # measure val loss every 100 steps to catch overfitting
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,          # keep disk usage bounded
        load_best_model_at_end=True, # restore the checkpoint with the lowest val loss
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # --- Logging / reproducibility ---
        logging_steps=20,
        report_to="none",            # set to "wandb" and `wandb login` to get live charts
        seed=SEED,
    )

    # ------------------------------------------------------------------
    # 4. Train.
    # ------------------------------------------------------------------
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("\nStarting training…\n")
    trainer.train()

    # ------------------------------------------------------------------
    # 5. Save the final (best) model + tokenizer together.
    # ------------------------------------------------------------------
    # Saving the tokenizer alongside the weights bakes in our chat template, so anything
    # that loads this folder later (stage 4, stage 5, the Phase-2 LoRA training) inherits
    # the exact same format for free.
    trainer.save_model(str(SFT_MODEL_DIR))
    tokenizer.save_pretrained(str(SFT_MODEL_DIR))
    print(f"\n[OK] Saved fine-tuned model to {SFT_MODEL_DIR}")
    print("     Next: python 04_evaluate.py")


if __name__ == "__main__":
    main()
