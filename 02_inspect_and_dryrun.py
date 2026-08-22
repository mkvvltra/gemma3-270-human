"""
Stage 2 — Inspect what the model sees, then dry-run the training loop
=====================================================================

Before spending money on a GPU, do two cheap, high-value things locally:

  A) INSPECT. Take one real conversation and show *exactly* what the model will be
     trained on:
        • the raw templated text (with Gemma's <start_of_turn> markers),
        • the token IDs (so you can confirm there is exactly ONE BOS token — the classic
          double-BOS bug),
        • which tokens are "labeled" vs masked out (i.e. which tokens the loss is
          computed on). With ASSISTANT_ONLY_LOSS, only the `model` reply turns should be
          labeled; the prompt turns should be masked (shown as -100).

  B) DRY-RUN. Run ~10 optimizer steps of the real SFTTrainer on a tiny slice, on whatever
     device you have (MPS/CPU is fine). If this completes without error, the full run in
     stage 3 will too — you've validated the data → template → tokenizer → trainer path.

This stage trains essentially nothing useful; its only job is to catch bugs for free.

Run:  python 02_inspect_and_dryrun.py
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from config import (
    MODEL_ID, TRAIN_FILE, CHAT_TEMPLATE, ASSISTANT_ONLY_LOSS, SEED,
)

# We reuse the exact device-selection logic from stage 0 to stay consistent.
from importlib import import_module
pick_device = import_module("00_setup_check").pick_device  # module name starts with a digit


def load_tokenizer() -> AutoTokenizer:
    """Load the tokenizer and attach OUR chat template.

    The *base* Gemma checkpoint has no chat template of its own (only `-it` does), so we
    install the one from config.py. This is the single most important line for
    reproducibility: base training AND future LoRA personalities must share this template.
    """
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.chat_template = CHAT_TEMPLATE
    return tok


def inspect_one_example(tokenizer) -> None:
    """Part A: print the templated text, token IDs, and the loss mask for one row."""
    print("-" * 70)
    print("A) INSPECT: what does one training example actually look like?")
    print("-" * 70)

    row = load_dataset("json", data_files=str(TRAIN_FILE), split="train")[0]
    messages = row["messages"]

    # 1. The templated *string*. `tokenize=False` returns text so we can read it.
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    print("\n[1] Templated text the model is trained on:\n")
    print(text)

    # 2. The token IDs. Check the very first id equals bos_token_id EXACTLY once.
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    print(f"\n[2] First 8 token IDs: {ids[:8]}")
    print(f"    tokenizer.bos_token_id = {tokenizer.bos_token_id}")
    print(f"    -> starts with exactly one BOS? {ids[0] == tokenizer.bos_token_id and ids[1] != tokenizer.bos_token_id}")

    # 3. The loss mask. `return_assistant_tokens_mask=True` uses the {% generation %}
    #    markers in our template to mark which tokens are the assistant/model reply.
    #    Those (mask == 1) are the ONLY tokens the loss is computed on when
    #    ASSISTANT_ONLY_LOSS is enabled; everything else is ignored (label -100).
    if ASSISTANT_ONLY_LOSS:
        enc = tokenizer.apply_chat_template(
            messages, return_dict=True, return_assistant_tokens_mask=True,
        )
        mask = enc["assistant_masks"]
        n_labeled = sum(mask)
        print(f"\n[3] Loss mask: {n_labeled}/{len(mask)} tokens are labeled (the `model` replies).")
        print("    The rest (the `user` prompts + structural tokens) are masked out (-100).")
        # Show the words we actually optimize on — should be only the reply text.
        labeled_ids = [tid for tid, m in zip(enc["input_ids"], mask) if m == 1]
        print("    Optimized-on text:", repr(tokenizer.decode(labeled_ids)))


def dry_run_training(tokenizer) -> None:
    """Part B: run ~10 real optimizer steps on a tiny slice to validate the pipeline."""
    print("\n" + "-" * 70)
    print("B) DRY-RUN: ~10 training steps on a tiny slice (device may be MPS/CPU)")
    print("-" * 70)

    device = pick_device()
    # bf16 is a CUDA thing; on MPS/CPU we keep full precision to avoid unsupported kernels.
    use_bf16 = device == "cuda"

    # Take just 64 conversations — enough for a few steps, fast on any hardware.
    train_ds = load_dataset("json", data_files=str(TRAIN_FILE), split="train[:64]")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if use_bf16 else torch.float32,  # `dtype` in transformers 5.x
    )

    # SFTConfig is TRL's training-arguments object. Here everything is scaled to "tiny".
    args = SFTConfig(
        output_dir="outputs/_dryrun",   # throwaway
        max_steps=10,                    # stop after 10 steps regardless of dataset size
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        logging_steps=1,                 # print loss every step so you can watch it move
        report_to="none",               # no W&B/etc. for a dry run
        bf16=use_bf16,
        max_length=512,
        packing=False,                   # keep it simple/observable for the dry run
        assistant_only_loss=ASSISTANT_ONLY_LOSS,
        seed=SEED,
        # `use_mps_device` is auto-detected by accelerate; we don't force a device here.
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,      # TRL applies our chat_template via this tokenizer
    )

    print(f"Training 10 steps on {device}… (watch the loss column)")
    trainer.train()
    print("\n[OK] Dry-run completed without errors. The pipeline is wired correctly.")
    print("     Next (on a CUDA GPU): python 03_train_sft.py")


def main() -> None:
    print("=" * 70)
    print("Stage 2: inspect + dry-run")
    print("=" * 70)
    tokenizer = load_tokenizer()
    inspect_one_example(tokenizer)
    dry_run_training(tokenizer)


if __name__ == "__main__":
    main()
