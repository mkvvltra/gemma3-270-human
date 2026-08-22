"""
Stage 0 — Setup & sanity check
==============================

Goal: fail *fast and clearly* before you invest time in data prep and training.

This script answers three yes/no questions:
  1. Are the libraries installed and importable?
  2. Which compute device will training use (CUDA / Apple-Silicon MPS / CPU)?
  3. Can we actually pull the gated Gemma weights with your Hugging Face login,
     and do a single forward pass?

Gemma is a *gated* model: you must accept Google's license on the model page and be
logged in (`huggingface-cli login`) before the download works. The most common first-run
failure is a 401/403 here — this script surfaces that with a friendly message instead of
a stack trace three stages later.

Run:  python 00_setup_check.py
"""

import sys

# We import inside a try/except so a missing package produces a readable instruction
# rather than a raw ImportError.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import trl  # noqa: F401  (imported only to confirm it's installed)
except ImportError as e:
    print(f"[X] Missing dependency: {e.name}")
    print("    Fix: pip install -r requirements.txt")
    sys.exit(1)

from config import MODEL_ID


def pick_device() -> str:
    """Return the best available device string.

    Priority: CUDA (real GPU) > MPS (Apple Silicon) > CPU.
    We centralize this here and reuse the same logic in later stages so the whole
    pipeline agrees on where tensors live.
    """
    if torch.cuda.is_available():
        return "cuda"
    # `torch.backends.mps` exists only on recent PyTorch builds; guard with getattr.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    print("=" * 70)
    print("Stage 0: setup & sanity check")
    print("=" * 70)

    # --- 1. Library versions ------------------------------------------------
    print(f"torch        : {torch.__version__}")
    print(f"trl          : {trl.__version__}")

    # --- 2. Compute device --------------------------------------------------
    device = pick_device()
    print(f"device       : {device}")
    if device == "cpu":
        print("   (!) No GPU/MPS found. Fine for stages 0–2; too slow for real training.")
    elif device == "mps":
        print("   (!) Apple Silicon MPS: OK for the dry-run, but do real training on CUDA.")

    # --- 3. Gated-model access + a single forward pass ----------------------
    # This is the real test. If your license/login isn't set up, it fails right here.
    print(f"\nLoading tokenizer + model '{MODEL_ID}' (first run downloads ~0.5 GB)…")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    except Exception as e:  # noqa: BLE001 - we want to translate ANY failure to guidance
        print("[X] Could not load the model. Most likely causes:")
        print("    • You haven't accepted the license: https://huggingface.co/google/gemma-3-270m")
        print("    • You aren't logged in: run `huggingface-cli login`")
        print(f"    Underlying error: {e}")
        sys.exit(1)

    # Report the parameter count — a good gut-check that this really is the 270M model.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model loaded. Parameter count: {n_params/1e6:.1f}M")

    # One tiny forward pass to prove the weights + tokenizer work end to end.
    inputs = tokenizer("The quick brown fox", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    print(f"[OK] Forward pass produced logits of shape {tuple(logits.shape)}.")

    print("\nAll checks passed. Next: python 01_prepare_data.py")


if __name__ == "__main__":
    main()
