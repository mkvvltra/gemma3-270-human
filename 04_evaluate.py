"""
Stage 4 — Evaluate: did it actually become more human?
======================================================

There are two complementary ways to judge this model, and you need BOTH:

  1. QUALITATIVE (the one that really matters here). Perplexity can't tell you whether the
     model "sounds human" — you have to read outputs. We generate replies to a fixed set
     of prompts and print them. The thing to look for: it should respond like a person
     ("Ugh, Mondays. Barely.") rather than like an assistant ("As an AI, I don't have
     feelings, but I'm here to help!"). Using a FIXED prompt set every iteration lets you
     compare runs fairly.

  2. QUANTITATIVE. We compute perplexity on the held-out validation set — roughly "how
     surprised is the model by real human replies it never saw". Lower = it models human
     dialogue better. It's a useful trend line across runs, not an absolute truth.

Run:  python 04_evaluate.py
"""

import math

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import SFT_MODEL_DIR, VAL_FILE, ROLE_USER
from importlib import import_module
pick_device = import_module("00_setup_check").pick_device

# A fixed battery of prompts. Keep this list stable across runs so comparisons are fair.
# These are chosen to tempt the old "assistant" reflexes (feelings, opinions, chit-chat).
EVAL_PROMPTS = [
    "Hey, how's your day going?",
    "What do you think about pineapple on pizza?",
    "I had a rough week honestly.",
    "Got any plans for the weekend?",
    "Do you ever get nervous before a big meeting?",
]


def load_model_and_tokenizer(device: str):
    """Load OUR fine-tuned model from stage 3 (not the base). The tokenizer we saved
    alongside it already carries the shared chat template, so generation uses the exact
    same format it was trained on."""
    tok = AutoTokenizer.from_pretrained(str(SFT_MODEL_DIR))
    model = AutoModelForCausalLM.from_pretrained(
        str(SFT_MODEL_DIR),
        # transformers 5.x renamed `torch_dtype` -> `dtype`.
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def qualitative(model, tok, device: str) -> None:
    print("-" * 70)
    print("1) QUALITATIVE: generations on a fixed prompt set")
    print("-" * 70)

    # The model was trained (via the chat template's loss mask) to end its reply with
    # <end_of_turn>. But the base Gemma generation config stops on <eos> (id 1), a token that
    # never appears between turns — so without overriding this, generate() would ignore the
    # model's <end_of_turn> and ramble on to max_new_tokens. Tell it to stop on <end_of_turn>
    # (keeping <eos> as a fallback) so replies end where the model intends.
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    stop_ids = [eot_id]
    if tok.eos_token_id is not None and tok.eos_token_id != eot_id:
        stop_ids.append(tok.eos_token_id)

    for prompt in EVAL_PROMPTS:
        # Wrap the prompt as a single `user` turn and ask for the model's reply. The
        # template's add_generation_prompt appends the "<start_of_turn>model" opener.
        messages = [{"role": ROLE_USER, "content": prompt}]
        # transformers 5.x: return_dict=True yields {input_ids, attention_mask}; splat with
        # **enc. (Passing the bare tensor positionally raises AttributeError: 'shape'.)
        enc = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(device)
        prompt_len = enc["input_ids"].shape[-1]

        out = model.generate(
            **enc,
            max_new_tokens=60,
            do_sample=True,        # sampling gives natural, varied phrasing…
            temperature=0.8,       # …with a bit of randomness
            top_p=0.9,             # nucleus sampling to avoid weird low-probability tokens
            # NOT repetition_penalty: it divides the logit of every token already in the
            # context, and the prompt ends with the user turn's <end_of_turn>. That penalizes
            # the very token the model must emit to stop, so it would ramble to max_new_tokens.
            # no_repeat_ngram_size curbs verbatim loops without touching the single stop token.
            no_repeat_ngram_size=3,
            eos_token_id=stop_ids,   # stop on <end_of_turn> (the turn terminator), not just <eos>
            pad_token_id=tok.eos_token_id,
        )
        # Decode only the NEW tokens (everything after the prompt) = the model's reply.
        reply = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
        print(f"\n  User : {prompt}")
        print(f"  Model: {reply}")


@torch.no_grad()
def perplexity(model, tok, device: str) -> None:
    print("\n" + "-" * 70)
    print("2) QUANTITATIVE: validation perplexity (lower is better)")
    print("-" * 70)

    val = load_dataset("json", data_files=str(VAL_FILE), split="train")

    total_loss, total_tokens = 0.0, 0
    for row in val:
        # Tokenize the full templated conversation; labels = input_ids means "predict the
        # next token everywhere". (For a purer 'reply-only' perplexity you'd reuse the
        # assistant mask from stage 2; full-sequence perplexity is fine as a trend metric.)
        # transformers 5.x: pull the input_ids tensor out of the returned dict.
        enc = tok.apply_chat_template(row["messages"], return_tensors="pt", return_dict=True).to(device)
        ids = enc["input_ids"]
        out = model(input_ids=ids, labels=ids)
        # HF returns the MEAN loss; multiply back up to sum, then re-normalize globally so
        # long and short conversations are weighted by their token counts, not equally.
        n = ids.shape[-1]
        total_loss += out.loss.item() * n
        total_tokens += n

    ppl = math.exp(total_loss / total_tokens)
    print(f"\n  Validation perplexity: {ppl:.2f}  (over {total_tokens} tokens)")
    print("  Compare this number across training runs — track it as you improve the data.")


def main() -> None:
    print("=" * 70)
    print("Stage 4: evaluate the fine-tuned model")
    print("=" * 70)
    device = pick_device()
    model, tok = load_model_and_tokenizer(device)
    qualitative(model, tok, device)
    perplexity(model, tok, device)
    print("\nHappy with it? Next: python 05_export_base.py")


if __name__ == "__main__":
    main()
