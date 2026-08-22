"""
debug_chat.py — a tiny interactive REPL to *talk to* the model
==============================================================

A debugging/inspection utility, not a pipeline stage (hence no number). Use it any time:
  • right now, against the raw base model, to feel what "un-fine-tuned" sounds like;
  • after stage 3, against your fine-tune, to eyeball whether it got more human.

What it demonstrates:
  Multi-turn chat is just a *growing list of messages*. Each turn we append the user's
  line, re-render the ENTIRE history through the chat template, generate one reply, and
  append that reply back. The model itself is stateless — the "memory" of the
  conversation is simply the message list we keep resending. Watching this loop is the
  clearest way to internalize how chat models actually work.

Model selection:
  --model sft   -> your fine-tune in outputs/ (default IF it exists)
  --model base  -> the original google/gemma-3-270m (default if no fine-tune yet)
  --model <path/or/hf-id>  -> anything else you want to poke at

In-chat commands:
  /reset  clears the conversation history (start fresh)
  /exit   quit  (Ctrl-C / Ctrl-D also work)

Run:  python debug_chat.py            # auto-picks sft if present, else base
      python debug_chat.py --model base
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_ID, SFT_MODEL_DIR, CHAT_TEMPLATE, ROLE_USER, ROLE_MODEL

# Reuse the pipeline's single device-selection logic so this tool agrees with training.
from importlib import import_module
pick_device = import_module("00_setup_check").pick_device


def resolve_model_source(choice: str) -> tuple[str, bool]:
    """Turn the --model choice into (path_or_id, is_finetuned).

    `is_finetuned` tells us whether the tokenizer already carries our chat template
    (saved in stage 3) or whether we must attach it ourselves (raw base model).
    """
    if choice == "base":
        return MODEL_ID, False
    if choice == "sft":
        if not SFT_MODEL_DIR.exists():
            raise SystemExit(
                f"[X] No fine-tune found at {SFT_MODEL_DIR}. Run 03_train_sft.py first, "
                f"or use: python debug_chat.py --model base"
            )
        return str(SFT_MODEL_DIR), True
    # An explicit path or HF id: assume it's a finetune-style checkpoint (has a template);
    # if it doesn't, we still set ours below as a fallback, so this is safe.
    return choice, True


def load(choice: str, device: str):
    source, is_finetuned = resolve_model_source(choice)
    print(f"Loading '{source}' on {device}…")

    tok = AutoTokenizer.from_pretrained(source)
    # The raw base model has no chat template; attach the SAME one training uses so the
    # base model is prompted in the exact format the fine-tune will expect. (For a
    # fine-tune the saved tokenizer already has it, but setting it again is harmless.)
    if not is_finetuned or tok.chat_template is None:
        tok.chat_template = CHAT_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        source,
        # transformers 5.x renamed `torch_dtype` -> `dtype` (the old name warns/deprecates).
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def generate_reply(model, tok, messages: list[dict], device: str) -> str:
    """Render the full history, generate one reply, return it as text."""
    # add_generation_prompt=True appends the "<start_of_turn>model\n" opener so the model
    # knows it's its turn to speak. We render the WHOLE message list every call.
    #
    # NOTE (transformers 5.x): with return_dict=True this returns a BatchEncoding
    # ({input_ids, attention_mask}), which we splat into generate() with **enc. Passing the
    # bare tensor positionally (the old style) now raises `AttributeError: 'shape'`, and
    # supplying attention_mask also silences the "right-padding" generation warning.
    enc = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    ).to(device)
    prompt_len = enc["input_ids"].shape[-1]

    # Stop as soon as the model closes its turn. Gemma marks the end of a reply with the
    # special token <end_of_turn>; we add it to the stop set alongside the normal EOS so a
    # (trained) model ends its reply cleanly instead of rambling to max_new_tokens. The
    # raw base model rarely emits it — which is exactly why base output runs on and on.
    stop_ids = [tok.eos_token_id]
    eot_id = tok.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id != tok.unk_token_id:
        stop_ids.append(eot_id)

    out = model.generate(
        **enc,
        max_new_tokens=80,
        do_sample=True,           # sampling => natural, varied phrasing
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3,   # discourage the little model from looping on a phrase
        pad_token_id=tok.eos_token_id,
        eos_token_id=stop_ids,    # stop on EOS or <end_of_turn>, whichever comes first
    )
    # Slice off the prompt tokens; decode only what the model newly generated.
    reply = tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()
    return reply


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=None,
        help="'sft', 'base', or a path/HF-id. Default: 'sft' if trained, else 'base'.",
    )
    args = parser.parse_args()

    # Smart default: prefer the fine-tune once it exists, otherwise the base model.
    choice = args.model or ("sft" if SFT_MODEL_DIR.exists() else "base")

    device = pick_device()
    model, tok = load(choice, device)

    print("\n" + "=" * 70)
    print("Interactive chat.  Commands:  /reset  (clear history)   /exit  (quit)")
    print("=" * 70)

    # THE conversation state — a plain list. This is the model's entire "memory".
    messages: list[dict] = []

    while True:
        try:
            user = input("\nYou : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue
        if user == "/exit":
            print("Bye.")
            break
        if user == "/reset":
            messages = []
            print("(history cleared)")
            continue

        # 1. Append the user's turn.
        messages.append({"role": ROLE_USER, "content": user})
        # 2. Generate the model's reply from the full history.
        reply = generate_reply(model, tok, messages, device)
        # 3. Append the reply so it becomes context for the NEXT turn.
        messages.append({"role": ROLE_MODEL, "content": reply})

        print(f"Bot : {reply}")


if __name__ == "__main__":
    main()
