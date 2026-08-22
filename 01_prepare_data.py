"""
Stage 1 — Prepare the human-dialogue data
==========================================

This is the stage that actually determines whether the finished model sounds human.
The code is easy; the *data choices* are the whole game.

What we do here:
  1. Download **DailyDialog** — ~13k clean, everyday, human↔human conversations. It is a
     great starting corpus because it is genuinely two people talking (not instructions
     and answers), it is small enough to iterate on, and it is fairly clean.
  2. Clean each utterance (DailyDialog has tokenization artifacts like "I ’ m").
  3. Reshape each conversation into a list of role/content "messages": we alternate
     `user` (incoming speaker) and `model` (the replying speaker we optimize for).
  4. Split into train/validation and write JSONL — one JSON object per line, each of the
     form:  {"messages": [{"role": "user", "content": "..."}, {"role": "model", ...}]}

Why NOT use Alpaca / Dolly / OpenAssistant here?
  Those are instruction→answer datasets — they teach the *assistant* voice, which is
  exactly what we're removing. We want human↔human dialogue.

Why alternate user/model instead of keeping real speaker names?
  Gemma's format has two roles. Mapping "speaker A → user, speaker B → model" lets the
  model learn: given a human turn, produce a human reply. In stage 3 we mask the loss to
  the `model` turns, so it learns to *generate* like a person, not to model the prompts.

Run:  python 01_prepare_data.py
(This runs comfortably on your Mac; no GPU needed.)
"""

import json
import random
import re

from datasets import load_dataset

from config import DATA_DIR, TRAIN_FILE, VAL_FILE, ROLE_USER, ROLE_MODEL, SEED

# What fraction of conversations to hold out for validation. We validate at the
# *conversation* level (not the turn level) so no conversation leaks across the split.
VAL_FRACTION = 0.05

# Drop absurdly short/long turns. Very short turns ("Yes.") add little; very long ones
# are often monologues or scraping noise. These are deliberately loose starting values —
# tuning them is part of improving your data.
MIN_CHARS_PER_TURN = 2
MAX_CHARS_PER_TURN = 500

# Keep only conversations with at least this many turns (need ≥1 user + ≥1 model pair).
MIN_TURNS = 2


def clean_utterance(text: str) -> str:
    """Undo DailyDialog's tokenization artifacts and normalize whitespace.

    DailyDialog ships pre-tokenized, so punctuation and contractions are spaced out:
        "I ’ m not sure ."  ->  "I'm not sure."
    Feeding that spaced-out text to the model would teach it an unnatural style, so we
    repair it. This function is a great place to add more cleaning as you inspect data.
    """
    # Collapse the space before punctuation:  "sure ."  -> "sure."
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    # Rejoin spaced contractions/possessives:  "I ' m" / "I ’ m" -> "I'm"
    text = re.sub(r"\s+([’'])\s*", r"\1", text)
    # Normalize fancy quotes to plain ASCII for consistency.
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    # Collapse any remaining runs of whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def dialog_to_messages(utterances: list[str]) -> list[dict] | None:
    """Turn a list of raw utterances into our role/content message list.

    Even indices (0, 2, 4, …) become `user`; odd indices become `model`. We return None
    if, after cleaning/filtering, the conversation is too short to be useful.
    """
    messages = []
    for i, raw in enumerate(utterances):
        content = clean_utterance(raw)
        # Skip turns that are empty or out of our length band. NOTE: skipping a middle
        # turn would break the alternation, so if a turn is invalid we simply stop the
        # conversation here and keep the clean prefix.
        if not (MIN_CHARS_PER_TURN <= len(content) <= MAX_CHARS_PER_TURN):
            break
        role = ROLE_USER if i % 2 == 0 else ROLE_MODEL
        messages.append({"role": role, "content": content})

    if len(messages) < MIN_TURNS:
        return None
    # We want the conversation to END on a `model` turn so the final training signal is a
    # human reply. If it ends on a `user` turn, drop that trailing turn.
    if messages[-1]["role"] == ROLE_USER:
        messages = messages[:-1]
    if len(messages) < MIN_TURNS:
        return None
    return messages


def main() -> None:
    print("=" * 70)
    print("Stage 1: prepare human-dialogue data (DailyDialog)")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Download the corpus.
    # ------------------------------------------------------------------
    # We use `OpenRL/daily_dialog`, a Parquet mirror of the original DailyDialog. It has
    # the IDENTICAL schema — a `dialog` field holding a list[str] of utterances per row
    # (plus unused `act`/`emotion` labels).
    #
    # Why not the canonical "daily_dialog"? That one ships as a Python *loading script*,
    # which recent `datasets` versions refuse to run ("Dataset scripts are no longer
    # supported"). Parquet mirrors like this one load with no `trust_remote_code`. If this
    # id ever disappears, other drop-in mirrors with the same `dialog` field include
    # `kmyoo/dailydialog-tiny` (100 rows, handy for quick tests).
    print("Downloading DailyDialog… (cached after first run)")
    raw = load_dataset("OpenRL/daily_dialog", split="train")
    print(f"  Loaded {len(raw)} raw conversations.")

    # ------------------------------------------------------------------
    # 2 + 3. Clean and reshape every conversation.
    # ------------------------------------------------------------------
    conversations = []
    for row in raw:
        messages = dialog_to_messages(row["dialog"])
        if messages is not None:
            conversations.append({"messages": messages})
    print(f"  Kept {len(conversations)} conversations after cleaning/filtering.")

    # ------------------------------------------------------------------
    # 4. Shuffle + split at the conversation level, then write JSONL.
    # ------------------------------------------------------------------
    random.seed(SEED)                 # deterministic split (config.SEED) => reproducible
    random.shuffle(conversations)
    n_val = max(1, int(len(conversations) * VAL_FRACTION))
    val_data = conversations[:n_val]
    train_data = conversations[n_val:]

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                # ensure_ascii=False keeps real apostrophes/accents instead of \u escapes.
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(TRAIN_FILE, train_data)
    write_jsonl(VAL_FILE, val_data)

    print(f"\n[OK] Wrote {len(train_data)} train / {len(val_data)} val conversations.")
    print(f"     train -> {TRAIN_FILE}")
    print(f"     val   -> {VAL_FILE}")

    # Show one example so you can see the exact shape the next stage consumes.
    print("\nExample conversation (first train row):")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2))

    print("\nNext: python 02_inspect_and_dryrun.py")


if __name__ == "__main__":
    main()
