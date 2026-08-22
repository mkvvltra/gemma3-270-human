"""
config.py — the single source of truth shared by every pipeline stage.
=======================================================================

Every numbered script imports from here so that paths, the model id, and — most
importantly — the *chat template* are defined in exactly one place. If the base model
and the LoRA personalities (Phase 2) are ever trained against different templates, they
will not compose. Locking the format here, once, is how we prevent that.

Nothing in this file does any work; it just declares constants.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Which model are we fine-tuning?
# ---------------------------------------------------------------------------
# We deliberately use the *base* (pretrained-only) checkpoint, NOT `-it`.
# The `-it` ("instruction-tuned") variant already has the assistant persona baked in —
# exactly the "As an AI, I can help you…" voice we are trying to remove. Starting from
# the raw base means there is no assistant persona to unlearn; we are shaping a blank
# slate toward "generic human", which is both easier and cleaner.
MODEL_ID = "google/gemma-3-270m"

# ---------------------------------------------------------------------------
# 2. Where do files live?
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"          # cleaned/formatted training data (JSONL)
OUTPUT_DIR = PROJECT_ROOT / "outputs"     # trained model checkpoints

TRAIN_FILE = DATA_DIR / "human_train.jsonl"
VAL_FILE = DATA_DIR / "human_val.jsonl"

# The final full fine-tuned "human base" lands here (stage 3 → stage 5).
SFT_MODEL_DIR = OUTPUT_DIR / "gemma3-270m-human-sft"

# ---------------------------------------------------------------------------
# 3. Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# 4. The chat template  (THE most important shared constant)
# ---------------------------------------------------------------------------
# A "chat template" is a Jinja string that turns a list of role/content messages into
# the exact text the model is trained and prompted with. Gemma uses special tokens to
# mark turns:
#
#     <start_of_turn>user
#     Hi, how are you?<end_of_turn>
#     <start_of_turn>model
#     Not bad, a bit tired honestly.<end_of_turn>
#
# Two Gemma-specific quirks reflected below:
#   * Roles are literally "user" and "model" (Gemma has no "assistant" role — which
#     conveniently fits our framing: the "model" turn is "a person replying").
#   * The sequence begins with the tokenizer's BOS token.
#
# The `{% generation %} … {% endgeneration %}` markers around the model's reply are
# what let TRL compute the loss on the reply turns ONLY (see `ASSISTANT_ONLY_LOSS`).
# Without those markers, "assistant-only" loss masking cannot know which tokens are the
# reply. We wrap only the model turns, so the model learns to *produce* human replies
# rather than to also model the incoming prompts.
#
# CRUCIAL: the generation span must include the closing `<end_of_turn>` token, not just the
# reply text. That token is how a turn *ends* — if it's outside the loss mask, the model
# gets no gradient teaching it to emit `<end_of_turn>`, so at inference it never learns to
# stop and rambles past its reply. So for model turns we emit `<end_of_turn>` *inside*
# `{% generation %}`; the trailing newline (a formatting separator the model needn't
# generate, since decoding stops at `<end_of_turn>`) stays outside the mask.
#
# The reply branch matches BOTH 'model' (our data's role) and 'assistant'. Our data only
# ever uses 'model', but TRL's stop-token sanity check probes the template with a hardcoded
# 'assistant' turn — if the template didn't treat that as a reply turn, the check would see
# an empty mask and (falsely) warn "the model may not learn to stop". Matching both keeps
# the check accurate without changing anything about how our 'model' data is rendered.
#
# NOTE on double-BOS: because this template emits `{{ bos_token }}`, when TRL tokenizes
# the resulting string it must NOT add another BOS. TRL/`apply_chat_template` handle this
# correctly, but stage 2 prints the raw token IDs so you can verify there is exactly one
# BOS at the start — a classic footgun worth seeing with your own eyes.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\n' }}"
    "{% if message['role'] in ['model', 'assistant'] %}"
    "{% generation %}{{ message['content'] | trim }}{{ '<end_of_turn>' }}{% endgeneration %}"
    "{{ '\n' }}"
    "{% else %}"
    "{{ message['content'] | trim }}{{ '<end_of_turn>\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<start_of_turn>model\n' }}{% endif %}"
)

# The two role names our data uses. Keeping them as named constants avoids typos and
# documents the convention: even-indexed dialogue turns are the "prompt" speaker,
# odd-indexed turns are the "reply" speaker we optimize for.
ROLE_USER = "user"      # the incoming speaker (their turns are context, not optimized)
ROLE_MODEL = "model"    # the replying speaker (their turns are what we train the model to produce)

# If True, compute the loss ONLY on the `model` (reply) turns, using the generation
# markers above. This teaches *response style* rather than also memorizing the prompts.
# If you hit a template/masking error on your TRL version, flip this to False to train on
# the full sequence (still works, just slightly less targeted) and revisit later.
ASSISTANT_ONLY_LOSS = True
