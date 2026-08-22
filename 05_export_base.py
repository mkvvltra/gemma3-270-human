"""
Stage 5 — Export the frozen "human base" (and a preview of Phase 2)
===================================================================

Two jobs:

  1. FINALIZE the base. Because stage 3 was a FULL fine-tune, the weights are already
     merged (there is no adapter to fold in) — the model in `outputs/gemma3-270m-human-sft`
     IS the finished base. This stage just (optionally) pushes it to the Hugging Face Hub
     so it becomes a stable, named foundation you can build personalities on.

  2. PREVIEW Phase 2. The whole point of doing Phase 1 as a full fine-tune was to freeze a
     clean human base and then add personalities as cheap, swappable **LoRA adapters**.
     The bottom of this file is a fully-commented (but not-executed) sketch of exactly how
     that Phase-2 training would look, so the path forward is concrete.

Run:  python 05_export_base.py                 # local finalize only
      python 05_export_base.py --push <repo>   # also push to the Hub
"""

import argparse

from config import SFT_MODEL_DIR


def finalize(push_to: str | None) -> None:
    print("=" * 70)
    print("Stage 5: export the frozen human base")
    print("=" * 70)

    if not SFT_MODEL_DIR.exists():
        print(f"[X] {SFT_MODEL_DIR} not found. Run stage 3 (03_train_sft.py) first.")
        return

    print(f"[OK] Frozen base is ready at: {SFT_MODEL_DIR}")
    print("     (Full fine-tune => weights already merged; nothing to fold in.)")

    if push_to:
        # Lazy import so the Hub dependency is only needed if you actually push.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"\nPushing to the Hugging Face Hub as '{push_to}'…")
        model = AutoModelForCausalLM.from_pretrained(str(SFT_MODEL_DIR))
        tok = AutoTokenizer.from_pretrained(str(SFT_MODEL_DIR))
        # private=True keeps it unlisted until you're ready to share. The tokenizer push
        # carries our chat template along, so downstream users get the right format.
        model.push_to_hub(push_to, private=True)
        tok.push_to_hub(push_to, private=True)
        print(f"[OK] Pushed. This is now the foundation for your Phase-2 LoRA personalities.")
    else:
        print("\n(No --push given; kept local. Add --push <username>/<repo> to publish.)")


# ---------------------------------------------------------------------------
# PHASE 2 PREVIEW — how a personality LoRA is trained on top of this base.
# ---------------------------------------------------------------------------
# This function is intentionally NOT called. It's a readable blueprint for the next repo,
# showing why the two-layer split pays off: same data pipeline, same chat template, but
# now the base is FROZEN and only a tiny adapter (~1–2% of params) learns the personality.
#
# def train_personality_lora(persona_data_path, adapter_out_dir):
#     import torch
#     from datasets import load_dataset
#     from peft import LoraConfig                       # the adapter definition
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     from trl import SFTConfig, SFTTrainer
#
#     # 1. Start from OUR frozen human base — not the original Gemma. The personality is a
#     #    delta on top of "generic human".
#     base = AutoModelForCausalLM.from_pretrained(str(SFT_MODEL_DIR), torch_dtype=torch.bfloat16)
#     tok = AutoTokenizer.from_pretrained(str(SFT_MODEL_DIR))   # inherits the shared template
#
#     # 2. Describe the LoRA adapter. Only these small matrices train; the base stays frozen.
#     peft_config = LoraConfig(
#         r=16,                    # adapter rank — capacity of the personality delta
#         lora_alpha=32,           # scaling; a common rule of thumb is alpha = 2*r
#         lora_dropout=0.05,
#         target_modules="all-linear",  # attach to all linear layers (attn + MLP)
#         task_type="CAUSAL_LM",
#     )
#
#     # 3. Train exactly like stage 3, but pass peft_config so SFTTrainer trains ONLY the
#     #    adapter. LoRA tolerates a higher LR because it updates far fewer parameters.
#     args = SFTConfig(output_dir=adapter_out_dir, num_train_epochs=3,
#                      per_device_train_batch_size=16, learning_rate=2e-4,
#                      assistant_only_loss=True, packing=True, bf16=True)
#     trainer = SFTTrainer(model=base, args=args, peft_config=peft_config,
#                          train_dataset=load_dataset("json", data_files=persona_data_path)["train"],
#                          processing_class=tok)
#     trainer.train()
#     trainer.save_model(adapter_out_dir)   # saves ONLY the small adapter weights
#
#     # At inference you load the frozen base once and hot-swap adapters:
#     #     model.load_adapter("persona-A"); model.set_adapter("persona-A")
#     #     model.load_adapter("persona-B"); model.set_adapter("persona-B")
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push", metavar="REPO", default=None,
        help="Hugging Face repo id (e.g. 'yourname/gemma3-270m-human') to push to.",
    )
    args = parser.parse_args()
    finalize(args.push)


if __name__ == "__main__":
    main()
