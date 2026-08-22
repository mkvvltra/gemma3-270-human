#!/usr/bin/env bash
# ===========================================================================
# setup_gpu.sh — one-shot environment setup for a fresh CUDA GPU box
# (written for Scaleway + Ubuntu Noble / 24.04, but works on any recent Ubuntu)
#
# What it does, in order:
#   1. Verifies an NVIDIA GPU is visible (nvidia-smi).
#   2. Installs the few system packages Noble needs (python venv/pip, git).
#   3. Creates a .venv and installs the Python dependencies (CUDA torch build).
#   4. Reminds you to log in to Hugging Face (Gemma is a gated model).
#
# What it deliberately does NOT do:
#   - It does not run the pipeline (stages 00–05) — you do that yourself so you
#     can watch each step. It only prepares the machine.
#   - It does not `huggingface-cli login` for you (that needs your token pasted
#     interactively); it just tells you to.
#
# Usage:
#   chmod +x setup_gpu.sh
#   ./setup_gpu.sh
#   # then:  source .venv/bin/activate  &&  huggingface-cli login
#
# `set -euo pipefail` makes the script stop on the first error instead of
# limping onward — you want setup to fail loudly, not half-succeed.
# ===========================================================================
set -euo pipefail

echo "=== 1/4  Checking for an NVIDIA GPU ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[X] nvidia-smi not found. This box has no NVIDIA driver."
    echo "    On a plain Ubuntu image: sudo ubuntu-drivers install && sudo reboot"
    echo "    (Scaleway GPU OS images usually ship drivers preinstalled.)"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo

echo "=== 2/4  Installing system packages (needs sudo) ==="
# Noble enforces PEP 668, so a virtualenv is mandatory — hence python3-venv.
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip git
echo

echo "=== 3/4  Creating .venv and installing Python dependencies ==="
# Only (re)create the venv if it doesn't already exist, so re-running is cheap.
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
# On Linux the default torch wheel is the CUDA build — no extra index-url needed.
pip install -r requirements.txt
echo

echo "=== 4/4  Done. Two manual steps remain ==="
cat <<'EOF'

  This machine is ready, but you still need to:

    1. Activate the environment in your shell:
         source .venv/bin/activate

    2. Log in to Hugging Face (Gemma is gated — accept its license first at
       https://huggingface.co/google/gemma-3-270m):
         huggingface-cli login

  Then run the pipeline:
         python 00_setup_check.py      # confirms CUDA + gated download
         python 01_prepare_data.py     # builds data/ on this box
         python 02_inspect_and_dryrun.py
         python 03_train_sft.py        # the real training run

EOF
