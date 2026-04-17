#!/usr/bin/env bash
# Launch GRPO training with reasoning_gym as a standard NeMo-RL ResponseDataset.
#
# Usage:
#   bash examples/reasoning_gym/launch.sh \
#       [++data.train.category_fractions.algebra=0.25] \
#       [++data.train.category_fractions.arithmetic=0.25] \
#       [++grpo.max_num_steps=100] ...
#
# Any extra args are forwarded to run_grpo.py as Hydra overrides.
#
# Categories follow Table 6 of arXiv:2505.24760 (arc merged into cognition,
# probability excluded): algebra, algorithmic, arithmetic, code, cognition,
# games, geometry, graphs, induction, logic.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

# Redirect caches to scratch (mirrors examples/nemo_gym/launch_rl_training.sh)
SCRATCH="${SCRATCH:-/network/scratch/m/mittalsa}"
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$SCRATCH/.cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/.cache/torch}"
export RAY_TMPDIR="${RAY_TMPDIR:-$SCRATCH/.cache/ray}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"

# Point CUDA_HOME to the nvidia cuda_runtime wheel bundled with PyTorch
# (no system CUDA install on Mila compute nodes)
VENV_SITE="$REPO_DIR/.venv/lib/python3.12/site-packages"
export CUDA_HOME="$VENV_SITE/nvidia/cuda_runtime"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# Pre-build DTensor venv if missing (needs GPU access)
DTENSOR_V2_VENV="$REPO_DIR/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
if [[ ! -f "$DTENSOR_V2_VENV/bin/python" ]]; then
  echo "Pre-building DTensorPolicyWorkerV2 venv..."
  UV_PROJECT_ENVIRONMENT="$DTENSOR_V2_VENV" uv sync --locked --extra automodel --directory "$REPO_DIR"
  echo "DTensorPolicyWorkerV2 venv ready."
fi

# Inject transformer-engine dist-info stub if missing
TE_DIST_INFO="$DTENSOR_V2_VENV/lib/python3.12/site-packages/transformer_engine-2.14.0+71bbefb.dist-info"
if [[ ! -d "$TE_DIST_INFO" ]]; then
  mkdir -p "$TE_DIST_INFO"
  printf 'Metadata-Version: 2.1\nName: transformer-engine\nVersion: 2.14.0+71bbefb\nSummary: stub\nRequires-Python: >=3.8\n' > "$TE_DIST_INFO/METADATA"
  touch "$TE_DIST_INFO/RECORD"
  echo "Injected transformer-engine stub."
fi

echo "Working directory: $REPO_DIR"
echo "Launching stock run_grpo.py with reasoning_gym config..."

uv run python examples/run_grpo.py \
    --config examples/reasoning_gym/grpo_reasoning_gym.yaml \
    "$@"

# sbatch examples/reasoning_gym/sbatch_train.sh \
#     --model Qwen/Qwen3-4B-Instruct-2507 \
#     --algebra 0.15 --algorithmic 0.15 --arithmetic 0.15 \
#     --cognition 0.10 --games 0.10 --geometry 0.05 \
#     --graphs 0.10 --logic 0.10 --code 0.05 --induction 0.05 \
#     --steps 500 --seqlen 8192 --mbs 1 --lbs 4