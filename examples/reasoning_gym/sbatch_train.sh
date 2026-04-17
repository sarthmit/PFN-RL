#!/usr/bin/env bash
#SBATCH --job-name=grpo-reasoning-gym
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=2:59:00
#SBATCH --output=/home/mila/m/mittalsa/scratch/RL/logs/reasoning-gym/slurm-%j.out
#SBATCH --error=/home/mila/m/mittalsa/scratch/RL/logs/reasoning-gym/slurm-%j.err
#
# Sbatch-launchable GRPO training on reasoning_gym.
#
# Usage:
#   sbatch examples/reasoning_gym/sbatch_train.sh \
#     --model Qwen/Qwen2.5-1.5B-Instruct \
#     --algebra 0.15 --algorithmic 0.15 --arithmetic 0.15 \
#     --cognition 0.10 --games 0.10 --geometry 0.05 \
#     --graphs 0.10 --logic 0.10 --code 0.05 --induction 0.05 \
#     --steps 500 --size 10000 \
#     --mbs 2 --lbs 2 --seqlen 4096
#
# Training knobs:
#   --mbs INT           policy.train_micro_batch_size
#   --lbs INT           policy.logprob_batch_size
#   --seqlen INT        policy.max_total_sequence_length (alias: --max-seq-len)
#   --num-prompts INT   grpo.num_prompts_per_step
#   --num-generations INT grpo.num_generations_per_prompt
#   --dtensor-tp INT    policy.dtensor_cfg.tensor_parallel_size
#   --vllm-tp INT       policy.generation.vllm_cfg.tensor_parallel_size
#
# Any flag without a matching case below is forwarded verbatim as a Hydra
# override to run_grpo.py (e.g. ++policy.dynamic_batching.enabled=true).
#
# Categories (Table 6 of arXiv:2505.24760; arc→cognition, probability excluded):
#   --algebra --algorithmic --arithmetic --code --cognition
#   --games --geometry --graphs --induction --logic
#
# SLURM overrides — request different resources by passing --export=NONE with
# sbatch flags, or by editing the #SBATCH lines above for a one-shot change.

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

REPO_DIR="/home/mila/m/mittalsa/scratch/RL"
cd "$REPO_DIR"

mkdir -p "$REPO_DIR/logs/reasoning-gym"

# ---------------------------------------------------------------------------
# Cache redirection (mirrors examples/nemo_gym/launch_rl_training.sh)
# ---------------------------------------------------------------------------
SCRATCH="${SCRATCH:-/network/scratch/m/mittalsa}"
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$SCRATCH/.cache/uv}"
export VLLM_CACHE_DIR="${VLLM_CACHE_DIR:-$SCRATCH/.cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/.cache/torch}"
export RAY_TMPDIR="${RAY_TMPDIR:-$SCRATCH/.cache/ray}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"

# Point CUDA_HOME to the PyTorch-bundled cuda_runtime (no system CUDA on Mila)
VENV_SITE="$REPO_DIR/.venv/lib/python3.12/site-packages"
export CUDA_HOME="$VENV_SITE/nvidia/cuda_runtime"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Parse named args; anything unrecognized is passed through as a Hydra override
# ---------------------------------------------------------------------------
MODEL=""
STEPS=""
SIZE=""
VAL_SIZE=""
NUM_PROMPTS=""
NUM_GENERATIONS=""
MAX_SEQ_LEN=""
MBS=""
LBS=""
DTENSOR_TP=""
VLLM_TP=""

declare -A FRAC
CATEGORIES=(algebra algorithmic arithmetic code cognition games geometry graphs induction logic)
for cat in "${CATEGORIES[@]}"; do FRAC[$cat]=""; done

EXTRA_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  matched=false
  case "$1" in
    --model)            MODEL="$2";            shift 2; matched=true ;;
    --steps)            STEPS="$2";            shift 2; matched=true ;;
    --size)             SIZE="$2";             shift 2; matched=true ;;
    --val-size)         VAL_SIZE="$2";         shift 2; matched=true ;;
    --num-prompts)      NUM_PROMPTS="$2";      shift 2; matched=true ;;
    --num-generations)  NUM_GENERATIONS="$2";  shift 2; matched=true ;;
    --seqlen|--max-seq-len) MAX_SEQ_LEN="$2";  shift 2; matched=true ;;
    --mbs)              MBS="$2";              shift 2; matched=true ;;
    --lbs)              LBS="$2";              shift 2; matched=true ;;
    --dtensor-tp)       DTENSOR_TP="$2";       shift 2; matched=true ;;
    --vllm-tp)          VLLM_TP="$2";          shift 2; matched=true ;;
  esac
  if ! $matched; then
    for cat in "${CATEGORIES[@]}"; do
      if [[ "$1" == "--$cat" ]]; then
        FRAC[$cat]="$2"; shift 2; matched=true; break
      fi
    done
  fi
  if ! $matched; then
    EXTRA_OVERRIDES+=("$1")
    shift
  fi
done

# ---------------------------------------------------------------------------
# Build Hydra override list
# ---------------------------------------------------------------------------
OVERRIDES=()
[[ -n "$MODEL" ]]           && OVERRIDES+=("++policy.model_name=$MODEL")
[[ -n "$STEPS" ]]           && OVERRIDES+=("++grpo.max_num_steps=$STEPS")
[[ -n "$SIZE" ]]            && OVERRIDES+=("++data.train.size=$SIZE")
[[ -n "$VAL_SIZE" ]]        && OVERRIDES+=("++data.validation.size=$VAL_SIZE")
[[ -n "$NUM_PROMPTS" ]]     && OVERRIDES+=("++grpo.num_prompts_per_step=$NUM_PROMPTS")
[[ -n "$NUM_GENERATIONS" ]] && OVERRIDES+=("++grpo.num_generations_per_prompt=$NUM_GENERATIONS")
[[ -n "$MAX_SEQ_LEN" ]]     && OVERRIDES+=("++policy.max_total_sequence_length=$MAX_SEQ_LEN")
[[ -n "$MBS" ]]             && OVERRIDES+=("++policy.train_micro_batch_size=$MBS")
[[ -n "$LBS" ]]             && OVERRIDES+=("++policy.logprob_batch_size=$LBS")
[[ -n "$DTENSOR_TP" ]]      && OVERRIDES+=("++policy.dtensor_cfg.tensor_parallel_size=$DTENSOR_TP")
[[ -n "$VLLM_TP" ]]         && OVERRIDES+=("++policy.generation.vllm_cfg.tensor_parallel_size=$VLLM_TP")

for cat in "${CATEGORIES[@]}"; do
  if [[ -n "${FRAC[$cat]}" ]]; then
    OVERRIDES+=("++data.train.category_fractions.$cat=${FRAC[$cat]}")
  fi
done
OVERRIDES+=("${EXTRA_OVERRIDES[@]}")

# ---------------------------------------------------------------------------
# Pre-build DTensor worker venv if absent. Needs GPU in scope, so we do it
# here (inside the sbatch allocation) before Ray starts.
# ---------------------------------------------------------------------------
DTENSOR_V2_VENV="$REPO_DIR/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
if [[ ! -f "$DTENSOR_V2_VENV/bin/python" ]]; then
  echo "Pre-building DTensorPolicyWorkerV2 venv..."
  UV_PROJECT_ENVIRONMENT="$DTENSOR_V2_VENV" uv sync --locked --extra automodel --directory "$REPO_DIR"
  echo "DTensorPolicyWorkerV2 venv ready."
fi

# ---------------------------------------------------------------------------
# Inject transformer-engine dist-info stub (Megatron-LM checks TE version at
# import; TE isn't buildable on Mila bare-metal)
# ---------------------------------------------------------------------------
TE_DIST_INFO="$DTENSOR_V2_VENV/lib/python3.12/site-packages/transformer_engine-2.14.0+71bbefb.dist-info"
if [[ ! -d "$TE_DIST_INFO" ]]; then
  mkdir -p "$TE_DIST_INFO"
  printf 'Metadata-Version: 2.1\nName: transformer-engine\nVersion: 2.14.0+71bbefb\nSummary: stub\nRequires-Python: >=3.8\n' > "$TE_DIST_INFO/METADATA"
  touch "$TE_DIST_INFO/RECORD"
  echo "Injected transformer-engine stub."
fi

# ---------------------------------------------------------------------------
# Install flash_attn from cached wheel if missing.
# grpo_math_1B.yaml defaults sequence_packing=true, which forces
# attn_implementation="flash_attention_2" at model init time (see
# nemo_rl/models/automodel/setup.py:307). The DTensor worker venv predates
# the flash-attn dep being added to pyproject, so we patch it in from the
# pre-built wheel under .cache/pip/wheels/.
# ---------------------------------------------------------------------------
FLASH_WHEEL="$REPO_DIR/.cache/pip/wheels/88/3e/3b/6b21b8f1b536ccdac5473d3afc2e8c8405cc1ce954fb9da941/flash_attn-2.8.1-cp312-cp312-linux_x86_64.whl"
if [[ ! -d "$DTENSOR_V2_VENV/lib/python3.12/site-packages/flash_attn" ]]; then
  if [[ -f "$FLASH_WHEEL" ]]; then
    echo "Installing flash_attn 2.8.1 from cached wheel..."
    uv pip install --python "$DTENSOR_V2_VENV/bin/python" "$FLASH_WHEEL"
  else
    echo "WARNING: flash_attn missing from DTensor venv and no cached wheel at $FLASH_WHEEL." >&2
    echo "         Run may fail on models whose HF config requests flash_attention_2." >&2
  fi
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "=================================================================="
echo " GRPO reasoning_gym training  (job ${SLURM_JOB_ID:-interactive})"
echo "=================================================================="
echo "  Repo:        $REPO_DIR"
echo "  Model:       ${MODEL:-<config default>}"
echo "  Categories:  ${CATEGORIES[*]}"
echo "  Fractions:"
for cat in "${CATEGORIES[@]}"; do
  [[ -n "${FRAC[$cat]}" ]] && printf '    %-12s %s\n' "$cat" "${FRAC[$cat]}"
done
echo "  Overrides:   ${OVERRIDES[*]:-<none>}"
echo "=================================================================="

exec uv run python "$REPO_DIR/examples/run_grpo.py" \
  --config "$REPO_DIR/examples/reasoning_gym/grpo_reasoning_gym.yaml" \
  "${OVERRIDES[@]}"
