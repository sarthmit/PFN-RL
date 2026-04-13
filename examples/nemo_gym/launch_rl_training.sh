#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Launch mixed-capability GRPO training with configurable environments, proportions,
# learning rate, and model.
#
# Usage:
#   bash examples/nemo_gym/launch_rl_training.sh \
#       --model Qwen/Qwen3-8B \
#       --math 0.40 --code 0.28 --gpqa 0.16 --reasoning 0.16 \
#       --lr 5e-6 \
#       --steps 1000 \
#       --max-seq-len 8192
#
# Environment flags (pass 0 or omit to disable):
#   --math FLOAT        math_with_judge   (competition math, targets AIME)
#   --code FLOAT        code_gen          (competitive coding, targets LiveCodeBench)
#   --mcqa FLOAT        mcqa              (graduate science MCQ, targets GPQA Diamond)
#   --reasoning FLOAT   reasoning_gym     (broad reasoning, 100+ task types)
#   --instruct FLOAT    instruction_following (targets IFEval / IFBench)
#
# Required:
#   --model STRING      HuggingFace model name or local path
#
# Optional:
#   --lr FLOAT          Learning rate (default: 5e-6)
#   --min-lr FLOAT      Min LR for scheduler (default: lr/10)
#   --steps INT         Max training steps (default: 1000)
#   --max-seq-len INT   Max total sequence length for training and generation (default: 8192)
#   --num-prompts INT   Prompts per step (default: 64)
#   --num-generations INT Generations per prompt (default: 16)
#   --dtensor-tp INT    DTensor tensor parallel size for training (default: 1)
#   --vllm-tp INT       vLLM tensor parallel size for generation (default: 1)
#   --gpus-per-node INT GPUs per node (default: 8)
#   --nodes INT         Number of nodes (default: 1)
#   --output-dir STRING Base directory; checkpoints go to <dir>/<run>/checkpoints, logs to <dir>/<run>/logs (default: results/grpo-custom)
#   --checkpoint-dir STRING Explicit checkpoint directory (overrides --output-dir derived path)
#   --log-dir STRING    Explicit log directory for tensorboard/wandb/nemo_gym logs (overrides --output-dir derived path)
#   --wandb             Enable Weights & Biases logging
#   --wandb-project STRING WandB project name (default: grpo-mixed)
#   --dry-run           Print generated config and exit without launching

set -euo pipefail

# Ensure uv is on PATH (installed to ~/.local/bin)
export PATH="$HOME/.local/bin:$PATH"

# Redirect all tool caches to scratch to avoid filling home quota.
# ~/.bashrc sets these for interactive shells, but Ray workers and subprocesses
# (launched non-interactively) would default to ~/.cache without this.
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
VENV_SITE="$(cd "$(dirname "$0")/../.." && pwd)/.venv/lib/python3.12/site-packages"
export CUDA_HOME="$VENV_SITE/nvidia/cuda_runtime"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODEL=""
LR="5e-6"
MIN_LR=""           # derived from LR if not set
MATH=0
CODE=0
MCQA=0
REASONING=0
INSTRUCT=0
STEPS=3
MAX_SEQ_LEN=8192
NUM_PROMPTS=16
NUM_GENERATIONS=8
DTENSOR_TP=1
VLLM_TP=1
GPUS_PER_NODE=4
NODES=1
OUTPUT_DIR="results/grpo-custom"
CHECKPOINT_DIR=""   # derived from OUTPUT_DIR if not set
LOG_DIR=""          # derived from OUTPUT_DIR if not set
WANDB=true
WANDB_PROJECT="grpo-mixed"
DRY_RUN=false
BACKEND="dtensor"   # dtensor or megatron
TRAIN_MBS=""        # train_micro_batch_size override (empty = use config default)
LOGPROB_BS=""       # logprob_batch_size override (empty = use config default)
GPU_MEM_UTIL=""     # vLLM gpu_memory_utilization override (empty = use config default)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)            MODEL="$2";           shift 2 ;;
    --lr)               LR="$2";              shift 2 ;;
    --min-lr)           MIN_LR="$2";          shift 2 ;;
    --math)             MATH="$2";            shift 2 ;;
    --code)             CODE="$2";            shift 2 ;;
    --mcqa)             MCQA="$2";            shift 2 ;;
    --reasoning)        REASONING="$2";       shift 2 ;;
    --instruct)         INSTRUCT="$2";        shift 2 ;;
    --steps)            STEPS="$2";           shift 2 ;;
    --max-seq-len)      MAX_SEQ_LEN="$2";     shift 2 ;;
    --num-prompts)      NUM_PROMPTS="$2";     shift 2 ;;
    --num-generations)  NUM_GENERATIONS="$2"; shift 2 ;;
    --dtensor-tp)       DTENSOR_TP="$2";      shift 2 ;;
    --vllm-tp)          VLLM_TP="$2";         shift 2 ;;
    --gpus-per-node)    GPUS_PER_NODE="$2";   shift 2 ;;
    --nodes)            NODES="$2";           shift 2 ;;
    --output-dir)       OUTPUT_DIR="$2";      shift 2 ;;
    --checkpoint-dir)   CHECKPOINT_DIR="$2"; shift 2 ;;
    --log-dir)          LOG_DIR="$2";        shift 2 ;;
    --wandb)            WANDB=true;           shift ;;
    --wandb-project)    WANDB_PROJECT="$2";   shift 2 ;;
    --dry-run)          DRY_RUN=true;         shift ;;
    --backend)          BACKEND="$2";         shift 2 ;;
    --mbs)              TRAIN_MBS="$2";       shift 2 ;;
    --lbs)              LOGPROB_BS="$2";      shift 2 ;;
    --gpu-mem-util)     GPU_MEM_UTIL="$2";    shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$MODEL" ]]; then
  echo "ERROR: --model is required."
  echo "Example: --model Qwen/Qwen3-8B"
  exit 1
fi

TOTAL_FRACTION=$(python3 -c "print($MATH + $CODE + $MCQA + $REASONING + $INSTRUCT)")
if python3 -c "import sys; sys.exit(0 if float('$TOTAL_FRACTION') > 0 else 1)"; then
  : # at least one env enabled
else
  echo "ERROR: at least one environment fraction must be > 0."
  echo "Pass e.g. --math 0.5 --code 0.5"
  exit 1
fi

# Derive min-lr from lr if not set
if [[ -z "$MIN_LR" ]]; then
  MIN_LR=$(python3 -c "print(float('$LR') / 10)")
fi

# Derive DP size and run name (DP is determined by DTensor TP)
DP=$(python3 -c "print($GPUS_PER_NODE * $NODES // $DTENSOR_TP)")

# Derive checkpoint/log dirs from output-dir if not explicitly set (done after RUN_NAME is known)
RUN_NAME=$(python3 -c "
parts = []
for name, val in [('math', $MATH), ('code', $CODE), ('mcqa', $MCQA), ('rsn', $REASONING), ('inst', $INSTRUCT)]:
    if val > 0:
        parts.append(f'{name}{val:.0%}')
model_short = '$MODEL'.split('/')[-1].lower()
print(f'{model_short}-' + '-'.join(parts))
")

# Derive checkpoint/log dirs after RUN_NAME is known
if [[ -z "$CHECKPOINT_DIR" ]]; then
  CHECKPOINT_DIR="$OUTPUT_DIR/$RUN_NAME/checkpoints"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="$OUTPUT_DIR/$RUN_NAME/logs"
fi

echo "========================================================"
echo " NeMo-RL Mixed GRPO Training"
echo "========================================================"
echo "  Model:       $MODEL"
echo "  LR:          $LR  (min: $MIN_LR)"
echo "  Steps:       $STEPS"
echo "  Seq len:     $MAX_SEQ_LEN"
echo "  Batch:       ${NUM_PROMPTS} prompts × ${NUM_GENERATIONS} gen = $(python3 -c "print($NUM_PROMPTS * $NUM_GENERATIONS)") samples/step  MBS=${TRAIN_MBS:-default}  LBS=${LOGPROB_BS:-default}"
echo "  Backend:     $BACKEND"
echo "  Cluster:     ${NODES} node(s) × ${GPUS_PER_NODE} GPUs  (DTensor TP=${DTENSOR_TP}, vLLM TP=${VLLM_TP}, DP=${DP})"
echo "  Environments:"
[[ $(echo "$MATH > 0" | python3 -c "import sys; a=float(input()); sys.exit(0 if a>0 else 1)" <<< "$MATH") ]] || python3 -c "
fracs = {'math_with_judge': $MATH, 'code_gen': $CODE, 'mcqa': $MCQA, 'reasoning_gym': $REASONING, 'instruction_following': $INSTRUCT}
total = sum(fracs.values())
for env, f in fracs.items():
    if f > 0:
        print(f'    {env:<25} fraction={f}  ({f/total:.1%} of batches)')
"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Logs:        $LOG_DIR"
echo "========================================================"

# ---------------------------------------------------------------------------
# Pre-build worker venvs on this node (which has GPU access) before Ray starts.
# The _env_builder Ray actor runs with num_cpus=1 and no GPU, so packages like
# nv-grouped-gemm that call into CUDA during their build fail there. Building
# here first means _env_builder finds python_path.exists() == True and skips.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
DTENSOR_V2_VENV="$SCRIPT_DIR/venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
if [[ ! -f "$DTENSOR_V2_VENV/bin/python" ]]; then
  echo "Pre-building DTensorPolicyWorkerV2 venv (requires GPU, doing it before Ray starts)..."
  UV_PROJECT_ENVIRONMENT="$DTENSOR_V2_VENV" uv sync --locked --extra automodel --directory "$SCRIPT_DIR"
  echo "DTensorPolicyWorkerV2 venv ready."
fi

# Inject transformer-engine dist-info stub if missing (Megatron-LM checks TE version at import
# time; TE cannot be built from source on Mila bare-metal nodes).
TE_DIST_INFO="$DTENSOR_V2_VENV/lib/python3.12/site-packages/transformer_engine-2.14.0+71bbefb.dist-info"
if [[ ! -d "$TE_DIST_INFO" ]]; then
  mkdir -p "$TE_DIST_INFO"
  printf 'Metadata-Version: 2.1\nName: transformer-engine\nVersion: 2.14.0+71bbefb\nSummary: Transformer acceleration library (stub)\nRequires-Python: >=3.8\n' > "$TE_DIST_INFO/METADATA"
  touch "$TE_DIST_INFO/RECORD"
  echo "Injected transformer-engine dist-info stub."
fi

# Pre-build the shared NemoGym vllm_model venv. All four NemoGym actors (one
# per environment) share a single .venv at responses_api_models/vllm_model/.venv
# and they start concurrently, creating a race. They also downgrade openai
# (2.7.x → 2.6.1) which leaves stale .pyc files that cause FileNotFoundError
# on retry.  Building here first means each actor sees python already present
# and skips_venv_if_present. The path inside Gym uses the uv_venv_dir default.
GYM_DIR="$SCRIPT_DIR/3rdparty/Gym-workspace/Gym"
VLLM_MODEL_VENV="$GYM_DIR/responses_api_models/vllm_model/.venv"
if [[ ! -f "$VLLM_MODEL_VENV/bin/python" ]]; then
  echo "Pre-building NemoGym vllm_model venv..."
  (cd "$GYM_DIR/responses_api_models/vllm_model" && \
   uv venv --seed --python 3.12.13 .venv && \
   uv pip install --python .venv/bin/python -e . 'ray[default]==2.54.0' 'openai==2.6.1')
  echo "NemoGym vllm_model venv ready."
fi

# Pre-build resource server venvs for each active environment. Without this,
# venv installation happens inside the NemoGym Ray actor's __init__, which
# blocks setup even when skip_venv_if_present=true (the flag only skips if the
# venv already exists). Building here means every subsequent run hits the skip.
_NEMO_GYM_PYTHON="3.12.13"
_NEMO_GYM_DEPS=('ray[default]==2.54.0' 'openai==2.6.1')
for _env_frac in "math_with_judge:$MATH" "code_gen:$CODE" "mcqa:$MCQA" "reasoning_gym:$REASONING" "instruction_following:$INSTRUCT"; do
  _env="${_env_frac%%:*}"
  _frac="${_env_frac##*:}"
  if python3 -c "import sys; sys.exit(0 if float('$_frac') > 0 else 1)" 2>/dev/null; then
    _rsrc_venv="$GYM_DIR/resources_servers/$_env/.venv"
    if [[ ! -f "$_rsrc_venv/bin/python" ]]; then
      echo "Pre-building NemoGym $_env venv..."
      (cd "$GYM_DIR/resources_servers/$_env" && \
       uv venv --seed --python "$_NEMO_GYM_PYTHON" .venv && \
       uv pip install --python .venv/bin/python -r requirements.txt "${_NEMO_GYM_DEPS[@]}")
      echo "NemoGym $_env venv ready."
    fi
  fi
done
unset _env_frac _env _frac _rsrc_venv _NEMO_GYM_PYTHON _NEMO_GYM_DEPS

# ---------------------------------------------------------------------------
# Generate temp config via Python (uses OmegaConf to preserve interpolations)
# ---------------------------------------------------------------------------
mkdir -p "$LOG_DIR"
TEMP_CONFIG="$LOG_DIR/generated_config.yaml"

uv run python - <<PYTHON
import sys
sys.path.insert(0, ".")

from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path

# Load base config (backend-specific)
backend = "$BACKEND"
if backend == "megatron":
    config_file = "examples/nemo_gym/grpo_mixed_capabilities_megatron.yaml"
else:
    config_file = "examples/nemo_gym/grpo_mixed_capabilities_dtensor.yaml"
base = OmegaConf.load(config_file)

# ------------------------------------------------------------------
# Environment metadata: data paths and Gym config_paths per env
# ------------------------------------------------------------------
GYM = "3rdparty/Gym-workspace/Gym"
ENV_META = {
    "math_with_judge": dict(
        train_path=f"{GYM}/resources_servers/math_with_judge/data/OpenMathReasoning_train.jsonl",
        val_path=f"{GYM}/resources_servers/math_with_judge/data/OpenMathReasoning_aime24_validation.jsonl",
        config_paths=[
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            "resources_servers/math_with_judge/configs/math_with_judge.yaml",
        ],
    ),
    "code_gen": dict(
        train_path=f"{GYM}/resources_servers/code_gen/data/opencodereasoning_filtered_25k_train.jsonl",
        val_path=f"{GYM}/resources_servers/code_gen/data/livecodebench_v5_2024-07-01_2025-02-01_validation.jsonl",
        config_paths=[
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            "resources_servers/code_gen/configs/code_gen.yaml",
        ],
    ),
    "mcqa": dict(
        train_path=f"{GYM}/resources_servers/mcqa/data/train.jsonl",
        val_path=None,
        config_paths=[
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            "resources_servers/mcqa/configs/mcqa.yaml",
        ],
    ),
    "reasoning_gym": dict(
        train_path=f"{GYM}/resources_servers/reasoning_gym/data/Nemotron-RL-ReasoningGym-v1_train.jsonl",
        val_path=None,
        config_paths=[
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            "resources_servers/reasoning_gym/configs/reasoning_gym.yaml",
        ],
    ),
    "instruction_following": dict(
        train_path=f"{GYM}/resources_servers/instruction_following/data/train.jsonl",
        val_path=None,
        config_paths=[
            "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
            "resources_servers/instruction_following/configs/instruction_following.yaml",
        ],
    ),
}

requested = {
    "math_with_judge":      $MATH,
    "code_gen":             $CODE,
    "mcqa":                 $MCQA,
    "reasoning_gym":        $REASONING,
    "instruction_following": $INSTRUCT,
}
active = {k: v for k, v in requested.items() if v > 0}

# ------------------------------------------------------------------
# data.train — only active environments
# ------------------------------------------------------------------
train_datasets = []
for env_name, fraction in active.items():
    meta = ENV_META[env_name]
    train_datasets.append({
        "data_path": meta["train_path"],
        "env_name":  env_name,
        "fraction":  fraction,
        "dataset_name": "NemoGymDataset",
        "processor": "nemo_gym_data_processor",
        "prompt_file": None,
        "system_prompt_file": None,
    })

# data.validation — only envs that have a val split
val_datasets = []
for env_name in active:
    val_path = ENV_META[env_name]["val_path"]
    if val_path:
        val_datasets.append({
            "data_path": val_path,
            "env_name":  env_name,
            "dataset_name": "NemoGymDataset",
            "processor": "nemo_gym_data_processor",
            "prompt_file": None,
            "system_prompt_file": None,
        })

# ------------------------------------------------------------------
# env.nemo_gym_envs — only active environments
# ------------------------------------------------------------------
gym_abs_path = str(Path(GYM).absolute())
nemo_gym_envs = {}
for env_name in active:
    meta = ENV_META[env_name]
    nemo_gym_envs[env_name] = {
        "rollout_max_attempts_to_avoid_lp_nan": 1,
        "config_paths": meta["config_paths"],
        "policy_model": {
            "responses_api_models": {
                "vllm_model": {"uses_reasoning_parser": False}
            }
        },
        "skip_venv_if_present": True,
        "uv_venv_dir": gym_abs_path,
        "nemo_gym_log_dir": str(Path("$LOG_DIR").absolute() / "nemo_gym"),
        "agent_name": f"{env_name}_simple_agent",
    }

# ------------------------------------------------------------------
# Apply overrides to base config
# ------------------------------------------------------------------
# Policy
OmegaConf.update(base, "policy.model_name", "$MODEL")
OmegaConf.update(base, "policy.max_total_sequence_length", $MAX_SEQ_LEN)
OmegaConf.update(base, "policy.tokenizer.name", "\${policy.model_name}")
OmegaConf.update(base, "policy.generation.vllm_cfg.max_model_len", $MAX_SEQ_LEN)

if backend == "megatron":
    OmegaConf.update(base, "policy.megatron_cfg.tensor_model_parallel_size", $DTENSOR_TP)
    OmegaConf.update(base, "policy.megatron_cfg.optimizer.lr", float("$LR"))
    OmegaConf.update(base, "policy.megatron_cfg.optimizer.min_lr", float("$MIN_LR"))
    OmegaConf.update(base, "policy.megatron_cfg.scheduler.lr_warmup_init", float("$MIN_LR"))
else:
    OmegaConf.update(base, "policy.dtensor_cfg.tensor_parallel_size", $DTENSOR_TP)
    OmegaConf.update(base, "policy.optimizer.kwargs.lr", float("$LR"))
OmegaConf.update(base, "policy.generation.vllm_cfg.tensor_parallel_size", $VLLM_TP)

# Optional MBS / LBS / gpu_memory_utilization overrides
if "$TRAIN_MBS":
    OmegaConf.update(base, "policy.train_micro_batch_size", int("$TRAIN_MBS"))
if "$LOGPROB_BS":
    OmegaConf.update(base, "policy.logprob_batch_size", int("$LOGPROB_BS"))
if "$GPU_MEM_UTIL":
    OmegaConf.update(base, "policy.generation.vllm_cfg.gpu_memory_utilization", float("$GPU_MEM_UTIL"))

# GRPO
OmegaConf.update(base, "grpo.max_num_steps", $STEPS)
OmegaConf.update(base, "grpo.num_prompts_per_step", $NUM_PROMPTS)
OmegaConf.update(base, "grpo.num_generations_per_prompt", $NUM_GENERATIONS)

# Cluster
OmegaConf.update(base, "cluster.gpus_per_node", $GPUS_PER_NODE)
OmegaConf.update(base, "cluster.num_nodes", $NODES)

# Logging
run_name = "$RUN_NAME"
OmegaConf.update(base, "logger.log_dir", "$LOG_DIR")
OmegaConf.update(base, "logger.wandb_enabled", $([[ "$WANDB" == "true" ]] && echo "True" || echo "False"))
OmegaConf.update(base, "logger.wandb.project", "$WANDB_PROJECT")
OmegaConf.update(base, "logger.wandb.name", run_name)
OmegaConf.update(base, "checkpointing.checkpoint_dir", "$CHECKPOINT_DIR")

# Data and environments
OmegaConf.update(base, "data.train", train_datasets)
OmegaConf.update(base, "data.validation", val_datasets if val_datasets else None)
OmegaConf.update(base, "env.nemo_gym_envs", nemo_gym_envs)

OmegaConf.save(base, "$TEMP_CONFIG")
print(f"Generated config: $TEMP_CONFIG")
PYTHON

# ---------------------------------------------------------------------------
# Dry run: print config and exit
# ---------------------------------------------------------------------------
if $DRY_RUN; then
  echo ""
  echo "--- Generated config (dry run) ---"
  cat "$TEMP_CONFIG"
  echo ""
  echo "Dry run complete. Remove --dry-run to launch training."
  exit 0
fi

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------
echo ""
echo "Launching training..."
uv run python examples/nemo_gym/run_grpo_nemo_gym.py --config "$TEMP_CONFIG"

# UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25
# UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-4B-Instruct-2507 --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25