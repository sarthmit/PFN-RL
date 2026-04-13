#!/usr/bin/env bash
set -euo pipefail

# Benchmark runner: TP/MBS/LBS sweep for Qwen3-0.6B on 4x A100-80GB
# Writes timing results to /network/scratch/m/mittalsa/RL/memory.md after each run

RLDIR="/network/scratch/m/mittalsa/RL"
BENCH_LOG="$RLDIR/bench_results.log"
MEMORY_MD="$RLDIR/memory.md"

cd "$RLDIR"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$RLDIR/.cache"
export HF_HOME="$RLDIR/.cache"
export VLLM_CACHE_DIR="$RLDIR/.cache"
export TRITON_CACHE_DIR="$RLDIR/.cache/triton"
export TORCH_HOME="$RLDIR/.cache/torch"
export RAY_TMPDIR="$RLDIR/.cache/ray"
export XDG_CACHE_HOME="$RLDIR/.cache"
VENV_SITE="$RLDIR/.venv/lib/python3.12/site-packages"
export CUDA_HOME="$VENV_SITE/nvidia/cuda_runtime"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# Common launch args (keep same prompts/generations as exp_054 baseline)
COMMON_ARGS="--model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 --steps 4 --num-prompts 64 --num-generations 8 --gpus-per-node 4"

# Extract timing from a log file
extract_timing() {
    local logfile="$1"
    grep "Total step time:" "$logfile" | tail -3 | awk '{print $NF}' | tr -d 's'
}

extract_phase() {
    local logfile="$1"
    local phase="$2"
    grep "  • $phase:" "$logfile" | tail -3 | awk '{print $NF}' | tr -d 's' | head -1
}

run_experiment() {
    local name="$1"
    local tp="$2"
    local mbs="$3"
    local lbs="$4"
    local outfile="$RLDIR/bench_${name}.log"

    echo "===== STARTING $name: TP=$tp MBS=$mbs LBS=$lbs =====" | tee -a "$BENCH_LOG"
    echo "$(date)" | tee -a "$BENCH_LOG"

    # Generate config via dry-run
    local cfgfile="$RLDIR/results/bench-runs/bench_${name}/logs/generated_config.yaml"
    mkdir -p "$(dirname "$cfgfile")"

    bash ./examples/nemo_gym/launch_rl_training.sh \
        $COMMON_ARGS \
        --dtensor-tp "$tp" \
        --vllm-tp 1 \
        --output-dir "$RLDIR/results/bench-runs" \
        --dry-run 2>&1 | tee "$outfile.dryrun" || true

    # Find the generated config
    local found_cfg
    found_cfg=$(find "$RLDIR/results/bench-runs" -name "generated_config.yaml" -newer "$RLDIR/memory.md" 2>/dev/null | head -1)
    if [[ -z "$found_cfg" ]]; then
        # dry-run outputs to a derived path; re-run with explicit log-dir
        local logdir="$RLDIR/results/bench-runs/bench_${name}/logs"
        mkdir -p "$logdir"
        bash ./examples/nemo_gym/launch_rl_training.sh \
            $COMMON_ARGS \
            --dtensor-tp "$tp" \
            --vllm-tp 1 \
            --log-dir "$logdir" \
            --checkpoint-dir "$RLDIR/results/bench-runs/bench_${name}/checkpoints" \
            --dry-run 2>&1 | tee -a "$outfile.dryrun"
        found_cfg="$logdir/generated_config.yaml"
    fi

    echo "Config at: $found_cfg"

    # Patch MBS and LBS into the config
    python3 - "$found_cfg" "$mbs" "$lbs" << 'PY'
import sys
from omegaconf import OmegaConf

cfg_path, mbs, lbs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
cfg = OmegaConf.load(cfg_path)
OmegaConf.update(cfg, "policy.train_micro_batch_size", mbs)
OmegaConf.update(cfg, "policy.logprob_batch_size", lbs)
OmegaConf.save(cfg, cfg_path)
print(f"Patched: train_micro_batch_size={mbs}, logprob_batch_size={lbs}")
PY

    # Run training
    echo "Running training..." | tee -a "$BENCH_LOG"
    uv run python examples/nemo_gym/run_grpo_nemo_gym.py \
        --config "$found_cfg" \
        2>&1 | tee "$outfile"

    # Extract timings (last 3 complete steps to get stable average)
    local step_times
    step_times=$(grep "Total step time:" "$outfile" | awk '{print $NF}' | tr -d 's')
    local avg_time
    avg_time=$(echo "$step_times" | tail -3 | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n}')
    local train_time
    train_time=$(grep "  • policy_training:" "$outfile" | awk '{print $NF}' | tr -d 's' | tail -3 | awk '{s+=$1;n++} END {if(n>0) printf "%.1f",s/n}')
    local lp_time
    lp_time=$(grep "  • policy_and_reference_logprobs:" "$outfile" | awk '{print $NF}' | tr -d 's' | tail -3 | awk '{s+=$1;n++} END {if(n>0) printf "%.1f",s/n}')
    local gen_time
    gen_time=$(grep "  • generation:" "$outfile" | awk '{print $NF}' | tr -d 's' | tail -3 | awk '{s+=$1;n++} END {if(n>0) printf "%.1f",s/n}')

    echo "RESULT $name: avg_step=${avg_time}s train=${train_time}s logprobs=${lp_time}s gen=${gen_time}s" | tee -a "$BENCH_LOG"

    # Update memory.md
    python3 - "$MEMORY_MD" "$name" "$tp" "$mbs" "$lbs" "$avg_time" "$train_time" "$lp_time" "$gen_time" << 'PY'
import sys, re
md, name, tp, mbs, lbs, total, train, lp, gen = sys.argv[1:]
with open(md) as f:
    content = f.read()
# Replace the TBD row for this experiment
old = f"| {name.replace('_',' ')} |.*| TBD |.*|"
pattern = re.compile(rf"\| Exp {name[-1].upper()} \|.*")
new_row = f"| Exp {name[-1].upper()} | {tp} | {mbs} | {lbs} | **{total}s** | {train}s | {lp}s | {gen}s | |"
new_content = pattern.sub(new_row, content)
with open(md, 'w') as f:
    f.write(new_content)
print(f"Updated memory.md: {new_row}")
PY
}

echo "=== BENCHMARK START $(date) ===" | tee "$BENCH_LOG"

run_experiment "exp_a" 1 8 8
run_experiment "exp_b" 1 16 16
run_experiment "exp_c" 1 32 16
run_experiment "exp_d" 2 8 8

echo "=== BENCHMARK COMPLETE $(date) ===" | tee -a "$BENCH_LOG"
