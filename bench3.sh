#!/usr/bin/env bash
# bench3: fix OOM fragmentation + test TP=4 with higher MBS/LBS
# Key fixes vs bench2:
#   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  (fixes fragmentation OOM)
#   - gpu_memory_utilization=0.7 (reduces vLLM KV cache pressure, ~22GB freed/GPU)
# Baseline (exp_054, H100): TP=4 MBS=4 LBS=4 → 131s/step
set -euo pipefail

RLDIR="/network/scratch/m/mittalsa/RL"
RESULTS="$RLDIR/bench3_results.txt"
STEPS=2
MODEL="Qwen/Qwen3-0.6B"

export RAY_TMPDIR="/tmp/r"
export UV_CACHE_DIR="$RLDIR/.cache"
export HF_HOME="$RLDIR/.cache"
export VLLM_CACHE_DIR="$RLDIR/.cache"
# Fix PyTorch fragmentation OOM: allows non-contiguous block reuse
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p /tmp/r

echo "==== bench3 started: $(date) ====" | tee -a "$RESULTS"

run_exp() {
    local name="$1"; local tp="$2"; local mbs="$3"; local lbs="$4"; local util="$5"
    echo "" | tee -a "$RESULTS"
    echo "---- $name: TP=$tp MBS=$mbs LBS=$lbs gpu_util=$util ----" | tee -a "$RESULTS"
    echo "Start: $(date)" | tee -a "$RESULTS"
    cd "$RLDIR"
    timeout 1800 bash examples/nemo_gym/launch_rl_training.sh \
        --model "$MODEL" \
        --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 \
        --steps "$STEPS" \
        --dtensor-tp "$tp" \
        --vllm-tp 1 \
        --mbs "$mbs" \
        --lbs "$lbs" \
        --gpu-mem-util "$util" \
        --gpus-per-node 4 \
        --num-prompts 64 \
        --num-generations 8 \
        --output-dir "$RLDIR/bench3_runs/$name" \
        2>&1 | tee -a "$RESULTS" || echo "FAILED: $name" | tee -a "$RESULTS"
    echo "End: $(date)" | tee -a "$RESULTS"
    pkill -f "ray" 2>/dev/null || true
    pkill -f "vllm" 2>/dev/null || true
    sleep 20
}

# TP=4 (DP=1): logits = MBS × 8192 × (151552/4) × 4 bytes
# At MBS=8: 8 × 8192 × 37888 × 4 = 9.3 GB/GPU → fits easily even at util=0.7

# Exp A: TP=4 MBS=8 LBS=8 (2× baseline MBS and LBS)
run_exp "ExpA_TP4_MBS8_LBS8"   4  8  8  0.7

# Exp B: TP=4 MBS=8 LBS=16 (4× baseline LBS)
run_exp "ExpB_TP4_MBS8_LBS16"  4  8 16  0.7

# Exp C: TP=4 MBS=16 LBS=16 (4× baseline MBS, 4× LBS)
run_exp "ExpC_TP4_MBS16_LBS16" 4 16 16  0.7

# Exp D: TP=4 MBS=4 LBS=4 (baseline config, util=0.7) — control to check util=0.7 effect vs baseline
run_exp "ExpD_TP4_MBS4_LBS4_ctrl" 4  4  4  0.7

echo "" | tee -a "$RESULTS"
echo "==== bench3 complete: $(date) ====" | tee -a "$RESULTS"
