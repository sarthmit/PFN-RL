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
# Benchmark suite for the mixed-capability training run.
#
# Benchmarks covered:
#   - AIME 2024       (math_with_judge, num_repeats=32)
#   - AIME 2025       (math_with_judge, num_repeats=32)
#   - GPQA Diamond    (mcqa,            num_repeats=8)
#   - LiveCodeBench v6 (code_gen,       num_repeats=3)
#   - Reasoning Gym   (reasoning_gym,   num_repeats=8, validation split)
#
# Not available in NeMo Gym:
#   - HMMT:  no benchmark config exists in this Gym workspace.
#   - IFEval: no dedicated benchmark config; instruction_following trains
#             toward IFEval/IFBench but there is no packaged benchmark runner.
#
# Prerequisites:
#   1. Set env.yaml in 3rdparty/Gym-workspace/Gym/ with your model endpoint:
#        policy_base_url: http://localhost:8000/v1
#        policy_api_key:  EMPTY
#        policy_model_name: <your-model>
#   2. Run this script from the NeMo-RL repo root.
#   3. Prepare benchmark data (first run only):
#        bash examples/nemo_gym/run_benchmark_suite.sh --prepare-only
#
# Usage:
#   bash examples/nemo_gym/run_benchmark_suite.sh [--prepare-only] [--results-dir DIR]

set -euo pipefail

GYM_DIR="3rdparty/Gym-workspace/Gym"
RESULTS_DIR="results/benchmark-suite"
PREPARE_ONLY=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-only)  PREPARE_ONLY=true; shift ;;
    --results-dir)   RESULTS_DIR="$2";  shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Helper: run a single benchmark end-to-end
#   $1 = benchmark name (for logging)
#   $2 = benchmark config path (relative to GYM_DIR)
#   $3 = agent name
#   $4 = input JSONL (relative to GYM_DIR)
#   $5 = num_repeats (passed as responses_create_params override)
# ---------------------------------------------------------------------------
run_benchmark() {
  local name="$1"
  local bench_cfg="$2"
  local agent_name="$3"
  local input_jsonl="$4"
  local num_repeats="$5"

  local out_rollouts="$RESULTS_DIR/${name}_rollouts.jsonl"
  local out_profiled="$RESULTS_DIR/${name}_profiled.jsonl"

  echo ""
  echo "========================================================"
  echo " Benchmark: $name"
  echo "========================================================"

  # Step 1: start servers in background
  echo "[${name}] Starting servers..."
  pushd "$GYM_DIR" > /dev/null
  ng_run "+config_paths=[${bench_cfg},responses_api_models/vllm_model/configs/vllm_model.yaml]" &
  local server_pid=$!
  popd > /dev/null

  # Step 2: wait for servers to be ready
  echo "[${name}] Waiting for servers to be ready..."
  pushd "$GYM_DIR" > /dev/null
  local max_wait=300
  local waited=0
  until ng_status 2>/dev/null | grep -q "All.*servers ready" || [ $waited -ge $max_wait ]; do
    sleep 5
    waited=$((waited + 5))
  done
  if [ $waited -ge $max_wait ]; then
    echo "[${name}] ERROR: servers did not become ready within ${max_wait}s"
    kill "$server_pid" 2>/dev/null || true
    popd > /dev/null
    return 1
  fi
  echo "[${name}] Servers ready."

  # Step 3: collect rollouts
  echo "[${name}] Collecting rollouts (num_repeats=${num_repeats})..."
  ng_collect_rollouts \
    +agent_name="${agent_name}" \
    +input_jsonl_fpath="${input_jsonl}" \
    +output_jsonl_fpath="../../${out_rollouts}" \
    +num_repeats="${num_repeats}"
  popd > /dev/null

  # Step 4: profile pass rates
  echo "[${name}] Profiling..."
  pushd "$GYM_DIR" > /dev/null
  ng_reward_profile \
    +input_jsonl_fpath="${input_jsonl}" \
    +rollouts_jsonl_fpath="../../${out_rollouts}" \
    +output_jsonl_fpath="../../${out_profiled}" \
    +pass_threshold=1.0
  popd > /dev/null

  # Step 5: stop servers
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true

  echo "[${name}] Done. Profiled results: ${out_profiled}"
}

# ---------------------------------------------------------------------------
# Step 0: prepare benchmark data
# ---------------------------------------------------------------------------
echo "========================================================"
echo " Preparing benchmark data"
echo "========================================================"

pushd "$GYM_DIR" > /dev/null

ng_prepare_benchmark "+config_paths=[benchmarks/aime24/config.yaml]"
ng_prepare_benchmark "+config_paths=[benchmarks/aime25/config.yaml]"
ng_prepare_benchmark "+config_paths=[benchmarks/gpqa/config.yaml]"
ng_prepare_benchmark "+config_paths=[benchmarks/livecodebench/v6_2408_2505/config.yaml]"

popd > /dev/null

if $PREPARE_ONLY; then
  echo "Data preparation complete. Exiting (--prepare-only)."
  exit 0
fi

# ---------------------------------------------------------------------------
# Run each benchmark
# ---------------------------------------------------------------------------

run_benchmark \
  "aime24" \
  "benchmarks/aime24/config.yaml" \
  "aime24_math_with_judge_simple_agent" \
  "benchmarks/aime24/data/aime24_benchmark.jsonl" \
  32

run_benchmark \
  "aime25" \
  "benchmarks/aime25/config.yaml" \
  "aime25_math_with_judge_simple_agent" \
  "benchmarks/aime25/data/aime25_benchmark.jsonl" \
  32

run_benchmark \
  "gpqa_diamond" \
  "benchmarks/gpqa/config.yaml" \
  "gpqa_mcqa_simple_agent" \
  "benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl" \
  8

run_benchmark \
  "livecodebench_v6" \
  "benchmarks/livecodebench/v6_2408_2505/config.yaml" \
  "livecodebench_v6_code_gen_simple_agent" \
  "benchmarks/livecodebench/v6_2408_2505/data/livecodebench_v6_validation.jsonl" \
  3

# Reasoning Gym: no packaged benchmark config, run against the validation split
# using the standard resources server config and 8 repeats for variance control.
run_benchmark \
  "reasoning_gym" \
  "resources_servers/reasoning_gym/configs/reasoning_gym.yaml" \
  "reasoning_gym_simple_agent" \
  "resources_servers/reasoning_gym/data/Nemotron-RL-ReasoningGym-v1_train.jsonl" \
  8

# ---------------------------------------------------------------------------
# Aggregate all results
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo " Aggregate results"
echo "========================================================"

pushd "$GYM_DIR" > /dev/null
for profiled in ../../"$RESULTS_DIR"/*_profiled.jsonl; do
  bench=$(basename "$profiled" _profiled.jsonl)
  echo ""
  echo "--- ${bench} ---"
  python scripts/print_aggregate_results.py +jsonl_fpath="${profiled}"
done
popd > /dev/null

echo ""
echo "All benchmark results saved to: ${RESULTS_DIR}/"
