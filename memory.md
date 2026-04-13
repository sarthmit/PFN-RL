# Step-Time Optimization: Qwen3-0.6B on 4× A100-80GB

## Setup
- Model: Qwen/Qwen3-0.6B
- 4× A100-80GB (cn-g021)
- vLLM TP=1 (fixed: 4 independent vLLM instances colocated)
- Run command: `UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25`
- GRPO config: 64 prompts × 8 generations = 512 sequences/step, max_seq_len=8192

## Phase Breakdown (from exp_054 baseline)
| Phase                | Time   | %     |
|----------------------|--------|-------|
| Generation (rollout) | 74.55s | 56.9% |
| Policy training      | 36.12s | 27.5% |
| Logprob inference    | 9.43s  | 7.2%  |
| Prepare for gen      | 6.02s  | 4.6%  |
| Training prep        | 2.60s  | 2.0%  |
| **Total**            | **131.11s** | |

## Experiment Results

| Config | TP | MBS | LBS | Step Time | Training | Logprobs | Gen | Notes |
|--------|----|-----|-----|-----------|----------|----------|-----|-------|
| Baseline (exp_054) | 4 | 4 | 4 | **131.1s** | 36.1s | 9.4s | 74.5s | Previous run |
| Exp A | 1 | 8 | 8 | TBD | | | | |
| Exp B | 1 | 16 | 16 | TBD | | | | |
| Exp C | 1 | 32 | 16 | TBD | | | | |
| Exp D | 2 | 8 | 8 | TBD | | | | |

## Notes
- Generation (57%) is dominated by env await_results (72s/74.5s) — not tunable via MBS/LBS/TP
- Training improvement potential: ~36s → smaller; TP=1 eliminates TP comm overhead on 0.6B
- Logprob improvement potential: ~9.4s → smaller with larger LBS
- Max total gain from tuning these 3 params: ~45s → maybe 15-20s → ~20-25% total speedup
- Key hypothesis: TP=1 with DP=4 should be ~4x faster for training vs TP=4 with DP=1

## Final Recommendation
TBD
