#!/usr/bin/env bash

UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 --mbs 2 --lbs 8 --dtensor-tp 1
# UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 --mbs 4 --lbs 16 --dtensor-tp 1
# UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 --mbs 8 --lbs 8 --dtensor-tp 1

# UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 --mbs 4 --lbs 8 --dtensor-tp 2
# UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 --mbs 4 --lbs 16 --dtensor-tp 2
# UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-0.6B --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 --mbs 8 --lbs 8 --dtensor-tp 2