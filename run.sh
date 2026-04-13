#!/usr/bin/env bash

UV_CACHE_DIR=.cache HF_HOME=.cache VLLM_CACHE_DIR=.cache ./examples/nemo_gym/launch_rl_training.sh --model Qwen/Qwen3-4B-Instruct-2507 --math 0.25 --code 0.25 --mcqa 0.25 --reasoning 0.25 --mbs 1 --lbs 8 --dtensor-tp 1