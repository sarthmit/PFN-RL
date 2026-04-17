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

import re
from typing import Any, TypedDict

import ray
import reasoning_gym
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)


class ReasoningGymMetadata(TypedDict):
    dataset_name: str
    entry: dict[str, Any]


_ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: str) -> str | None:
    matches = _ANSWER_TAG_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()
    return None


@ray.remote
class ReasoningGymEnvironment(EnvironmentInterface[ReasoningGymMetadata]):
    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self._score_fns: dict[str, Any] = {}

    def _get_score_fn(self, dataset_name: str):
        if dataset_name not in self._score_fns:
            self._score_fns[dataset_name] = reasoning_gym.get_score_answer_fn(
                dataset_name
            )
        return self._score_fns[dataset_name]

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ReasoningGymMetadata],
    ) -> EnvironmentReturn[ReasoningGymMetadata]:
        observations = []
        rewards = []
        terminateds = []
        all_stop_strings: list[list[str] | None] = []
        all_metadata = []
        all_answers: list[str | None] = []

        for msg_log, meta in zip(message_log_batch, metadata):
            response = ""
            if msg_log and msg_log[-1]["role"] == "assistant":
                response = msg_log[-1]["content"].strip()

            answer = extract_answer(response) or response
            all_answers.append(answer)

            score_fn = self._get_score_fn(meta["dataset_name"])
            try:
                score = score_fn(answer=answer, entry=meta["entry"])
            except Exception:
                score = 0.0

            reward = float(score) if score is not None else 0.0
            rewards.append(reward)
            terminateds.append(True)
            observations.append({"role": "environment", "content": ""})
            all_stop_strings.append(None)
            all_metadata.append(None)

        return EnvironmentReturn(
            observations=observations,
            metadata=all_metadata,
            next_stop_strings=all_stop_strings,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=all_answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        final_rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        )
        accuracy = (
            (final_rewards > 0.5).float().mean().item()
            if len(final_rewards) > 0
            else 0.0
        )

        metrics: dict[str, float] = {"accuracy": accuracy}

        # Per-env breakdown: keyed by reasoning_gym env name stored in
        # extra_env_info (sample-level task_name is uniform = "reasoning_gym").
        env_infos = batch.get("extra_env_info", [])
        if env_infos:
            per_env: dict[str, list[float]] = {}
            for i, info in enumerate(env_infos):
                if i >= len(final_rewards):
                    break
                name = info.get("dataset_name") if isinstance(info, dict) else None
                if not name:
                    continue
                per_env.setdefault(name, []).append(final_rewards[i].item())
            for name, rews in per_env.items():
                metrics[f"accuracy/{name}"] = (
                    sum(1 for r in rews if r > 0.5) / len(rews)
                )

        return batch, metrics
