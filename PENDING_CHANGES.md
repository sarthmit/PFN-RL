# Pending Changes in Gym Submodule

## Context
These changes were made locally on top of commit `1a4912e` (detached HEAD) of the
[NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) repo.
The parent repo is `git@github.com:sarthmit/PFN-RL.git` (branch `main`).

The overall theme is **migrating dataset identifiers from GitLab to HuggingFace**
and updating dataset statistics to reflect larger dataset sizes.

---

## Modified Files

### 1. `resources_servers/code_gen/data/livecodebench_v5_2024-07-01_2025-02-01_validation_metrics.json`
- Added `huggingface_identifier` block:
  ```json
  "huggingface_identifier": {
      "repo_id": "nvidia/nemotron-RL-coding-competitive_coding",
      "artifact_fpath": "validation.jsonl"
  }
  ```

### 2. `resources_servers/code_gen/data/opencodereasoning_filtered_25k_train_metrics.json`
- Replaced `gitlab_identifier` (dataset_name: `opencodereasoning_filtered`, version: `0.0.1`) with:
  ```json
  "gitlab_identifier": null,
  "huggingface_identifier": {
      "repo_id": "nvidia/nemotron-RL-coding-competitive_coding",
      "artifact_fpath": "opencodereasoning_filtered_25k_train.jsonl"
  }
  ```

### 3. `resources_servers/code_gen/data/train_metrics.json`
- Renamed dataset: `train` → `opencodereasoning_filtered_train`
- Updated `jsonl_fpath`: `train.jsonl` → `opencodereasoning_filtered_25k_train.jsonl`
- Replaced `gitlab_identifier` (dataset_name: `code_gen`) with HuggingFace identifier (same as above)
- Updated statistics: 5,000 → 23,971 examples; updated avg/min/max/std dev for word count
- Added new stat fields:
  ```json
  "hash_id": { "unique_count": 23971, "total_count": 23971 },
  "dataset": { "unique_count": 4, "total_count": 23971 },
  "source": { "unique_count": 8, "total_count": 23971 }
  ```

### 4. `resources_servers/math_with_judge/data/train_metrics.json`
- Updated `jsonl_fpath`: `train.jsonl` → `OpenMathReasoning_train.jsonl`
- Replaced `gitlab_identifier` (dataset_name: `math_open_math_reasoning`) with:
  ```json
  "gitlab_identifier": null,
  "huggingface_identifier": {
      "repo_id": "nvidia/Nemotron-RL-math-OpenMathReasoning",
      "artifact_fpath": "train.jsonl"
  }
  ```

### 5. `resources_servers/mcqa/data/train_metrics.json`
- Added `num_repeats: 1`
- Added HuggingFace identifier alongside existing GitLab identifier:
  ```json
  "huggingface_identifier": {
      "repo_id": "nvidia/Nemotron-RL-knowledge-mcqa",
      "artifact_fpath": null
  }
  ```
- Updated statistics: 27,568 → 617,020 examples; updated avg/min/max, added std dev fields
- Added new stat fields: `expected_answer`, `uuid`

---

## New (Untracked) Files

| File | Description |
|------|-------------|
| `resources_servers/code_gen/data/validation_metrics.json` | New validation metrics for code_gen |
| `resources_servers/code_gen/data/train_metrics_conflict.json` | Merge conflict artifact |
| `resources_servers/code_gen/data/opencodereasoning_filtered_25k_train_metrics_conflict.json` | Merge conflict artifact |
| `resources_servers/math_with_judge/data/validation_metrics.json` | New validation metrics for math_with_judge |
| `resources_servers/math_with_judge/data/train_metrics_conflict.json` | Merge conflict artifact |
| `resources_servers/mcqa/data/validation_metrics.json` | New validation metrics for mcqa |
| `resources_servers/mcqa/data/train_metrics_conflict.json` | Merge conflict artifact |
| `resources_servers/reasoning_gym/data/train_metrics.json` | New train metrics for reasoning_gym |
| `resources_servers/reasoning_gym/data/Nemotron-RL-ReasoningGym-v1_train_metrics.json` | New versioned train metrics for reasoning_gym |

> **Note:** The `*_conflict.json` files are likely merge conflict artifacts and
> should be reviewed before committing — they may be safe to delete if the
> corresponding non-conflict files already contain the resolved content.

---

## How to Push These Changes

Since the Gym submodule is in **detached HEAD** state and the `origin` remote points
to the upstream `https://github.com/NVIDIA-NeMo/Gym.git` (no push access), you need
to push to a fork. Steps:

```bash
cd /path/to/Gym

# Create a branch
git checkout -b <your-branch-name>

# Stage changes (decide whether to include conflict files)
git add resources_servers/

# Commit
git commit -m "chore: migrate dataset identifiers from GitLab to HuggingFace, update stats"

# Add your fork as a remote (if not already)
git remote add myfork git@github.com:<your-username>/Gym.git

# Push
git push myfork <your-branch-name>
```
