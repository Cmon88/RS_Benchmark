# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RS_Benchmark studies the **structural stability of recommender system datasets** using [RecBole](https://recbole.io/). The workflow generates controlled subsamples of a dataset, analyzes their graph properties, trains a battery of RS models on each subsample, and compares performance across samples to quantify structural complexity.

## Setup

Python 3.12 required.

```bash
pip install -r requirements.txt
```

Datasets must be placed in `dataset/<name>/<name>.inter` (tab-separated, RecBole format). Download from [RecBole's Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). For Time Ordering (TO) experiments, the `.inter` file must include a `timestamp:float` column.

## Typical Workflow

### 1. Configure — edit a YAML file (e.g., `test.yaml`)

Key sections:
- `eval_args.order`: `RO` (Random Order) or `TO` (Time Ordering — requires `load_col: null` in the YAML)
- `sampling.enabled`, `sampling.n_samples`, `sampling.target_interactions`

Use `test.yaml` for RO runs, `test_TO.yaml` for TO runs.

### 2. Generate subsamples

```bash
python sample_dense.py --dataset <dataset_name> --config test.yaml
```

Output goes to `dataset_sampled/<dataset_name>_sample{N}/`.

### 3. Verify subsample quality

```bash
python diagnostic.py --dataset <dataset_name> --n_samples 3
```

Compares original, subsamples, and `ml-100k` as a fixed reference.

### 4. Run the benchmark

```bash
python general.py --dataset <dataset_name> --config test.yaml
```

This batches all models in `general_models` into groups of 2 and calls `run_recbole_group.py` for each group. Results land in `latex/`.

### 5. View results

Final outputs are in `latex/`:
- `final_test_<dataset>_benchmark.csv` — raw metrics per model
- `final_test_<dataset>_benchmark.tex` — LaTeX table with best values bolded

## Architecture

### Entry points and their roles

| Script | Role |
|---|---|
| `sample_dense.py` | Creates `N` balanced subsamples targeting a fixed interaction count via iterative k-core-style filtering |
| `diagnostic.py` | Prints density/user/item stats for the original dataset and all samples |
| `general.py` | Orchestrator: splits `general_models` into groups of 2, spawns `run_recbole_group.py` per group, then consolidates CSVs into final tables |
| `run_recbole_group.py` | Modified RecBole runner — trains each model on each sample, evaluates ranking metrics + RMSE (via `test_rmse.yaml`), saves averaged results per group to `latex/` |

### Two-metric evaluation in `run_recbole_group.py`

Each model is evaluated twice:
1. **Ranking metrics** (MRR, NDCG, Precision, Recall) using the primary config
2. **RMSE** using a separate `Config` object built from `test_rmse.yaml` overlaid on the same trained model weights (no re-training)

Results are merged before saving.

### Structural perturbation module (`structural_perturbation/`)

A separate analytical module (not called by `general.py`) that measures how SVD factorization changes under controlled rating matrix perturbations. Entry point: `run_perturbation_analysis.py`, which reads training splits from `dataset_sampled_train_split_full/`. The core math lives in `structural_perturbation.py`.

### Sampling algorithm (`sample_dense.py`)

1. Remove saturated users (interacted with ≥95% of all items)
2. Remove cold users (fewer than `min_items_per_user` interactions)
3. Estimate how many users are needed to hit `target_interactions` (2× safety factor)
4. Random-sample users, then apply light k-core filtering on items (min 2 interactions per item)
5. Inject extra items if any user has interacted with all available items (needed for negative sampling)
6. Trim or warn to hit the exact target count

Each sample uses `seed = base_seed + sample_index` for reproducibility.

## Key Configuration Details

- **`group_size = 2`** in `general.py` controls how many models run per subprocess. Increase cautiously — each group runs sequentially within its subprocess and can OOM on GPUs.
- **Model list** is `general_models` in `general.py`. Comment/uncomment to test subsets.
- **Train splits** are saved to `train_splits/` after the first run per dataset/sample (skipped on subsequent runs).
- **Errors** per model/sample are written to `errors/error_<model>_<dataset>_sample<N>.log` and do not abort the run.
- **Saved model checkpoints** are deleted immediately after each run to conserve disk space.
