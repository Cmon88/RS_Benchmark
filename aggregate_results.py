"""
Aggregate benchmark results from latex/ folder.

For each final_test_*_benchmark.csv, select the best mrr@10, ndcg@10,
precision@10, recall@10 per sample (max across all models).
Outputs one row per (dataset, sample) combining all datasets into a single CSV.
"""

import argparse
import glob
import os
import pandas as pd
METRICS = ["mrr@10", "ndcg@10", "precision@10", "recall@10"]
STRUCTURAL = ["rmse", "sparsity", "effective_rank", "norm_rmse", "spectral_distance", "rmse_svd"]


def extract_dataset_name(filename: str) -> str:
    base = os.path.basename(filename)
    # final_test_<dataset>_benchmark.csv
    return base.removeprefix("final_test_").removesuffix("_benchmark.csv")


def aggregate_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    dataset_name = extract_dataset_name(path)

    # Best metric per sample (independent max across models)
    best = df.groupby("Sample")[METRICS].max()

    # Track which model achieved the best value for each metric
    rows = {}
    for sample, group in df.groupby("Sample"):
        rows[sample] = {f"best_model_{m}": group.loc[group[m].idxmax(), "Model"] for m in METRICS}
    best_models = pd.DataFrame.from_dict(rows, orient="index")
    best_models.index.name = "Sample"

    # Structural columns are constant per sample — take the first value
    structural = df.groupby("Sample")[STRUCTURAL].first()
    agg = best.join(best_models).join(structural).reset_index()
    agg.insert(0, "dataset", dataset_name)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Aggregate benchmark results from a results folder.")
    parser.add_argument(
        "folder",
        nargs="?",
        default=os.path.join(os.path.dirname(__file__), "latex"),
        help="Folder containing final_test_*_benchmark.csv files (default: latex/)",
    )
    args = parser.parse_args()

    folder = args.folder
    output_file = os.path.join(folder, "aggregated_best_metrics.csv")
    pattern = os.path.join(folder, "final_test_*_benchmark.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No files found matching {pattern}")
        return

    frames = []
    for path in files:
        print(f"Processing {os.path.basename(path)}")
        frames.append(aggregate_file(path))

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["dataset", "Sample"]).reset_index(drop=True)
    combined.to_csv(output_file, index=False)
    print(f"\nSaved {len(combined)} rows to {output_file}")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
