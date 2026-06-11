import argparse
import os
import pandas as pd
import numpy as np
from structural_perturbation import analytical_structural_perturbation_v2


BEER_ADVOCATE_SUBRATINGCOLS = {'appearance', 'aroma', 'palate', 'taste', 'overall'}


def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, sep='\t')
    rename_map = {
        'user_id:token': 'user_id', 'item_id:token': 'item_id',
        'rating:float': 'rating', 'timestamp:float': 'timestamp',
    }
    df.rename(columns=rename_map, inplace=True)

    if 'rating' not in df.columns:
        sub_cols = BEER_ADVOCATE_SUBRATINGCOLS & set(df.columns)
        if sub_cols:
            df['rating'] = df[list(sub_cols)].mean(axis=1)
        else:
            df['rating'] = 1.0

    # Normalize ratings to [0, 1] so datasets with different scales are comparable
    r_min, r_max = df['rating'].min(), df['rating'].max()
    if r_max > r_min:
        df['rating'] = (df['rating'] - r_min) / (r_max - r_min)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, choices=[50, 100], default=50,
                        help='Dataset size: 50 (50k) or 100 (100k). Default: 50')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override the train splits directory')
    parser.add_argument('--dataset', nargs='+', default=None,
                        help='Base dataset name(s) to process (e.g. netflix). '
                             'If omitted, all datasets are processed.')
    args = parser.parse_args()

    data_dir = args.data_dir or f'./train_splits_{args.size}k/'
    output_file = f'./structural_perturbation/ablation_results_{args.size}k.csv'

    p_values = [0.05, 0.1, 0.2, 0.25]
    alpha_values = [0.0, 0.3, 0.7, 1.0]
    n_iterations = 3
    n_components = 50

    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return

    dataset_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.inter'))
    if args.dataset:
        requested = set(args.dataset)
        dataset_files = [
            f for f in dataset_files
            if any(f.startswith(ds + '_sample') for ds in requested)
        ]
        print(f"Filtering to datasets: {args.dataset}")
    print(f"Datasets found: {len(dataset_files)}")
    print(f"Combinations to run: {len(p_values) * len(alpha_values)} per dataset")

    loaded_datasets = {}
    for filename in dataset_files:
        try:
            df = load_and_prepare_data(os.path.join(data_dir, filename))
            loaded_datasets[filename] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    ablation_results = []
    total_steps = len(p_values) * len(alpha_values)
    current_step = 0

    for p in p_values:
        for alpha in alpha_values:
            current_step += 1
            print(f"[{current_step}/{total_steps}] p={p}, alpha={alpha} ...")

            for filename, df in loaded_datasets.items():
                try:
                    time_sampling = 'timestamp' in df.columns
                    rmse, _, s_distance, _, svd_rmse, _, _, _ = analytical_structural_perturbation_v2(
                        df,
                        p=p,
                        n_iterations=n_iterations,
                        n_components=n_components,
                        alpha=alpha,
                        time_sampling=time_sampling,
                    )
                    ablation_results.append({
                        'dataset_name': filename.replace('.inter', ''),
                        'size': f'{args.size}k',
                        'p': p,
                        'alpha': alpha,
                        'Structural Perturbation RMSE': rmse,
                        'Spectral Distance': s_distance,
                        'Standard SVD RMSE': svd_rmse,
                    })
                except Exception as e:
                    print(f"  Error in {filename}: {e}")

    if ablation_results:
        new_df = pd.DataFrame(ablation_results)
        if args.dataset and os.path.exists(output_file):
            existing = pd.read_csv(output_file)
            requested = set(args.dataset)
            keep = existing["dataset_name"].apply(
                lambda n: not any(n.startswith(ds + "_sample") for ds in requested)
            )
            new_df = pd.concat([existing[keep], new_df], ignore_index=True)
        new_df.to_csv(output_file, index=False)
        print(f"\nAblation results saved to: {output_file}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
