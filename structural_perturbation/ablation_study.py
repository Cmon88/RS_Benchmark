import os
import pandas as pd
import numpy as np
from structural_perturbation import analytical_structural_perturbation_v2

DATA_DIR = './dataset_sampled_train_split_full/'

def load_and_prepare_data(filepath):
    """Load and prepare the dataset."""
    df = pd.read_csv(filepath, sep='\t')
    rename_map = {
        'user_id:token': 'user_id', 'item_id:token': 'item_id',
        'rating:float': 'rating', 'timestamp:float': 'timestamp',
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def main():
    # 1. Define the search space (Grid)
    # Range of p (perturbation fraction): typically small values
    p_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    # Range of alpha (0.0 = structural only, 1.0 = value only)
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Fixed configurations
    n_iterations = 3     # Reduced so the grid search doesn't take too long
    n_components = 50

    # 2. Load list of files
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory {DATA_DIR} not found.")
        return
    
    dataset_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.inter')]
    dataset_files.sort()
    
    print(f"Datasets found: {len(dataset_files)}")
    print(f"Combinations to run: {len(p_values) * len(alpha_values)} per dataset")

    ablation_results = []

    # 3. Cache DataFrames to avoid reloading in each inner loop
    loaded_datasets = {}
    for filename in dataset_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = load_and_prepare_data(filepath)
            loaded_datasets[filename] = df
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    # 4. Run Grid Search
    # Iterate over p and alpha
    total_steps = len(p_values) * len(alpha_values)
    current_step = 0

    for p in p_values:
        for alpha in alpha_values:
            current_step += 1
            print(f"[{current_step}/{total_steps}] Processing Grid: p={p}, alpha={alpha} ...")
            
            for filename, df in loaded_datasets.items():
                try:
                    time_sampling = 'timestamp' in df.columns
                    
                    # Calculate metrics
                    rmse, _, s_distance, _, svd_rmse, _, _, _ = analytical_structural_perturbation_v2(
                        df,
                        p=p,
                        n_iterations=n_iterations,
                        n_components=n_components,
                        alpha=alpha,
                        time_sampling=time_sampling
                    )

                    ablation_results.append({
                        'dataset_name': filename.replace('.inter', ''),
                        'p': p,
                        'alpha': alpha,
                        'Structural Perturbation RMSE': rmse,
                        'Spectral Distance': s_distance,
                        'Standard SVD RMSE': svd_rmse
                    })
                    
                except Exception as e:
                    print(f"  Error in {filename}: {e}")

    # 5. Save results
    output_file = './structural_perturbation/ablation_results.csv'
    if ablation_results:
        results_df = pd.DataFrame(ablation_results)
        results_df.to_csv(output_file, index=False)
        print(f"\nAblation results saved to: {output_file}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()