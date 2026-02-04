import os
import pandas as pd
import numpy as np
from structural_perturbation import analytical_structural_perturbation_v2

def load_and_prepare_data(filepath):
    """
    Load a .inter file and prepare it for perturbation analysis.
    """
    df = pd.read_csv(filepath, sep='\t')
    
    # Rename columns to the format expected by the analysis function
    rename_map = {
        'user_id:token': 'user_id',
        'item_id:token': 'item_id',
        'rating:float': 'rating',
        'timestamp:float': 'timestamp',
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['user_id', 'item_id', 'rating']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {filepath}")
            
    return df

def main():
    # Directory with full training splits
    data_dir = './dataset_sampled_train_split_full/'
    
    # Get the list of all .inter files in the directory
    try:
        dataset_files = [f for f in os.listdir(data_dir) if f.endswith('.inter')]
        if not dataset_files:
            print(f"No .inter files found in directory: {data_dir}")
            return
    except FileNotFoundError:
        print(f"Directory does not exist: {data_dir}")
        return

    print(f"Found {len(dataset_files)} datasets to analyze.\n")

    # Configure parameters for the analysis
    p = 0.1              # Perturb 10% of the ratings
    n_iterations = 5     # Average over 5 runs
    n_components = 50    # Use 50 latent factors
    alpha = 0.7          # 70% value perturbation, 30% structural
    #time_sampling = True # Weight newer ratings in sampling
    
    results_list = []   

    # Iterate over each dataset file and run the analysis
    for filename in sorted(dataset_files):
        print(f"--- Analyzing: {filename} ---")
        filepath = os.path.join(data_dir, filename)
        
        try:
            # Load and prepare the data
            df = load_and_prepare_data(filepath)
            time_sampling = 'timestamp' in df.columns
            if not time_sampling:
                print("  -> Warning: 'timestamp' column not found. Using uniform sampling.")
            # Run structural perturbation analysis
            rmse, std_rmse, s_distance, std_s_distance, rmse_svd, std_rmse_svd, spectral_similarity, std_spectral_similarity = \
                analytical_structural_perturbation_v2(
                    df,
                    p=p,
                    n_iterations=n_iterations,
                    n_components=n_components,
                    alpha=alpha,
                    time_sampling=time_sampling
                )
            
            results_list.append({
                'dataset': filename,
                'Structural Perturbation RMSE': rmse,
                'Spectral Distance': s_distance,
                'Standard SVD RMSE': rmse_svd,
                'Spectral Similarity': spectral_similarity

            })

            # Display results for the current dataset
            print("=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"  Structural Perturbation RMSE: {rmse:.4f} ± {std_rmse:.4f}")
            print(f"  Spectral Distance:            {s_distance:.4f} ± {std_s_distance:.4f}")
            print(f"  Standard SVD RMSE:            {rmse_svd:.4f} ± {std_rmse_svd:.4f}")
            print(f"  Normalized RMSE:              {(rmse / rmse_svd if rmse_svd > 0 else float('inf')):.4f}")
            print(f"  Spectral Similarity:          {spectral_similarity:.4f} ± {std_spectral_similarity:.4f}")
            print()

        except Exception as e:
            print(f"\n  ERROR processing {filename}: {e}\n")
    
    # Save all results to a CSV file
    output_csv_file = './structural_perturbation/perturbation_results.csv'
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_csv_file, index=False, float_format='%.4f')
        print(f"Results successfully saved to '{output_csv_file}'")

    print("Interpretation:")
    print(f"  - Lower RMSE indicates more structural consistency")
    print(f"  - Lower spectral distance indicates more stable latent structure")
    print(f"  - Normalized RMSE allows comparison across different datasets")
    print("=" * 60)

if __name__ == "__main__":
    main()