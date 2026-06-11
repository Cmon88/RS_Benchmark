import subprocess
import time
import yaml
import pandas as pd
import os
import glob
import argparse

def consolidate_results(dataset_name, output_suffix="", results_dir="./latex"):
    """Consolidate all group results into final files"""

    valid_csv_files = glob.glob(os.path.join(results_dir, f"valid_{dataset_name}_group*.csv"))
    test_csv_files = glob.glob(os.path.join(results_dir, f"test_{dataset_name}_group*.csv"))

    all_valid_results = []
    all_test_results = []

    for csv_file in valid_csv_files:
        try:
            all_valid_results.append(pd.read_csv(csv_file))
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    for csv_file in test_csv_files:
        try:
            all_test_results.append(pd.read_csv(csv_file))
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    if all_valid_results:
        final_valid_df = pd.concat(all_valid_results, ignore_index=True)
        final_valid_file = os.path.join(results_dir, f"final_valid_{dataset_name}{output_suffix}.csv")
        final_valid_df.to_csv(final_valid_file, index=False)
        print(f"Consolidated validation results: {final_valid_file}")
        for f in valid_csv_files:
            os.remove(f)

    if all_test_results:
        final_test_df = pd.concat(all_test_results, ignore_index=True)
        final_test_file = os.path.join(results_dir, f"final_test_{dataset_name}{output_suffix}.csv")
        final_test_df.to_csv(final_test_file, index=False)
        print(f"Consolidated test results: {final_test_file}")
        for f in test_csv_files:
            os.remove(f)



# General parameters
# general_models = ['Pop', 'BPR', 'FISM', 'ItemKNN', 'CDAE', 'DMF', 'NeuMF', 'NNCF', 'ConvNCF', 'GCMC', 'MultiDAE', 'MultiVAE', 'SpectralCF', 'EASE', 'MacridVAE', 'NCEPLRec', 'NGCF', 'DGCF', 'ENMF', 'LightGCN', 'RecVAE', 'SGL', 'SimpleX', 'LDiffRec']
general_models = ['EASE', 'SGL', 'MultiDAE', 'DGCF', 'RecVAE', 'FISM','LightGCN', 'CDAE', 'DMF', 'NeuMF']
# general_models = ['SGL', 'MultiDAE', 'EASE', 'RecVAE', 'FISM','LightGCN', 'CDAE', 'DMF', 'NeuMF']
# general_models = ['BPR', 'LightGCN']  # For testing

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run general benchmark')
parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset name')
parser.add_argument('--config', type=str, default='test_dense.yaml', help='Config file path')
parser.add_argument('--data-path', type=str, default=None, help='Override data path for sampled datasets')
parser.add_argument('--results-dir', type=str, default='./latex', help='Directory to write result CSV files')
args = parser.parse_args()

dataset_name = args.dataset
config_file = args.config


# Load sampling configuration
def load_sampling_config(config_path='test_dense.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        print(f"Order config: {config.get('eval_args', {}).get('order', 'N/A')}")
        return config.get('sampling', {'enabled': False, 'n_samples': 1})

sampling_config = load_sampling_config(config_file)

# Split models into groups
group_size = 2
model_groups = [general_models[i:i + group_size] for i in range(0, len(general_models), group_size)]
times = []

print(f"Sampling configuration: {sampling_config}")
if sampling_config['enabled']:
    print(f"Running {sampling_config['n_samples']} samples per model")
    

results_dir = args.results_dir
os.makedirs(results_dir, exist_ok=True)

# Execute the benchmark for each group
models_done = 0
for idx, group in enumerate(model_groups):
    general_list = ",".join(group)
    output_suffix = f"_group{idx+1}"
    command = f"python run_recbole_group.py --model_list={general_list} --dataset={dataset_name} --config_files={config_file} --output_suffix={output_suffix} --results_dir={results_dir}"
    if args.data_path:
        command += f" --data_path={args.data_path}"

    print(f"\n[{models_done+1}-{min(models_done+len(group), len(general_models))}/{len(general_models)}] Running models: {group}")

    start_time = time.time()
    subprocess.run(command, shell=True, check=True)
    end_time = time.time()

    models_done += len(group)
    remaining = len(general_models) - models_done
    print(f"  Done in {end_time - start_time:.2f}s — {remaining} model(s) remaining")
    times.append(end_time - start_time)


# Consolidate all results at the end
print("\nConsolidating all results...")
consolidate_results(dataset_name, "_benchmark", results_dir=results_dir)

print("\nSummary")
print("=======")
print(f"General Models: {len(general_models)}")
print(f"Sampling: {'Enabled' if sampling_config['enabled'] else 'Disabled'}")
if sampling_config['enabled']:
    print(f"Samples per model: {sampling_config['n_samples']}")
    print(f"{sampling_config.get('target_users')} users, {sampling_config.get('target_items')} items")
    print(f"Total runs: {len(general_models) * sampling_config['n_samples']}")
    
for group, t in enumerate(times):
    print(f"Group {group+1} Time: {t:.2f} seconds")
print(f"Total Time: {sum(times):.2f} seconds")

print(f"\nFinal results saved in:")
print(f"- {results_dir}/final_valid_{dataset_name}_benchmark.csv")
print(f"- {results_dir}/final_test_{dataset_name}_benchmark.csv")