# @Time   : 2023/2/13
# @Author : Gaowei Zhang
# @Email  : zgw2022101006@ruc.edu.cn

### Modified from RecBole's run_recbole_group.py to add sampling support and save train splits, and others features ###

import argparse
import sys
import numpy as np
from recbole.quick_start import run
from recbole.utils import init_seed, get_model, get_trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import yaml
import torch
import gc
import pandas as pd
import traceback
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structural_perturbation'))
from structural_perturbation import (
    sparsity, effective_rank,
    analytical_structural_perturbation_v2, convert_to_sparse_matrix,
)

def save_train_split(dataset_obj, dataset_name):
    """
    Saves the training split as a .inter file.
    Only saves it if it doesn't exist previously, assuming consistency via seed.
    """
    save_dir = 'train_splits'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{dataset_name}_train.inter')
    
    # If it already exists, we assume it is the same split (thanks to init_seed) and skip
    if os.path.exists(file_path):
        return

    try:
        df = pd.DataFrame()
        # RecBole stores columns in inter_feat
        for field in dataset_obj.inter_feat:
            ids = dataset_obj.inter_feat[field]
            
            # Try to recover original tokens (e.g., original user_id string instead of internal integer)
            # This makes the file portable and readable
            if field in dataset_obj.field2id_token:
                try:
                    tokens = dataset_obj.id2token(field, ids)
                    df[field] = tokens
                except:
                    # Fallback to internal IDs if it fails
                    if hasattr(ids, 'numpy'):
                        df[field] = ids.numpy()
                    else:
                        df[field] = ids
            else:
                if hasattr(ids, 'numpy'):
                    df[field] = ids.numpy()
                else:
                    df[field] = ids
        
        df.to_csv(file_path, sep='\t', index=False)
        print(f"Saved train split to {file_path}")
    except Exception as e:
        print(f"Error saving train split: {e}")

def load_sampling_config(config_files):
    """Load sampling configuration from YAML files"""
    sampling_config = {
        'enabled': False,
        'n_samples': 1,
        'interactions_per_sample': 100000,
        'random_seed': 42
    }
    if config_files:
        for config_file in config_files:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if 'sampling' in config_data:
                    sampling_config.update(config_data['sampling'])

    return sampling_config


def load_eval_args(config_files):
    """Load eval_args dict from YAML config files (last file wins per key)."""
    eval_args = {}
    if config_files:
        for config_file in config_files:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
                if 'eval_args' in config_data:
                    eval_args.update(config_data['eval_args'])
    return eval_args


def dataset_has_timestamp(dataset_name, data_path):
    """Return True if the dataset .inter file has a timestamp column, False if not, None if file not found."""
    inter_path = os.path.join(data_path, dataset_name, f'{dataset_name}.inter')
    if not os.path.exists(inter_path):
        return None
    with open(inter_path, 'r') as f:
        header = f.readline().strip()
    columns = [col.split(':')[0] for col in header.split('\t')]
    return 'timestamp' in columns

def extract_train_df(dataset_obj):
    """Convert a RecBole training dataset to a plain DataFrame for structural analysis."""
    inter = dataset_obj.inter_feat

    def to_numpy(v):
        return v.numpy() if hasattr(v, 'numpy') else np.array(v)

    df = pd.DataFrame({
        'user_id': to_numpy(inter[dataset_obj.uid_field]),
        'item_id': to_numpy(inter[dataset_obj.iid_field]),
    })

    label = dataset_obj.label_field
    if label and label in inter:
        df['rating'] = to_numpy(inter[label]).astype(float)
    else:
        df['rating'] = 1.0

    if 'timestamp' in inter:
        df['timestamp'] = to_numpy(inter['timestamp']).astype(float)

    return df


def compute_structural_metrics(dataset_obj, n_components=50):
    """Return sparsity, effective_rank, norm_rmse, spectral_distance for a training split."""
    try:
        train_df = extract_train_df(dataset_obj)
        R, _, _ = convert_to_sparse_matrix(train_df)
        R = R.astype('float32')

        sp = float(sparsity(R))
        er = float(effective_rank(R, k=n_components))

        avg_rmse, _, avg_s_dist, _, avg_rmse_svd, _, _, _ = analytical_structural_perturbation_v2(
            train_df, p=0.1, n_iterations=1, n_components=n_components, alpha=0.7
        )
        norm_rmse = float(avg_rmse / avg_rmse_svd) if avg_rmse_svd > 0 else 0.0

        return {
            'sparsity': sp,
            'effective_rank': er,
            'norm_rmse': norm_rmse,
            'spectral_distance': float(avg_s_dist),
            'rmse': avg_rmse,
            'rmse_svd': avg_rmse_svd,
        }
    except Exception as e:
        print(f"  Warning: structural metrics failed: {e}")
        return {
            'sparsity': float('nan'),
            'effective_rank': float('nan'),
            'norm_rmse': float('nan'),
            'spectral_distance': float('nan'),
            'rmse': float('nan'),
            'rmse_svd': float('nan'),
        }


def save_results_to_csv(result_list, filename, subset_columns):
    """Save all sample results to CSV for later consolidation"""
    if result_list:
        df = pd.DataFrame(result_list)
        columns = ['Dataset', 'Model', 'Sample'] + [col for col in subset_columns if col in df.columns]
        columns = [col for col in columns if col in df.columns]
        df = df[columns]
        df.to_csv(filename, index=False)
        print(f"Saved results to {filename}")
        return df
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_list", "-m", type=str, default="BPR", help="name of models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    parser.add_argument(
        "--output_suffix", type=str, default="", help="suffix for output files"
    )
    parser.add_argument(
        "--data_path", type=str, default=None, help="override data path for sampled datasets"
    )
    parser.add_argument(
        "--results_dir", type=str, default="./latex", help="directory to write result CSV files"
    )

    args, _ = parser.parse_known_args()

    model_list = args.model_list.strip().split(",")
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    dataset = args.dataset.strip()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    # CSV files for consolidation
    valid_csv_group = os.path.join(results_dir, f"valid_{dataset}{args.output_suffix}.csv")
    test_csv_group = os.path.join(results_dir, f"test_{dataset}{args.output_suffix}.csv")
    error_dir = './errors'
    os.makedirs(error_dir, exist_ok=True)

    # Load sampling configuration
    sampling_config = load_sampling_config(config_file_list)
    print(f"Sampling config: {sampling_config}")

    # Detect if TO ordering is configured but the dataset lacks a timestamp column;
    # if so, fall back to RO to avoid per-sample failures downstream.
    eval_args_override = None
    _eval_args = load_eval_args(config_file_list)
    if _eval_args.get('order') == 'TO':
        # Check the first sample path (or the original dataset path as fallback)
        if sampling_config['enabled']:
            ts_data_path = args.data_path if args.data_path else '../dataset_sampled/'
            ts_dataset_name = f'{dataset}_sample1'
        else:
            ts_data_path = args.data_path if args.data_path else '../dataset/'
            ts_dataset_name = dataset
        has_ts = dataset_has_timestamp(ts_dataset_name, ts_data_path)
        if has_ts is False:
            print(f"WARNING: Dataset '{dataset}' has no timestamp column — falling back to RO ordering.")
            eval_args_override = dict(_eval_args, order='RO')
        elif has_ts is None:
            print(f"WARNING: Could not verify timestamp column for '{dataset}' at '{ts_data_path}' — falling back to RO ordering.")
            eval_args_override = dict(_eval_args, order='RO')

    valid_result_list = []
    test_result_list = []
    structural_metrics_cache = {}  # keyed by sample_idx — same dataset, same metrics
    subset_columns = []

    run_times = len(model_list)

    for idx in range(run_times):
        model = model_list[idx]

        print(f"[{idx+1}/{run_times}] Training {model}...")

        # Clean memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        model_valid_results = []
        model_test_results = []

        # Run multiple samples if sampling is enabled
        n_samples = sampling_config['n_samples'] if sampling_config['enabled'] else 1
        
        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx+1}/{n_samples}")

            # Specific configuration for this sample
            sample_config_dict = {}
            dataset_to_use = dataset        # Default to using original dataset

            if sampling_config['enabled']:
                dataset_to_use = f'{dataset}_sample{sample_idx+1}'
                sample_config_dict = {
                    'data_path': args.data_path if args.data_path else '../dataset_sampled/',
                }

            # Apply eval_args override (e.g. TO→RO fallback when timestamp is absent)
            if eval_args_override:
                sample_config_dict['eval_args'] = eval_args_override

            valid_res_dict = {"Dataset": dataset_to_use, "Model": model, "Sample": sample_idx+1}
            test_res_dict = {"Dataset": dataset_to_use, "Model": model, "Sample": sample_idx+1}
            
            # Run the model
            try:
                # Load configuration
                config = Config(
                    model=model,
                    dataset=dataset_to_use,
                    config_file_list=config_file_list,
                    config_dict=sample_config_dict
                )
                init_seed(config['seed'], config['reproducibility'])
                # Prepare data
                dataset_obj = create_dataset(config)
                train_data, valid_data, test_data = data_preparation(config, dataset_obj)
                # Save training split (will run only the first time per dataset/sample)
                save_train_split(train_data.dataset, dataset_to_use)
                # Compute structural metrics once per sample (same dataset → same result)
                if sample_idx not in structural_metrics_cache:
                    print(f"  Computing structural metrics for sample {sample_idx+1}...")
                    structural_metrics_cache[sample_idx] = compute_structural_metrics(train_data.dataset)
                # Initialize model
                model_obj = get_model(config['model'])(config, train_data.dataset).to(config['device'])
                # Initialize Trainer
                trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model_obj)
                # Train
                best_valid_score, best_valid_result = trainer.fit(
                    train_data, valid_data, saved=True, show_progress=True
                )
                # Evaluate: Ranking metrics
                # Some models (e.g. EASE) are closed-form and never save a .pth file
                _saved_file = getattr(trainer, 'saved_model_file', None)
                _load_best = bool(_saved_file and os.path.exists(_saved_file))
                test_result_ranking = trainer.evaluate(test_data, load_best_model=_load_best, show_progress=True)

                # Evaluate: Value Metrics
                with open('test_rmse.yaml', 'r') as f:
                    rmse_yaml_config = yaml.safe_load(f)
                # Combine current sampling config with RMSE config
                rmse_config_dict = sample_config_dict.copy()
                rmse_config_dict.update(rmse_yaml_config)

                # Copy original evaluation arguments to maintain split
                if 'eval_args' not in rmse_config_dict:
                    rmse_config_dict['eval_args'] = config['eval_args'].copy()
                
                rmse_config_dict['eval_args'].update({
                    'group_by': None,      # Do not group by user
                    'order': 'RO',         # Random order (doesn't matter in labeled)
                    'mode': 'uni100'       # Point-wise mode (necessary for regression/RMSE)
                })
                rmse_config_dict['metrics'] = ['RMSE']
                
                # Create NEW Config and Dataset objects
                config_rmse = Config(
                    model=model,
                    dataset=dataset_to_use,
                    config_file_list=config_file_list,
                    config_dict=rmse_config_dict
                )
                # Reset seed
                init_seed(config_rmse['seed'], config_rmse['reproducibility'])
                # Create clean dataset and dataloaders
                dataset_rmse = create_dataset(config_rmse)
                _, _, test_data_rmse = data_preparation(config_rmse, dataset_rmse)
                # New Trainer with ALREADY TRAINED model
                trainer_rmse = get_trainer(config_rmse['MODEL_TYPE'], config_rmse['model'])(config_rmse, model_obj)
                # Use NEW dataloader (test_data_rmse)
                test_result_rmse = trainer_rmse.evaluate(test_data_rmse, load_best_model=False, show_progress=True)
                print(f"    RMSE: Result: {test_result_rmse}")
                # Combine test results
                final_test_result = test_result_ranking.copy()
                final_test_result.update(test_result_rmse)
                valid_res_dict.update(best_valid_result)
                test_res_dict.update(final_test_result)
                struct = structural_metrics_cache[sample_idx]
                valid_res_dict.update(struct)
                test_res_dict.update(struct)
                bigger_flag = config['valid_metric_bigger']
                subset_columns = list(best_valid_result.keys())
                for k in final_test_result.keys():
                    if k not in subset_columns:
                        subset_columns.append(k)
                for k in structural_metrics_cache[sample_idx].keys():
                    if k not in subset_columns:
                        subset_columns.append(k)

                model_valid_results.append(valid_res_dict)
                model_test_results.append(test_res_dict)
                
                # Delete saved model to free up space
                if hasattr(trainer, 'saved_model_file') and trainer.saved_model_file:
                    if os.path.exists(trainer.saved_model_file):
                        try:
                            os.remove(trainer.saved_model_file)
                            print(f"Deleted saved model file: {trainer.saved_model_file}")
                        except OSError as e:
                            print(f"Error deleting model file: {e}")

                # Delete objects to free up memory
                del model_obj
                del trainer
                del trainer_rmse
                del dataset_obj
                del dataset_rmse
                del train_data
                del valid_data
                del test_data
                del test_data_rmse

            except Exception as e:
                print(f"ERROR with model {model}, sample {sample_idx+1}: {e}")
                traceback.print_exc()
                error_filename = f"error_{model}_{dataset}_sample{sample_idx+1}.log"
                error_path = os.path.join(error_dir, error_filename)
                
                with open(error_path, 'w') as f:
                    f.write(f"Model: {model}\n")
                    f.write(f"Dataset: {dataset}\n")
                    f.write(f"Sample: {sample_idx+1}\n")
                    f.write(f"Error Message: {str(e)}\n")
                    f.write("-" * 60 + "\n")
                    f.write("Full Traceback:\n")
                    traceback.print_exc(file=f)
                continue

            # Clean memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        # Calculate averages if sampling was used
        if len(model_valid_results)>0:
            valid_result_list.extend(model_valid_results)
            test_result_list.extend(model_test_results)

    # Save results to CSV
    if valid_result_list:
        save_results_to_csv(valid_result_list, valid_csv_group, subset_columns)
        save_results_to_csv(test_result_list, test_csv_group, subset_columns)