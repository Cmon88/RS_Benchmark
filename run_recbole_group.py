# @Time   : 2023/2/13
# @Author : Gaowei Zhang
# @Email  : zgw2022101006@ruc.edu.cn

### Modified from RecBole's run_recbole_group.py to add sampling support and save train splits, and others features ###

import argparse
from recbole.quick_start import run
from recbole.utils import list_to_latex, init_seed, get_model, get_trainer
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import yaml
import torch
import gc
import pandas as pd
import traceback
import os

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

def save_averages_to_csv(result_list, filename, subset_columns):
    """Save averages to CSV for later consolidation"""
    if result_list:
        # Filter only averages
        averages = [res for res in result_list if res.get('Sample') == 'Average']
        if averages:
            df = pd.DataFrame(averages)
            # Reorder columns: Model first, then metrics
            columns = ['Model'] + [col for col in subset_columns if col in df.columns]
            df = df[columns]
            df.to_csv(filename, index=False)
            print(f"Saved averages to {filename}")
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
        "--valid_latex", type=str, default="./latex/valid.tex", help="config files"
    )
    parser.add_argument(
        "--test_latex", type=str, default="./latex/test.tex", help="config files"
    )
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

    args, _ = parser.parse_known_args()

    model_list = args.model_list.strip().split(",")
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    dataset = args.dataset.strip()
    # Files for this group
    valid_file = f"./latex/valid_{dataset}{args.output_suffix}.tex"
    test_file = f"./latex/test_{dataset}{args.output_suffix}.tex"
    # CSV files for consolidation
    valid_csv_group = f"./latex/valid_{dataset}{args.output_suffix}.csv"
    test_csv_group = f"./latex/test_{dataset}{args.output_suffix}.csv"
    error_dir = './errors'
    os.makedirs(error_dir, exist_ok=True)

    # Load sampling configuration
    sampling_config = load_sampling_config(config_file_list)
    print(f"Sampling config: {sampling_config}")

    valid_result_list = []
    test_result_list = []

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
                    'data_path': './dataset_sampled/',
                }

            valid_res_dict = {"Model": model, "Sample": sample_idx+1}
            test_res_dict = {"Model": model, "Sample": sample_idx+1}
            
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
                # Initialize model
                model_obj = get_model(config['model'])(config, train_data.dataset).to(config['device'])
                # Initialize Trainer
                trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model_obj)
                # Train
                best_valid_score, best_valid_result = trainer.fit(
                    train_data, valid_data, saved=True, show_progress=True
                )
                # Evaluate: Ranking metrics
                test_result_ranking = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

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
                bigger_flag = config['valid_metric_bigger']
                subset_columns = list(best_valid_result.keys()) # Use validation metrics to sort columns
                for k in final_test_result.keys():
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
            if n_samples > 1:
                # Calculate average of metrics
                avg_valid = {"Model": model, "Sample": "Average"}
                avg_test = {"Model": model, "Sample": "Average"}

                for metric in subset_columns:
                    valid_values = [res[metric] for res in model_valid_results if metric in res]
                    test_values = [res[metric] for res in model_test_results if metric in res]

                    if valid_values:
                        avg_valid[metric] = sum(valid_values) / len(valid_values)
                    if test_values:
                        avg_test[metric] = sum(test_values) / len(test_values)

                valid_result_list.append(avg_valid)
                test_result_list.append(avg_test)
            else:
                # If there is only one sample, use it directly as "average"
                single_valid = model_valid_results[0].copy()
                single_valid['Sample'] = 'Average'
                single_test = model_test_results[0].copy()
                single_test['Sample'] = 'Average'
                valid_result_list.append(single_valid)
                test_result_list.append(single_test)

    # Save results in LaTeX and CSV
    if valid_result_list:
        # Filter only averages for LaTeX
        avg_valid_list = [res for res in valid_result_list if res.get('Sample') == 'Average']
        avg_test_list = [res for res in test_result_list if res.get('Sample') == 'Average']
        if avg_valid_list:
            # avoid rmse as it was not used in validation
            valid_keys = set().union(*(d.keys() for d in avg_valid_list))
            valid_subset = [col for col in subset_columns if col in valid_keys]

            df_valid, tex_valid = list_to_latex(
                convert_list=avg_valid_list,
                bigger_flag=bigger_flag,
                subset_columns=valid_subset,
            )
            test_keys = set().union(*(d.keys() for d in avg_test_list))
            test_subset = [col for col in subset_columns if col in test_keys]
            df_test, tex_test = list_to_latex(
                convert_list=avg_test_list,
                bigger_flag=bigger_flag,
                subset_columns=test_subset,
            )

            with open(valid_file, "w") as f:
                f.write(tex_valid)
            with open(test_file, "w") as f:
                f.write(tex_test)
            # Also save in CSV for consolidation
            save_averages_to_csv(valid_result_list, valid_csv_group, subset_columns)
            save_averages_to_csv(test_result_list, test_csv_group, subset_columns)