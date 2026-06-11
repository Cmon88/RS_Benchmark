"""
Generate train splits for 50k or 100k sampled datasets using RecBole's data pipeline.
Saves one <dataset>_train.inter file per dataset into train_splits_<size>k/.

Usage (from RS_Benchmark/):
    python generate_train_splits.py --size 100
    python generate_train_splits.py --size 50
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed


def save_train_split(dataset_obj, dataset_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{dataset_name}_train.inter')
    if os.path.exists(file_path):
        print(f"  Already exists, skipping: {file_path}")
        return

    df = pd.DataFrame()
    for field in dataset_obj.inter_feat:
        ids = dataset_obj.inter_feat[field]
        if field in dataset_obj.field2id_token:
            try:
                tokens = dataset_obj.id2token(field, ids)
                df[field] = tokens
            except Exception:
                df[field] = ids.numpy() if hasattr(ids, 'numpy') else ids
        else:
            df[field] = ids.numpy() if hasattr(ids, 'numpy') else ids

    df.to_csv(file_path, sep='\t', index=False)
    print(f"  Saved: {file_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, choices=[50, 100], required=True,
                        help='Dataset size: 50 (for 50k) or 100 (for 100k)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override data path (default: ../dataset_sampled_<size>k/)')
    parser.add_argument('--dataset', nargs='+', default=None,
                        help='Base dataset name(s) to process (e.g. epinions). '
                             'If omitted, all datasets are processed.')
    args = parser.parse_args()

    data_path = args.data_path or f'../dataset_sampled_{args.size}k/'
    save_dir = f'./train_splits_{args.size}k/'

    if not os.path.isdir(data_path):
        print(f"Error: data path not found: {data_path}")
        sys.exit(1)

    dataset_names = sorted(os.listdir(data_path))
    # Only keep directories that contain a matching .inter file
    valid = []
    for name in dataset_names:
        inter = os.path.join(data_path, name, f'{name}.inter')
        if os.path.isfile(inter):
            valid.append(name)
    if args.dataset:
        requested = set(args.dataset)
        valid = [n for n in valid if any(n.startswith(ds + '_sample') for ds in requested)]
        print(f"Filtering to datasets: {args.dataset}")
    print(f"Found {len(valid)} datasets in {data_path}")
    print(f"Saving train splits to {save_dir}\n")

    errors = []
    for i, dataset_name in enumerate(valid, 1):
        print(f"[{i}/{len(valid)}] {dataset_name}")
        try:
            inter_path = os.path.join(data_path, dataset_name, f'{dataset_name}.inter')
            all_cols = pd.read_csv(inter_path, sep='\t', nrows=0).columns.tolist()
            standard_fields = {'user_id', 'item_id', 'rating', 'timestamp'}
            load_cols = [c.split(':')[0] for c in all_cols if c.split(':')[0] in standard_fields]
            config = Config(
                model='BPR',
                dataset=dataset_name,
                config_dict={
                    'data_path': data_path,
                    'eval_args': {
                        'group_by': 'user',
                        'order': 'RO',
                        'split': {'RS': [0.8, 0.1, 0.1]},
                        'mode': {'valid': 'uni100', 'test': 'uni100'},
                    },
                    'load_col': {'inter': load_cols},
                    'neg_sampling': None,
                    'train_neg_sample_args': None,
                },
            )
            init_seed(config['seed'], config['reproducibility'])
            dataset_obj = create_dataset(config)
            train_data, _, _ = data_preparation(config, dataset_obj)
            save_train_split(train_data.dataset, dataset_name, save_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((dataset_name, str(e)))

    print(f"\nDone. {len(valid) - len(errors)}/{len(valid)} succeeded.")
    if errors:
        print("Failures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")


if __name__ == '__main__':
    main()
