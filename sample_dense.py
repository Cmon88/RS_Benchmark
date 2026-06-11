import pandas as pd
import os
import yaml
import numpy as np
import argparse

def create_interaction_based_subsamples(dataset_name, target_interactions, n_samples, base_seed=42,
                                      min_items_per_user=5, output_dir='../dataset_sampled'):
    """
    Creates subsamples aiming to reach 'target_interactions' while maintaining consistency.
    Uses an iterative approach to find the correct number of users.
    """
    
    original_path = f'../dataset/{dataset_name}/{dataset_name}.inter'
    
    if not os.path.exists(original_path):
        print(f"Original dataset not found: {original_path}")
        return
    
    df = pd.read_csv(original_path, sep='\t')

    token_cols = [c for c in df.columns if c.endswith(':token')]
    if len(token_cols) < 2:
        print(f"Could not detect user/item columns (need at least 2 *:token columns, found: {token_cols})")
        return
    user_col, item_col = token_cols[0], token_cols[1]
    if user_col != 'user_id:token' or item_col != 'item_id:token':
        print(f"  Detected columns: user='{user_col}', item='{item_col}'")

    # Original statistics for initial estimation
    n_users_orig = df[user_col].nunique()
    n_items_orig = df[item_col].nunique()
    orig_density = len(df) / (n_users_orig * n_items_orig)
    avg_inter_per_user = len(df) / n_users_orig
    
    print(f"=== Creating Interaction-Targeted Subsamples for {dataset_name} ===")
    print(f"Target Interactions: {target_interactions}")
    print(f"Original Density: {orig_density:.4%}")
    print(f"Avg interactions/user (original): {avg_inter_per_user:.2f}")

    sampled_base_dir = output_dir
    os.makedirs(sampled_base_dir, exist_ok=True)
    
    for i in range(n_samples):
        print(f"\n--- Creating sample {i+1}/{n_samples} ---")
        seed = base_seed + i
        np.random.seed(seed)

        df_work = df.copy()

        # 0. Saturation Filter
        total_items_global = df[item_col].nunique()
        items_per_user = df_work.groupby(user_col)[item_col].nunique()
        saturated_users_ids = items_per_user[items_per_user >= (total_items_global * 0.95)].index
        if len(saturated_users_ids) > 0:
            print(f"  Excluding {len(saturated_users_ids)} saturated users.")
            df_work = df_work[~df_work[user_col].isin(saturated_users_ids)]

        # 1. Filter "cold" users
        user_counts = df_work[user_col].value_counts()
        valid_candidates = user_counts[user_counts >= min_items_per_user].index
        df_candidates = df_work[df_work[user_col].isin(valid_candidates)]

        # Recalculate average on valid candidates for better estimation
        avg_valid = len(df_candidates) / df_candidates[user_col].nunique()

        # 2. Initial estimation of users needed
        # We add a safety factor (1.1) because we will lose interactions when filtering items later
        estimated_users = int((target_interactions / avg_valid) * 2.0)

        # Safety limit
        max_users = df_candidates[user_col].nunique()
        n_users_to_sample = min(estimated_users, max_users)

        print(f"Estimating we need ~{n_users_to_sample} users to reach target...")

        # 3. User Selection (Random or Top-Active? The paper uses Random on filtered users)
        # We will use Random to avoid extreme popularity bias, but on valid users
        selected_users = np.random.choice(valid_candidates, n_users_to_sample, replace=False)
        df_sample = df[df[user_col].isin(selected_users)].copy()

        # 4. Item Cleaning (Paper Guarantee: "Union of 2 items per user")
        # This ensures we don't have items with a single lost interaction
        # For large datasets, we can relax this and simply remove items with < X global interactions in the sample

        # Simple iterative filter to clean the graph (light k-core decomposition)
        # We remove items that were left with very few interactions in this subgroup
        min_item_support = 2
        item_counts = df_sample[item_col].value_counts()
        valid_items = item_counts[item_counts >= min_item_support].index
        df_sample = df_sample[df_sample[item_col].isin(valid_items)]

        # Check if any user has interacted with ALL available items
        df_injection = pd.DataFrame()

        current_items_set = set(df_sample[item_col].unique())
        n_items_current = len(current_items_set)

        user_inter_counts = df_sample[user_col].value_counts()
        max_inter_user = user_inter_counts.max()
        # We need at least 1 negative item, ideally more.
        if max_inter_user >= n_items_current:
            print("  Injecting random items to allow negative sampling...")
            # We want n_items > max_inter_user. Let's say a 10% margin or at least 5 items.
            needed_total = int(max_inter_user * 1.1) + 5
            needed_new = needed_total - n_items_current
            # We look for items in the original dataset that are NOT in the current sample
            all_original_items = set(df[item_col].unique())
            available_to_add = list(all_original_items - current_items_set)

            if len(available_to_add) > 0:
                # Add real interactions of these new items
                items_to_inject = np.random.choice(available_to_add, min(len(available_to_add), needed_new), replace=False)

                # ATTEMPT 1: Search in already selected users (Ideal for not inflating users)
                df_injection_found = df[
                    df[item_col].isin(items_to_inject) &
                    df[user_col].isin(df_sample[user_col].unique())
                ]

                # ATTEMPT 2: If it fails, fetch interactions from ANY user (Necessary to save the dataset)
                if len(df_injection_found) == 0:
                    print("  No interactions found in current users. Fetching from external users...")
                    # We take 1 interaction for each new item to guarantee its existence
                    # This will bring a few new users, but it is a lesser evil
                    df_injection_found = pd.DataFrame()
                    for item in items_to_inject:
                        item_inters = df[df[item_col] == item]
                        if len(item_inters) > 0:
                            # We take 1 random interaction from this item
                            df_injection_found = pd.concat([df_injection_found, item_inters.sample(1)])

                if len(df_injection_found) > 0:
                    df_injection = df_injection_found
                    print(f"  Prepared {len(df_injection)} interactions from {len(items_to_inject)} extra items (PROTECTED).")
                else:
                    print("  Could not find valid interactions for extra items. Dropping saturated users...")
                    # If we cannot inject, we must clean df_sample
                    current_items_count = df_sample[item_col].nunique()
                    user_counts_check = df_sample[user_col].value_counts()

                    saturated_users = user_counts_check[user_counts_check >= current_items_count].index

                    if len(saturated_users) > 0:
                        df_sample = df_sample[~df_sample[user_col].isin(saturated_users)]
                        print(f"  Dropped {len(saturated_users)} users who had interacted with ALL items.")
                    else:
                        print("  No saturated users found (check logic).")
        # 5. Fine Adjustment to Target
        total_available = len(df_sample) + len(df_injection)
        print(f"Interactions available: {total_available} (Core: {len(df_sample)}, Injected: {len(df_injection)})")

        if total_available > target_interactions:
            # Must sample to reduce to the requested amount of interactions
            needed_from_core = target_interactions - len(df_injection)      # If we injected interactions, we keep them
            if needed_from_core > 0:
                # We sample only from the core
                df_core_sampled = df_sample.sample(n=needed_from_core, random_state=seed)
                # We join sampled core + full injection
                df_final = pd.concat([df_core_sampled, df_injection])
            else:
                # The injection is larger than the total target
                df_final = df_injection.sample(n=target_interactions, random_state=seed)
        else:
            # To fix this, the initial safety factor should be increased (1.1 -> 1.3)
            print(f"Warning: Could not reach {target_interactions}, got {total_available}. (Try increasing source pool)")
            df_final = pd.concat([df_sample, df_injection])

        # --- FINALIZATION AND SAVING ---
        final_users = df_final[user_col].nunique()
        final_items = df_final[item_col].nunique()
        final_interactions = len(df_final)
        
        print(f"Final Stats Sample {i+1}:")
        print(f"  Interactions: {final_interactions}")
        print(f"  Users: {final_users}")
        print(f"  Items: {final_items}")
        print(f"  Density: {final_interactions/(final_users*final_items):.4%}")

        # Save
        sampled_dataset_name = f'{dataset_name}_sample{i+1}'
        sampled_dir = f'{sampled_base_dir}/{sampled_dataset_name}'
        os.makedirs(sampled_dir, exist_ok=True)

        # Normalize column names to RecBole standard before saving
        df_save = df_final.rename(columns={user_col: 'user_id:token', item_col: 'item_id:token'})
        df_save.to_csv(os.path.join(sampled_dir, f'{sampled_dataset_name}.inter'), sep='\t', index=False)
        
        # Copy metadata (.user, .item)
        original_dir = f'./dataset/{dataset_name}'
        
        if os.path.exists(os.path.join(original_dir, f'{dataset_name}.user')):
            u_df = pd.read_csv(os.path.join(original_dir, f'{dataset_name}.user'), sep='\t')
            u_df = u_df[u_df[user_col].isin(df_final[user_col].unique())]
            u_df = u_df.rename(columns={user_col: 'user_id:token'})
            u_df.to_csv(os.path.join(sampled_dir, f'{sampled_dataset_name}.user'), sep='\t', index=False)

        if os.path.exists(os.path.join(original_dir, f'{dataset_name}.item')):
            i_df = pd.read_csv(os.path.join(original_dir, f'{dataset_name}.item'), sep='\t')
            item_col_in_item_file = item_col if item_col in i_df.columns else 'item_id:token'
            i_df = i_df[i_df[item_col_in_item_file].isin(df_final[item_col].unique())]
            i_df = i_df.rename(columns={item_col_in_item_file: 'item_id:token'})
            i_df.to_csv(os.path.join(sampled_dir, f'{sampled_dataset_name}.item'), sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--config', type=str, default='test.yaml')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for samples (overrides default ../dataset_sampled)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        sampling_config = config.get('sampling', {})

    if sampling_config.get('enabled', False):
        kwargs = dict(
            dataset_name=args.dataset,
            target_interactions=sampling_config.get('target_interactions', 100000),
            n_samples=sampling_config.get('n_samples', 3),
            base_seed=sampling_config.get('random_seed', 42),
            min_items_per_user=sampling_config.get('min_items_per_user', 5),
        )
        if args.output_dir:
            kwargs['output_dir'] = args.output_dir
        create_interaction_based_subsamples(**kwargs)
        print("\nSubsamples created successfully!")