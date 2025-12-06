import pandas as pd
import os
import yaml
import shutil
import numpy as np
import argparse

def calculate_target_parameters(df, target_interactions=100000):
    """Calcular parámetros objetivo basados en las proporciones del dataset original"""
    
    original_users = df['user_id:token'].nunique()
    original_items = df['item_id:token'].nunique()
    original_interactions = len(df)
    
    avg_interactions_per_user = original_interactions / original_users
    avg_interactions_per_item = original_interactions / original_items
    
    print(f"Original stats:")
    print(f"  Users: {original_users}, Items: {original_items}, Interactions: {original_interactions}")
    print(f"  Avg interactions/user: {avg_interactions_per_user:.2f}")
    print(f"  Avg interactions/item: {avg_interactions_per_item:.2f}")
    
    # Calcular targets basados en proporciones
    target_users = int(target_interactions / avg_interactions_per_user)
    target_items = int(target_interactions / avg_interactions_per_item)
    
    # Ajustar para mantener densidad similar
    density = original_interactions / (original_users * original_items)
    estimated_density = target_interactions / (target_users * target_items)
    
    print(f"Targets for {target_interactions} interactions:")
    print(f"  Target users: {target_users}")
    print(f"  Target items: {target_items}")
    print(f"  Estimated density: {estimated_density:.4%}")
    print(f"  Original density: {density:.4%}")
    
    return target_users, target_items

def create_balanced_subsamples(dataset_name, target_interactions, n_samples, base_seed=42, 
                              min_items_per_user=3, min_total_items=10):
    """
    Crear subsamples balanceados que mantengan las proporciones del original
    """
    original_path = f'./dataset/{dataset_name}/{dataset_name}.inter'
    
    if not os.path.exists(original_path):
        print(f"Original dataset not found: {original_path}")
        return
    
    # Leer datos originales
    df = pd.read_csv(original_path, sep='\t')
    
    print(f"=== Creating balanced subsamples for {dataset_name} ===")
    print(f"Target interactions: {target_interactions}")
    
    # Calcular parámetros automáticamente
    target_users, target_items = calculate_target_parameters(df, target_interactions)
    
    # Ajustar mínimos
    min_items_per_user = max(min_items_per_user, 2)  # Al menos 2
    target_users = max(target_users, min_total_items)
    target_items = max(target_items, min_total_items)
    
    print(f"Adjusted targets:")
    print(f"  Users: {target_users}, Items: {target_items}")
    
    # Crear directorio base
    sampled_base_dir = f'./dataset_sampled'
    os.makedirs(sampled_base_dir, exist_ok=True)
    
    for i in range(n_samples):
        print(f"\n--- Creating balanced sample {i+1}/{n_samples} ---")
        seed = base_seed + i
        
        # Seleccionar usuarios activos primero
        user_interaction_counts = df['user_id:token'].value_counts()
        
        # Ordenar usuarios por actividad (más activos primero)
        active_users = user_interaction_counts[user_interaction_counts >= min_items_per_user]
        if len(active_users) < target_users:
            print(f"Warning: Only {len(active_users)} users have ≥{min_items_per_user} interactions")
            target_users = len(active_users)
        
        # Seleccionar los usuarios más activos
        selected_users = active_users.head(target_users).index
        df_user_filtered = df[df['user_id:token'].isin(selected_users)]
        
        print(f"After selecting {target_users} most active users: {len(df_user_filtered)} interactions")
        
        # Seleccionar items populares entre estos usuarios
        item_interaction_counts = df_user_filtered['item_id:token'].value_counts()
        
        if len(item_interaction_counts) < target_items:
            print(f"Warning: Only {len(item_interaction_counts)} items available")
            target_items = len(item_interaction_counts)
        
        # Seleccionar los items más populares
        selected_items = item_interaction_counts.head(target_items).index
        df_filtered = df_user_filtered[df_user_filtered['item_id:token'].isin(selected_items)]
        
        print(f"After selecting {target_items} most popular items: {len(df_filtered)} interactions")
        
        # Ajustar tamaño final
        if len(df_filtered) > target_interactions:
            # Muestrear para alcanzar el target exacto
            df_final = df_filtered.sample(n=target_interactions, random_state=seed)
        else:
            df_final = df_filtered.copy()
            print(f"Note: Only {len(df_final)} interactions available (less than target {target_interactions})")
        
        # Estadísticas finales
        final_users = df_final['user_id:token'].nunique()
        final_items = df_final['item_id:token'].nunique()
        final_interactions = len(df_final)
        
        print(f"Final sample {i+1}:")
        print(f"  Interactions: {final_interactions}")
        print(f"  Users: {final_users}")
        print(f"  Items: {final_items}")
        print(f"  Avg interactions/user: {final_interactions/final_users:.2f}")
        print(f"  Avg interactions/item: {final_interactions/final_items:.2f}")
        print(f"  Density: {final_interactions/(final_users*final_items):.4%}")
        
        # Guardar el dataset
        sampled_dataset_name = f'{dataset_name}_sample{i+1}'
        sampled_dir = f'{sampled_base_dir}/{sampled_dataset_name}'
        os.makedirs(sampled_dir, exist_ok=True)
        
        sampled_inter_path = os.path.join(sampled_dir, f'{sampled_dataset_name}.inter')
        df_final.to_csv(sampled_inter_path, sep='\t', index=False)
        
        # Copiar archivos .user y .item filtrados
        original_dir = f'./dataset/{dataset_name}'
        
        # Archivo .user
        user_file = f'{dataset_name}.user'
        if os.path.exists(os.path.join(original_dir, user_file)):
            user_df = pd.read_csv(os.path.join(original_dir, user_file), sep='\t')
            sampled_users = df_final['user_id:token'].unique()
            filtered_user_df = user_df[user_df['user_id:token'].isin(sampled_users)]
            filtered_user_df.to_csv(os.path.join(sampled_dir, f'{sampled_dataset_name}.user'), sep='\t', index=False)
        
        # Archivo .item
        item_file = f'{dataset_name}.item'
        if os.path.exists(os.path.join(original_dir, item_file)):
            item_df = pd.read_csv(os.path.join(original_dir, item_file), sep='\t')
            sampled_items = df_final['item_id:token'].unique()
            filtered_item_df = item_df[item_df['item_id:token'].isin(sampled_items)]
            filtered_item_df.to_csv(os.path.join(sampled_dir, f'{sampled_dataset_name}.item'), sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create balanced subsamples')
    parser.add_argument('--dataset', type=str, default='ml-100k', help='Name of the dataset to sample')
    parser.add_argument('--config', type=str, default='test.yaml', help='Path to configuration yaml file')
    args = parser.parse_args()

    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        sampling_config = config.get('sampling', {})
    
    if sampling_config.get('enabled', False):
        create_balanced_subsamples(
            dataset_name=args.dataset,
            target_interactions=sampling_config.get('target_interactions', 100000),
            n_samples=sampling_config.get('n_samples', 3),
            base_seed=sampling_config.get('random_seed', 42),
            min_items_per_user=sampling_config.get('min_items_per_user', 3),
            min_total_items=sampling_config.get('min_total_items', 10)
        )
        print("\nBalanced subsamples created successfully!")
    else:
        print("Sampling is disabled in config")