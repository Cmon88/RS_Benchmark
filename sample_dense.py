import pandas as pd
import os
import yaml
import numpy as np
import argparse

def create_interaction_based_subsamples(dataset_name, target_interactions, n_samples, base_seed=42, 
                                      min_items_per_user=5):
    """
    Crea subsamples buscando alcanzar 'target_interactions' manteniendo la consistencia.
    Usa un enfoque iterativo para encontrar el número correcto de usuarios.
    """
    
    original_path = f'./dataset/{dataset_name}/{dataset_name}.inter'
    
    if not os.path.exists(original_path):
        print(f"Original dataset not found: {original_path}")
        return
    
    df = pd.read_csv(original_path, sep='\t')
    
    # Estadísticas originales para estimación inicial
    avg_inter_per_user = len(df) / df['user_id:token'].nunique()
    
    print(f"=== Creating Interaction-Targeted Subsamples for {dataset_name} ===")
    print(f"Target Interactions: {target_interactions}")
    print(f"Avg interactions/user (original): {avg_inter_per_user:.2f}")

    sampled_base_dir = f'./dataset_sampled'
    os.makedirs(sampled_base_dir, exist_ok=True)
    
    for i in range(n_samples):
        print(f"\n--- Creating sample {i+1}/{n_samples} ---")
        seed = base_seed + i
        np.random.seed(seed)
        
        # 1. Filtrar usuarios "fríos" primero (necesitamos usuarios útiles)
        user_counts = df['user_id:token'].value_counts()
        valid_candidates = user_counts[user_counts >= min_items_per_user].index
        df_candidates = df[df['user_id:token'].isin(valid_candidates)]
        
        # Recalcular promedio sobre candidatos válidos para mejor estimación
        avg_valid = len(df_candidates) / df_candidates['user_id:token'].nunique()
        
        # 2. Estimación inicial de usuarios necesarios
        # Añadimos un factor de seguridad (1.1) porque al filtrar items después, perderemos interacciones
        estimated_users = int((target_interactions / avg_valid) * 1.1)
        
        # Límite de seguridad
        max_users = df_candidates['user_id:token'].nunique()
        n_users_to_sample = min(estimated_users, max_users)
        
        print(f"Estimating we need ~{n_users_to_sample} users to reach target...")
        
        # 3. Selección de Usuarios (Random o Top-Active? El paper usa Random sobre filtrados)
        # Usaremos Random para evitar sesgo extremo de popularidad, pero sobre usuarios válidos
        selected_users = np.random.choice(valid_candidates, n_users_to_sample, replace=False)
        df_sample = df[df['user_id:token'].isin(selected_users)].copy()
        
        # 4. Limpieza de Items (Garantía del Paper: "Union of 2 items per user")
        # Esto asegura que no tengamos items con 1 sola interacción perdida
        # Para datasets grandes, podemos relajar esto y simplemente quitar items con < X interacciones globales en el sample
        
        # Filtro iterativo simple para limpiar el grafo (k-core decomposition light)
        # Quitamos items que quedaron con muy pocas interacciones en este subgrupo
        min_item_support = 2
        item_counts = df_sample['item_id:token'].value_counts()
        valid_items = item_counts[item_counts >= min_item_support].index
        df_sample = df_sample[df_sample['item_id:token'].isin(valid_items)]
        
        # 5. Ajuste Fino al Target
        current_interactions = len(df_sample)
        print(f"Interactions after filtering: {current_interactions}")
        
        if current_interactions > target_interactions:
            # Si nos pasamos, hacemos un sample aleatorio simple para llegar al número exacto
            # Esto mantiene la distribución general pero clava el número
            df_final = df_sample.sample(n=target_interactions, random_state=seed)
        else:
            # Si nos faltan, es difícil "inventar". 
            # Opción A: Aceptar menos.
            # Opción B (Implementada): Avisar y devolver lo que hay.
            # Para arreglar esto en el futuro, habría que aumentar el factor de seguridad inicial (1.1 -> 1.3)
            print(f"Warning: Could not reach {target_interactions}, got {current_interactions}. (Try increasing source pool)")
            df_final = df_sample
            
        # --- FINALIZACIÓN Y GUARDADO ---
        final_users = df_final['user_id:token'].nunique()
        final_items = df_final['item_id:token'].nunique()
        final_interactions = len(df_final)
        
        print(f"Final Stats Sample {i+1}:")
        print(f"  Interactions: {final_interactions}")
        print(f"  Users: {final_users}")
        print(f"  Items: {final_items}")
        print(f"  Density: {final_interactions/(final_users*final_items):.4%}")

        # Guardar
        sampled_dataset_name = f'{dataset_name}_sample{i+1}'
        sampled_dir = f'{sampled_base_dir}/{sampled_dataset_name}'
        os.makedirs(sampled_dir, exist_ok=True)
        
        df_final.to_csv(os.path.join(sampled_dir, f'{sampled_dataset_name}.inter'), sep='\t', index=False)
        
        # Copiar metadatos (.user, .item)
        original_dir = f'./dataset/{dataset_name}'
        
        if os.path.exists(os.path.join(original_dir, f'{dataset_name}.user')):
            u_df = pd.read_csv(os.path.join(original_dir, f'{dataset_name}.user'), sep='\t')
            u_df = u_df[u_df['user_id:token'].isin(df_final['user_id:token'].unique())]
            u_df.to_csv(os.path.join(sampled_dir, f'{sampled_dataset_name}.user'), sep='\t', index=False)
            
        if os.path.exists(os.path.join(original_dir, f'{dataset_name}.item')):
            i_df = pd.read_csv(os.path.join(original_dir, f'{dataset_name}.item'), sep='\t')
            i_df = i_df[i_df['item_id:token'].isin(df_final['item_id:token'].unique())]
            i_df.to_csv(os.path.join(sampled_dir, f'{sampled_dataset_name}.item'), sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k')
    parser.add_argument('--config', type=str, default='test.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        sampling_config = config.get('sampling', {})
    
    if sampling_config.get('enabled', False):
        create_interaction_based_subsamples(
            dataset_name=args.dataset,
            target_interactions=sampling_config.get('target_interactions', 100000),
            n_samples=sampling_config.get('n_samples', 3),
            base_seed=sampling_config.get('random_seed', 42),
            min_items_per_user=sampling_config.get('min_items_per_user', 5)
        )
        print("\nSubsamples created successfully!")