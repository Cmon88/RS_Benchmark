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

        df_work = df.copy()

        # 0. Filtro de Saturación
        total_items_global = df['item_id:token'].nunique()
        items_per_user = df_work.groupby('user_id:token')['item_id:token'].nunique()
        saturated_users_ids = items_per_user[items_per_user >= (total_items_global * 0.95)].index
        if len(saturated_users_ids) > 0:
            print(f"  Excluding {len(saturated_users_ids)} saturated users.")
            df_work = df_work[~df_work['user_id:token'].isin(saturated_users_ids)]
        
        # 1. Filtrar usuarios "fríos" (Usamos df_clean en lugar de df)
        user_counts = df_work['user_id:token'].value_counts()
        valid_candidates = user_counts[user_counts >= min_items_per_user].index
        df_candidates = df_work[df_work['user_id:token'].isin(valid_candidates)]
        
        # Recalcular promedio sobre candidatos válidos para mejor estimación
        avg_valid = len(df_candidates) / df_candidates['user_id:token'].nunique()
        
        # 2. Estimación inicial de usuarios necesarios
        # Añadimos un factor de seguridad (1.1) porque al filtrar items después, perderemos interacciones
        estimated_users = int((target_interactions / avg_valid) * 2.0)
        
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

        # Verificar si algún usuario ha interactuado con TODOS los items disponibles
        df_injection = pd.DataFrame()

        current_items_set = set(df_sample['item_id:token'].unique())
        n_items_current = len(current_items_set)
        
        user_inter_counts = df_sample['user_id:token'].value_counts()
        max_inter_user = user_inter_counts.max()
        # Necesitamos al menos 1 item negativo, idealmente más.
        if max_inter_user >= n_items_current:
            print("  Injecting random items to allow negative sampling...")
            # Queremos que n_items > max_inter_user. Digamos un margen del 10% o al menos 5 items.
            needed_total = int(max_inter_user * 1.1) + 5
            needed_new = needed_total - n_items_current
            # Buscamos items en el dataset original que NO estén en el sample actual
            all_original_items = set(df['item_id:token'].unique())
            available_to_add = list(all_original_items - current_items_set)
            
            if len(available_to_add) > 0:
                # Añadir interacciones reales de estos items nuevos
                items_to_inject = np.random.choice(available_to_add, min(len(available_to_add), needed_new), replace=False)
                
                # INTENTO 1: Buscar en usuarios ya seleccionados (Ideal para no inflar usuarios)
                df_injection_found = df[
                    df['item_id:token'].isin(items_to_inject) & 
                    df['user_id:token'].isin(df_sample['user_id:token'].unique())
                ]
                
                # INTENTO 2: Si falla, traer interacciones de CUALQUIER usuario (Necesario para salvar el dataset)
                if len(df_injection_found) == 0:
                    print("  No interactions found in current users. Fetching from external users...")
                    # Tomamos 1 interacción por cada item nuevo para garantizar su existencia
                    # Esto traerá unos pocos usuarios nuevos, pero es un mal menor
                    df_injection_found = pd.DataFrame()
                    for item in items_to_inject:
                        item_inters = df[df['item_id:token'] == item]
                        if len(item_inters) > 0:
                            # Tomamos 1 interacción aleatoria de este item
                            df_injection_found = pd.concat([df_injection_found, item_inters.sample(1)])
                
                if len(df_injection_found) > 0:
                    df_injection = df_injection_found
                    print(f"  Prepared {len(df_injection)} interactions from {len(items_to_inject)} extra items (PROTECTED).")
                else:
                    print("  Could not find valid interactions for extra items. Dropping saturated users...")
                    # Si no podemos inyectar, debemos limpiar df_sample
                    current_items_count = df_sample['item_id:token'].nunique()
                    user_counts_check = df_sample['user_id:token'].value_counts()
                    
                    saturated_users = user_counts_check[user_counts_check >= current_items_count].index
                    
                    if len(saturated_users) > 0:
                        df_sample = df_sample[~df_sample['user_id:token'].isin(saturated_users)]
                        print(f"  Dropped {len(saturated_users)} users who had interacted with ALL items.")
                    else:
                        print("  No saturated users found (check logic).")
        # 5. Ajuste Fino al Target
        total_available = len(df_sample) + len(df_injection)
        print(f"Interactions available: {total_available} (Core: {len(df_sample)}, Injected: {len(df_injection)})")
        
        if total_available > target_interactions:
            # Debe samplear para reducir a las cantidad pedida de interacciones
            needed_from_core = target_interactions - len(df_injection) # Si inyectamos interacciones, las mantenemos
            if needed_from_core > 0:
                # Sampleamos solo del core
                df_core_sampled = df_sample.sample(n=needed_from_core, random_state=seed)
                # Unimos core sampleado + inyección completa
                df_final = pd.concat([df_core_sampled, df_injection])
            else:
                # La inyección es más grande que el target total
                df_final = df_injection.sample(n=target_interactions, random_state=seed)
        else:
            # Para arreglar esto en el futuro, habría que aumentar el factor de seguridad inicial (1.1 -> 1.3)
            print(f"Warning: Could not reach {target_interactions}, got {total_available}. (Try increasing source pool)")
            df_final = pd.concat([df_sample, df_injection])
            
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