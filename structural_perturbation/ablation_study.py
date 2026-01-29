import os
import pandas as pd
import numpy as np
from structural_perturbation import analytical_structural_perturbation_v2

# Ajusta esta función si tus archivos están en otro lugar
DATA_DIR = './dataset_sampled_train_split_full/'

def load_and_prepare_data(filepath):
    """Carga y prepara el dataset."""
    df = pd.read_csv(filepath, sep='\t')
    rename_map = {
        'user_id:token': 'user_id', 'item_id:token': 'item_id',
        'rating:float': 'rating', 'timestamp:float': 'timestamp',
    }
    df.rename(columns=rename_map, inplace=True)
    return df

def main():
    # 1. Definir el espacio de búsqueda (Grid)
    # Rango de p (perturbation fraction): típicamente valores pequeños
    p_values = [0.05, 0.1, 0.15, 0.2, 0.25]
    
    # Rango de alpha (0.0 = solo estructural, 1.0 = solo valor)
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Configuraciones fijas
    n_iterations = 3     # Reducido para que el grid search no tarde demasiado
    n_components = 50

    # 2. Cargar lista de archivos
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directorio {DATA_DIR} no encontrado.")
        return
    
    dataset_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.inter')]
    dataset_files.sort()
    
    print(f"Datasets encontrados: {len(dataset_files)}")
    print(f"Combinaciones a ejecutar: {len(p_values) * len(alpha_values)} por dataset")

    ablation_results = []

    # 3. Cachear DataFrames para no recargarlos en cada loop interno
    loaded_datasets = {}
    for filename in dataset_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            df = load_and_prepare_data(filepath)
            loaded_datasets[filename] = df
        except Exception as e:
            print(f"Error cargando {filename}: {e}")

    # 4. Ejecutar Grid Search
    # Iteramos p y alpha
    total_steps = len(p_values) * len(alpha_values)
    current_step = 0

    for p in p_values:
        for alpha in alpha_values:
            current_step += 1
            print(f"[{current_step}/{total_steps}] Procesando Grid: p={p}, alpha={alpha} ...")
            
            for filename, df in loaded_datasets.items():
                try:
                    time_sampling = 'timestamp' in df.columns
                    
                    # Calcular métricas
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
                    print(f"  Error en {filename}: {e}")

    # 5. Guardar resultados
    output_file = './structural_perturbation/ablation_results.csv'
    if ablation_results:
        results_df = pd.DataFrame(ablation_results)
        results_df.to_csv(output_file, index=False)
        print(f"\nResultados de ablación guardados en: {output_file}")
    else:
        print("No se generaron resultados.")

if __name__ == "__main__":
    main()