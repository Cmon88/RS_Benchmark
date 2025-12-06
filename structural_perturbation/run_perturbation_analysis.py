import os
import pandas as pd
import numpy as np
from structural_perturbation import analytical_structural_perturbation_v2

def load_and_prepare_data(filepath):
    """
    Carga un archivo .inter y lo prepara para el análisis de perturbación.
    """
    df = pd.read_csv(filepath, sep='\t')
    
    # Renombrar columnas al formato esperado por la función de análisis
    rename_map = {
        'user_id:token': 'user_id',
        'item_id:token': 'item_id',
        'rating:float': 'rating',
        'timestamp:float': 'timestamp'
    }
    df.rename(columns=rename_map, inplace=True)
    
    # Asegurarse de que las columnas requeridas existan
    required_cols = ['user_id', 'item_id', 'rating', 'timestamp']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"La columna requerida '{col}' no se encuentra en {filepath}")
            
    return df

def main():
    # Directorio con los splits de entrenamiento completos
    data_dir = './dataset_sampled_train_split_full/'
    
    # Obtener la lista de todos los archivos .inter
    try:
        dataset_files = [f for f in os.listdir(data_dir) if f.endswith('.inter')]
        if not dataset_files:
            print(f"No se encontraron archivos .inter en el directorio: {data_dir}")
            return
    except FileNotFoundError:
        print(f"El directorio no existe: {data_dir}")
        return

    print(f"Se encontraron {len(dataset_files)} datasets para analizar.\n")

    # Configurar parámetros para el análisis
    p = 0.1              # Perturbar 10% de los ratings
    n_iterations = 5     # Promediar sobre 5 ejecuciones
    n_components = 50    # Usar 50 factores latentes
    alpha = 0.7          # 70% perturbación de valor, 30% estructural
    time_sampling = True # Ponderar ratings más nuevos en el muestreo
    
    results_list = []   

    # Iterar sobre cada archivo de dataset y ejecutar el análisis
    for filename in sorted(dataset_files):
        print(f"--- Analizando: {filename} ---")
        filepath = os.path.join(data_dir, filename)
        
        try:
            # Cargar y preparar los datos
            df = load_and_prepare_data(filepath)

            # Ejecutar el análisis de perturbación estructural
            rmse, std_rmse, s_distance, std_s_distance, rmse_svd, std_rmse_svd = \
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
            })

            # Mostrar resultados para el dataset actual
            print("=" * 60)
            print("RESULTS")
            print("=" * 60)
            print(f"  Structural Perturbation RMSE: {rmse:.4f} ± {std_rmse:.4f}")
            print(f"  Spectral Distance:            {s_distance:.4f} ± {std_s_distance:.4f}")
            print(f"  Standard SVD RMSE:            {rmse_svd:.4f} ± {std_rmse_svd:.4f}")
            print(f"  Normalized RMSE:              {(rmse / rmse_svd if rmse_svd > 0 else float('inf')):.4f}")
            print()

        except Exception as e:
            print(f"\n  ERROR al procesar {filename}: {e}\n")
    
    # Guardar todos los resultados en un archivo CSV
    output_csv_file = './structural_perturbation/perturbation_results.csv'
    if results_list:
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(output_csv_file, index=False, float_format='%.4f')
        print(f"Resultados guardados exitosamente en '{output_csv_file}'")

    print("Interpretation:")
    print(f"  - Lower RMSE indicates more structural consistency")
    print(f"  - Lower spectral distance indicates more stable latent structure")
    print(f"  - Normalized RMSE allows comparison across different datasets")
    print("=" * 60)

if __name__ == "__main__":
    main()