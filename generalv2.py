import subprocess
import time
import yaml
import pandas as pd
import os
import glob
import argparse

def consolidate_results(dataset_name, output_suffix=""):
    """Consolidar todos los resultados de grupos en archivos finales"""
    
    # Buscar todos los archivos CSV del benchmark
    valid_csv_files = glob.glob(f"./latex/valid_{dataset_name}_group*.csv")
    test_csv_files = glob.glob(f"./latex/test_{dataset_name}_group*.csv")
    
    all_valid_results = []
    all_test_results = []
    
    # Consolidar resultados de validación
    for csv_file in valid_csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_valid_results.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Consolidar resultados de test
    for csv_file in test_csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_test_results.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    # Combinar todos los resultados
    if all_valid_results:
        final_valid_df = pd.concat(all_valid_results, ignore_index=True)
        final_valid_file = f"./latex/final_valid_{dataset_name}{output_suffix}.csv"
        final_valid_df.to_csv(final_valid_file, index=False)
        print(f"Consolidated validation results: {final_valid_file}")
        
        # Generar LaTeX con highlighting entre todos los modelos
        generate_final_latex(final_valid_df, f"./latex/final_valid_{dataset_name}{output_suffix}.tex", "Validation")
    
    if all_test_results:
        final_test_df = pd.concat(all_test_results, ignore_index=True)
        final_test_file = f"./latex/final_test_{dataset_name}{output_suffix}.csv"
        final_test_df.to_csv(final_test_file, index=False)
        print(f"Consolidated test results: {final_test_file}")
        
        # Generar LaTeX con highlighting entre todos los modelos
        generate_final_latex(final_test_df, f"./latex/final_test_{dataset_name}{output_suffix}.tex", "Test")


def generate_final_latex(df, output_file, caption_prefix):
    """Generar archivo LaTeX final con highlighting entre todos los modelos"""
    
    if df.empty:
        return
    
    # Identificar columnas de métricas (todas excepto 'Model')
    metric_columns = [col for col in df.columns if col != 'Model']
    
    # Crear LaTeX manualmente con highlighting
    latex_content = f"""\\begin{{table}}
\\caption{{{caption_prefix} Results - {dataset_name}}}
\\label{{{caption_prefix.lower()}_results}}
\\begin{{tabular}}{{{'l' + 'c' * len(metric_columns)}}}
\\toprule
Model & {' & '.join(metric_columns)} \\\\
\\midrule
"""
    lower_is_better = ['rmse', 'mae', 'logloss']
    # Encontrar los mejores valores para cada métrica
    best_values = {}
    for metric in metric_columns:
        if metric in df.columns:
            if metric.lower() in lower_is_better:
                best_values[metric] = df[metric].min()
            else:
                best_values[metric] = df[metric].max()
    
    # Generar filas para cada modelo
    for _, row in df.iterrows():
        model_name = row['Model']
        metric_values = []
        
        for metric in metric_columns:
            if metric in row:
                value = row[metric]
                # Formatear valor y agregar negrita si es el mejor
                formatted_value = f"{value:.4f}"
                if value == best_values.get(metric):
                    formatted_value = f"\\bfseries {formatted_value}"
                metric_values.append(formatted_value)
        
        latex_content += f"{model_name} & {' & '.join(metric_values)} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}"""
    
    # Guardar archivo LaTeX
    with open(output_file, "w") as f:
        f.write(latex_content)
    
    print(f"Generated final LaTeX: {output_file}")

# General parameters
general_models = ['Pop', 'BPR', 'FISM', 'ItemKNN', 'CDAE', 'DMF', 'NeuMF', 'NNCF', 'ConvNCF', 'GCMC', 'MultiDAE', 'MultiVAE', 'SpectralCF', 'EASE', 'MacridVAE', 'NCEPLRec', 'NGCF', 'DGCF', 'ENMF', 'LightGCN', 'RecVAE', 'SGL', 'SimpleX', 'LDiffRec']
#general_models = ['BPR']  # Para pruebas

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run general benchmark')
parser.add_argument('--dataset', type=str, default='ml-1m', help='Dataset name')
parser.add_argument('--config', type=str, default='test_dense.yaml', help='Config file path')
args = parser.parse_args()

dataset_name = args.dataset
config_file = args.config


# Cargar configuración de muestreo
def load_sampling_config():
    with open('test_dense.yaml', 'r') as f:
        config = yaml.safe_load(f)
        return config.get('sampling', {'enabled': False, 'n_samples': 1})

sampling_config = load_sampling_config()

# Split models into groups
group_size = 2
model_groups = [general_models[i:i + group_size] for i in range(0, len(general_models), group_size)]
times = []

print(f"Sampling configuration: {sampling_config}")
if sampling_config['enabled']:
    print(f"Running {sampling_config['n_samples']} samples per model")
    

os.makedirs('./latex', exist_ok=True)

# Execute the benchmark for each group
for idx, group in enumerate(model_groups):
    general_list = ",".join(group)
    output_suffix = f"_group{idx+1}"
    command = f"python run_recbole_groupv2.py --model_list={general_list} --dataset={dataset_name} --config_files={config_file} --output_suffix={output_suffix}"
    
    # Measure execution time
    start_time = time.time()
    subprocess.run(command, shell=True, check=True)
    end_time = time.time()
    
    # Print the elapsed time for the group
    print(f"Group {idx+1} benchmark time: {end_time - start_time:.2f} seconds")
    times.append(end_time - start_time)


# Consolidar todos los resultados al final
print("\nConsolidating all results...")
consolidate_results(dataset_name, "_benchmark")

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
print(f"- ./latex/final_valid_{dataset_name}_benchmark.tex")
print(f"- ./latex/final_test_{dataset_name}_benchmark.tex")
print(f"- ./latex/final_valid_{dataset_name}_benchmark.csv") 
print(f"- ./latex/final_test_{dataset_name}_benchmark.csv")