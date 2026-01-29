# @Time   : 2023/2/13
# @Author : Gaowei Zhang
# @Email  : zgw2022101006@ruc.edu.cn


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
    Guarda el split de entrenamiento como archivo .inter.
    Solo lo guarda si no existe previamente, asumiendo consistencia por seed.
    """
    save_dir = 'train_splits'
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{dataset_name}_train.inter')
    
    # Si ya existe, asumimos que es el mismo split (gracias a init_seed) y saltamos
    if os.path.exists(file_path):
        return

    try:
        df = pd.DataFrame()
        # RecBole almacena las columnas en inter_feat
        for field in dataset_obj.inter_feat:
            ids = dataset_obj.inter_feat[field]
            
            # Intentar recuperar los tokens originales (ej. user_id original string en vez de entero interno)
            # Esto hace que el archivo sea portable y legible
            if field in dataset_obj.field2id_token:
                try:
                    tokens = dataset_obj.id2token(field, ids)
                    df[field] = tokens
                except:
                    # Fallback a IDs internos si falla
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
    """Cargar configuracion de muestreo desde archivos YAML"""
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
    """Guardar promedios en CSV para consolidación posterior"""
    if result_list:
        # Filtrar solo los promedios
        averages = [res for res in result_list if res.get('Sample') == 'Average']
        if averages:
            df = pd.DataFrame(averages)
            # Reordenar columnas: Model primero, luego métricas
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
    # Archivos para este grupo
    valid_file = f"./latex/valid_{dataset}{args.output_suffix}.tex"
    test_file = f"./latex/test_{dataset}{args.output_suffix}.tex"
    # Archivos CSV para consolidación
    valid_csv_group = f"./latex/valid_{dataset}{args.output_suffix}.csv"
    test_csv_group = f"./latex/test_{dataset}{args.output_suffix}.csv"
    error_dir = './errors'
    os.makedirs(error_dir, exist_ok=True)

    # Cargar configuración de muestreo
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

        # Ejecutar múltiples muestras si el muestreo está habilitado
        n_samples = sampling_config['n_samples'] if sampling_config['enabled'] else 1
        
        for sample_idx in range(n_samples):
            print(f"  Sample {sample_idx+1}/{n_samples}")

            # Configuracion especifica para esta muestra
            sample_config_dict = {}
            dataset_to_use = dataset        # Por defecto usar el dataset original

            if sampling_config['enabled']:
                dataset_to_use = f'{dataset}_sample{sample_idx+1}'
                sample_config_dict = {
                    'data_path': './dataset_sampled/',
                }

            valid_res_dict = {"Model": model, "Sample": sample_idx+1}
            test_res_dict = {"Model": model, "Sample": sample_idx+1}
            
            # Ejecutar el modelo
            try:
                # Cargar configuracion
                config = Config(
                    model=model,
                    dataset=dataset_to_use,
                    config_file_list=config_file_list,
                    config_dict=sample_config_dict
                )
                init_seed(config['seed'], config['reproducibility'])
                # Preparar datos
                dataset_obj = create_dataset(config)
                train_data, valid_data, test_data = data_preparation(config, dataset_obj)
                # Guardar split de entrenamiento (se ejecutará solo la primera vez por dataset/sample)
                save_train_split(train_data.dataset, dataset_to_use)
                # Iniciar modelo
                model_obj = get_model(config['model'])(config, train_data.dataset).to(config['device'])
                # Iniciar Trainer
                trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model_obj)
                # Entrenar
                best_valid_score, best_valid_result = trainer.fit(
                    train_data, valid_data, saved=True, show_progress=True
                )
                # Evaluar: Metricas de ranking
                test_result_ranking = trainer.evaluate(test_data, load_best_model=True, show_progress=True)

                # Evaluar : Metricas de Valor
                with open('test_rmse.yaml', 'r') as f:
                    rmse_yaml_config = yaml.safe_load(f)
                # Combinamos la config de muestreo actual con la de RMSE
                rmse_config_dict = sample_config_dict.copy()
                rmse_config_dict.update(rmse_yaml_config)

                # Copiamos los argumentos de evaluación originales para mantener el split
                if 'eval_args' not in rmse_config_dict:
                    rmse_config_dict['eval_args'] = config['eval_args'].copy()
                
                rmse_config_dict['eval_args'].update({
                    'group_by': None,      # No agrupar por usuario
                    'order': 'RO',         # Orden aleatorio (no importa en labeled)
                    'mode': 'uni100'      # Modo Point-wise (necesario para regresión/RMSE)
                })
                rmse_config_dict['metrics'] = ['RMSE']
                
                # Crear NUEVOS objetos Config y Dataset
                config_rmse = Config(
                    model=model,
                    dataset=dataset_to_use,
                    config_file_list=config_file_list,
                    config_dict=rmse_config_dict
                )
                # Reiniciar semilla
                init_seed(config_rmse['seed'], config_rmse['reproducibility'])
                # Crear dataset limpio y dataloaders
                dataset_rmse = create_dataset(config_rmse)
                _, _, test_data_rmse = data_preparation(config_rmse, dataset_rmse)
                # Nuevo Trainer con el modelo YA ENTRENADO
                trainer_rmse = get_trainer(config_rmse['MODEL_TYPE'], config_rmse['model'])(config_rmse, model_obj)
                # Usar el NUEVO dataloader (test_data_rmse)
                test_result_rmse = trainer_rmse.evaluate(test_data_rmse, load_best_model=False, show_progress=True)
                print(f"    RMSE: Result: {test_result_rmse}")
                # Combinar resultados de test
                final_test_result = test_result_ranking.copy()
                final_test_result.update(test_result_rmse)
                valid_res_dict.update(best_valid_result)
                test_res_dict.update(final_test_result)
                bigger_flag = config['valid_metric_bigger']
                subset_columns = list(best_valid_result.keys()) # Usamos métricas de validación para ordenar columnas
                for k in final_test_result.keys():
                    if k not in subset_columns:
                        subset_columns.append(k)

                model_valid_results.append(valid_res_dict)
                model_test_results.append(test_res_dict)
                
                # Borra el modelo guardado para liberar espacio
                if hasattr(trainer, 'saved_model_file') and trainer.saved_model_file:
                    if os.path.exists(trainer.saved_model_file):
                        try:
                            os.remove(trainer.saved_model_file)
                            print(f"Deleted saved model file: {trainer.saved_model_file}")
                        except OSError as e:
                            print(f"Error deleting model file: {e}")

                # Borra los objetos para liberar memoria
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

        # Calcular promedios si se usó muestreo
        if len(model_valid_results)>0:
            if n_samples > 1:
                # Calcular promedio de metricas
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
                # Si solo hay una muestra, usar esa directamente como "promedio"
                single_valid = model_valid_results[0].copy()
                single_valid['Sample'] = 'Average'
                single_test = model_test_results[0].copy()
                single_test['Sample'] = 'Average'
                valid_result_list.append(single_valid)
                test_result_list.append(single_test)

    # Guardar resultados en LaTeX y CSV
    if valid_result_list:
        # Filtrar solo promedios para LaTeX
        avg_valid_list = [res for res in valid_result_list if res.get('Sample') == 'Average']
        avg_test_list = [res for res in test_result_list if res.get('Sample') == 'Average']
        if avg_valid_list:
            # evitar rmse pues no se uso en validacion
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
            # Guardar también en CSV para consolidación
            save_averages_to_csv(valid_result_list, valid_csv_group, subset_columns)
            save_averages_to_csv(test_result_list, test_csv_group, subset_columns)