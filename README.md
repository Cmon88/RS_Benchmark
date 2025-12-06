# RS_Benchmark

Benchmark de Sistemas de Recomendación utilizando [RecBole](https://recbole.io/) para el estudio de la estabilidad estructural de sus datasets. Este repositorio permite generar submuestras controladas de un dataset, analizar sus propiedades y ejecutar una batería de modelos de recomendación para comparar su rendimiento.

## Requisitos e Instalación

Este proyecto utiliza Python 3.12. Para instalar las dependencias necesarias, sigue estos pasos:

1.  Clona el repositorio:
    ```bash
    git clone <url-del-repo>
    cd RS_Benchmark
    ```

2.  (Opcional pero recomendado) Crea y activa un entorno virtual:
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Linux/Mac
    # venv\Scripts\activate   # En Windows
    ```

3.  Instala las dependencias desde el archivo `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

**Nota:** Asegúrate de tener instalado [RecBole](https://recbole.io/) correctamente, especialmente si requieres soporte para GPU (Torch).


## Estructura de Directorios

El proyecto espera la siguiente estructura para los datasets originales:

```text
RS_Benchmark/
├── dataset/
│   └── nombre_dataset/
│       ├── nombre_dataset.inter
│       ├── nombre_dataset.item  (opcional)
│       └── nombre_dataset.user  (opcional)
├── dataset_sampled/             (Generado automáticamente)
├── latex/                       (Resultados generados)
├── test_dense.yaml              (Archivo de configuración)
└── ... scripts .py ...
```

## Flujo de Trabajo

1. Configuración (YAML)

    Antes de ejecutar nada, asegúrate de tener un archivo .yaml (por ejemplo, test_dense.yaml) con la configuración de muestreo.

    Ejemplo de configuración:
    ```yaml
    sampling:
    enabled: true
    n_samples: 3              # Número de muestras a generar
    target_interactions: 100000 # Interacciones objetivo por muestra
    random_seed: 42
    min_items_per_user: 3
    min_total_items: 10
    
    ```

2. Generación de Muestras (sample_dense.py)

    Este script toma el dataset original y genera submuestras balanceadas basándose en la configuración del YAML. Las muestras se guardan en la carpeta dataset_sampled.
    
    Uso:
    ```terminal
    python sample_dense.py --dataset <nombre_dataset> --config <archivo_config.yaml>
    
    ```
    Ejemplo:
    ```terminal
    python sample_dense.py --dataset amazon_digital_music --config test_dense.yaml
    ```

3. Diagnóstico de Datos (diagnostic.py)

    Una vez generadas las muestras, utiliza este script para verificar las estadísticas (densidad, número de usuarios/items, interacciones) y compararlas con el dataset original y una referencia (ml-100k).

    Uso:
    ```terminal
    python diagnostic.py --dataset <nombre_dataset> --n_samples <numero_de_muestras>
    ```
    
    Ejemplo:
    ```terminal
    python diagnostic.py --dataset amazon_digital_music --n_samples 3
    ```

4. Ejecución del Benchmark (general.py)

    Este es el script principal que entrena y evalúa los modelos definidos en la lista general_models. Ejecuta los modelos sobre las muestras generadas y consolida los resultados.

    Uso:
    ```terminal
    python general.py --dataset <nombre_dataset> --config <archivo_config.yaml>
    ```
    
    Ejemplo:
    ```terminal
    python general.py --dataset amazon_digital_music --config test_dense.yaml
    ```

    El Script realizara lo siguiente:
    1. Cargará la configuración de muestreo.
    2. Dividirá los modelos en grupos para su ejecución secuencial.
    3. Entrenará cada modelo sobre cada muestra generada.
    4. Consolidará los resultados en archivos CSV y tablas LaTeX.

## Resultados

Los resultados finales se guardan en la carpeta latex:
- Archivos CSV: Contienen los datos crudos de las métricas para validación y test (final_test_...csv).

- Archivos TeX: Tablas formateadas listas para incluir en documentos LaTeX, con los mejores resultados resaltados en negrita (final_test_...tex).

## Datasets
Los datos utilizados son los entregados por RecBole en su [Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI),  ya formateados en su propia estructura. \

Para utilizarlos basta descargar los .inter del dataset y dejarlo en su directorio propio en el directorio `dataset`. \
Por ejemplo: `dataset/ml-100k/ml-100k.inter`
