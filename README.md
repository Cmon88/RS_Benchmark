# RS_Benchmark

Recommendation Systems Benchmark using [RecBole](https://recbole.io/) to study the structural stability of their datasets. This repository allows generating controlled subsamples of a dataset, analyzing their properties, and executing a group of recommendation models to compare their performance.

## Requirements and Installation

This project uses Python 3.12. To install the necessary dependencies, follow these steps:

1.  Clone the repository:
    ```bash
    git clone https://github.com/Cmon88/RS_Benchmark.git
    cd RS_Benchmark
    ```

2.  (Optional but recommended) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/Mac
    # venv\Scripts\activate   # On Windows
    ```

3.  Install dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```


## Directory Structure

The project expects the following structure for original datasets:

```text
RS_Benchmark/
├── dataset/
│   └── dataset_name/
│       ├── dataset_name.inter
│       ├── dataset_name.item  (optional)
│       └── dataset_name.user  (optional)
├── dataset_sampled/             (Automatically generated)
├── latex/                       (Generated results)
├── test_dense.yaml              (Configuration file)
└── ... .py scripts ...
```

## Workflow

1. Configuration (YAML)

    Before running anything, ensure you have a .yaml file (e.g., `test_dense.yaml`) with the sampling configuration.

    Configuration example:
    ```yaml
    sampling:
    enabled: true
    n_samples: 3              # Number of samples to generate
    target_interactions: 100000 # Target interactions per sample
    random_seed: 42
    min_items_per_user: 3
    min_total_items: 10
    
    ```

2. Sample Generation (sample_dense.py)

    This script takes the original dataset and generates balanced subsamples based on the YAML configuration. The samples are saved in the `dataset_sampled` folder.
    
    Usage:
    ```terminal
    python sample_dense.py --dataset <dataset_name> --config <config_file.yaml>
    
    ```
    Example:
    ```terminal
    python sample_dense.py --dataset amazon_digital_music --config test_dense.yaml
    ```

3. Data Diagnostics (diagnostic.py)

    Once samples are generated, use this script to verify statistics (density, number of users/items, interactions) and compare them with the original dataset and a reference (ml-100k).

    Usage:
    ```terminal
    python diagnostic.py --dataset <dataset_name> --n_samples <number_of_samples>
    ```
    
    Example:
    ```terminal
    python diagnostic.py --dataset amazon_digital_music --n_samples 3
    ```

4. Benchmark Execution (general.py)

    This is the main script that trains and evaluates the models defined in the `general_models` list. It runs the models on the generated samples and consolidates the results.

    Usage:
    ```terminal
    python general.py --dataset <dataset_name> --config <config_file.yaml>
    ```
    
    Example:
    ```terminal
    python general.py --dataset amazon_digital_music --config test_dense.yaml
    ```

    The script will perform the following:
    1. Load the sampling configuration.
    2. Split models into groups for sequential execution.
    3. Train each model on each generated sample.
    4. Consolidate results into CSV files and LaTeX tables.

## Results

Final results are saved in the `latex` folder:
- **CSV Files:** Contain raw metric data for validation and test (`final_test_...csv`).
- **TeX Files:** Formatted tables ready for inclusion in LaTeX documents, with best results highlighted in bold (`final_test_...tex`).

## Datasets
The data used are those provided by RecBole in their [Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI), already formatted in their own structure.

To use them, simply download the dataset's `.inter` file and place it in its own directory within the `dataset` folder. \
For example: `dataset/ml-100k/ml-100k.inter`\
Some datasets might be missing the `Timestamp` column, in such cases to use Time Ordering you will need to add it.
