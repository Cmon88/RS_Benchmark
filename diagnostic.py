import pandas as pd
import os
import argparse

def analyze_subsample(dataset_path, sample_name):
    df = pd.read_csv(dataset_path, sep='\t')
    
    users = df['user_id:token'].nunique()
    items = df['item_id:token'].nunique()
    interactions = len(df)
    
    user_counts = df['user_id:token'].value_counts()
    item_counts = df['item_id:token'].value_counts()
    
    print(f"\n=== {sample_name} ===")
    print(f"Users: {users}, Items: {items}, Interactions: {interactions}")
    print(f"Avg interactions/user: {interactions/users:.2f}")
    print(f"Avg interactions/item: {interactions/items:.2f}")
    print(f"Users with <2 interactions: {(user_counts < 2).sum()}")
    print(f"Items with <1 interactions: {(item_counts < 1).sum()}")
    print(f"Density: {interactions/(users*items)*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnostic dataset samples')
    parser.add_argument('--dataset', type=str, default='amazon_digital_music', help='Name of the dataset to diagnose')
    parser.add_argument('--n_samples', type=int, default=3, help='Number of samples to check')
    args = parser.parse_args()
    
    dataset_name = args.dataset
    n_samples = args.n_samples
    # Analyze the original dataset
    original_path = f'dataset/{dataset_name}/{dataset_name}.inter'
    if os.path.exists(original_path):
        m1 = pd.read_csv(original_path, sep='\t')
        usersm = m1['user_id:token'].nunique()
        itemsm= m1['item_id:token'].nunique()
        interactionsm = len(m1)
            
        user_counts = m1['user_id:token'].value_counts()
        item_counts = m1['item_id:token'].value_counts()
        print(f"\n=== {dataset_name} (Original) ===")
        print(f"Users: {usersm}, Items: {itemsm}, Interactions: {interactionsm}")
        print(f"Avg interactions/user: {interactionsm/usersm:.2f}")
        print(f"Avg interactions/item: {interactionsm/itemsm:.2f}")
        print(f"Users with <2 interactions: {(user_counts < 2).sum()}")
        print(f"Items with <1 interactions: {(item_counts < 1).sum()}")
        print(f"Density: {interactionsm/(usersm*itemsm)*100:.2f}%")
    else:
        print(f"Original dataset not found at {original_path}")

    # Comparison with ml-100k (fixed reference)
    if os.path.exists('dataset/ml-100k/ml-100k.inter'):
        k100 = pd.read_csv('dataset/ml-100k/ml-100k.inter', sep='\t')
        usersk = k100['user_id:token'].nunique()
        itemsk= k100['item_id:token'].nunique()
        interactions = len(k100)
            
        user_counts = k100['user_id:token'].value_counts()
        item_counts = k100['item_id:token'].value_counts()
        print(f"\n=== ml-100k (Reference) ===")
        print(f"Users: {usersk}, Items: {itemsk}, Interactions: {interactions}")
        print(f"Avg interactions/user: {interactions/usersk:.2f}")
        print(f"Avg interactions/item: {interactions/itemsk:.2f}")
        print(f"Users with <2 interactions: {(user_counts < 2).sum()}")
        print(f"Items with <1 interactions: {(item_counts < 1).sum()}")
        print(f"Density: {interactions/(usersk*itemsk)*100:.2f}%")

    # Analyze generated samples
    for i in range(n_samples):
        sample_path = f'./dataset_sampled/{dataset_name}_sample{i+1}/{dataset_name}_sample{i+1}.inter'
        if os.path.exists(sample_path):
            analyze_subsample(sample_path, f"{dataset_name}_sample{i+1}")
        else:
            print(f"Sample {i+1} not found at {sample_path}")