import os
import json
import csv
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def load_features_json(json_path):
    """Load features from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def detect_feature_columns(sample_dict):
    """Detect feature columns automatically (exclude metadata)."""
    metadata_keys = {'label', 'filename', 'original_category'}
    feature_cols = [k for k in sample_dict.keys() if k not in metadata_keys]
    return sorted(feature_cols)


def extract_features_and_labels(data, feature_cols):
    """Extract feature matrix X and labels y."""
    X = []
    y = []
    
    for sample in data:
        features = [float(sample.get(col, 0)) for col in feature_cols]
        X.append(features)
        y.append(sample.get('label', 0))
    
    return np.array(X, dtype=np.float32), np.array(y)


def save_csv(filepath, X, y, feature_cols, standardized=False):
    """Save train/test data to CSV."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(feature_cols + ['label'])
        # Data rows
        for features, label in zip(X, y):
            writer.writerow(list(features) + [int(label)])


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    features_dir = os.path.join(repo_root, 'hids_dataset', 'features')
    
    json_path = os.path.join(features_dir, 'dataset_features.json')
    
    if not os.path.exists(json_path):
        print(f'Error: {json_path} not found')
        return
    
    print('Loading features from JSON...')
    data = load_features_json(json_path)
    
    # Detect feature columns
    feature_cols = detect_feature_columns(data[0])
    print(f'[+] Detected {len(feature_cols)} features')
    
    # Extract features and labels
    X, y = extract_features_and_labels(data, feature_cols)
    print(f'[+] Total samples: {X.shape[0]}, Features: {X.shape[1]}')
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f'\n[+] Training: {X_train.shape[0]} samples')
    print(f'[+] Testing: {X_test.shape[0]} samples')
    print(f'\n[+] Class distribution:')
    print(f'    Benign (label=1): {sum(y == 1)}')
    print(f'    Malicious (label=0): {sum(y == 0)}')
    
    # Save CSV files
    print(f'\n=== Saving CSV Files ===')
    
    train_path = os.path.join(features_dir, 'train.csv')
    test_path = os.path.join(features_dir, 'test.csv')
    
    save_csv(train_path, X_train, y_train, feature_cols)
    print(f'[+] Saved train.csv ({len(X_train)} rows)')
    
    save_csv(test_path, X_test, y_test, feature_cols)
    print(f'[+] Saved test.csv ({len(X_test)} rows)')
    
    # Also save unscaled versions
    train_unscaled_path = os.path.join(features_dir, 'train_unscaled.csv')
    test_unscaled_path = os.path.join(features_dir, 'test_unscaled.csv')
    
    X_train_unscaled = X[:len(X_train)]
    X_test_unscaled = X[len(X_train):]
    
    save_csv(train_unscaled_path, X_train_unscaled, y_train, feature_cols)
    print(f'[+] Saved train_unscaled.csv ({len(X_train_unscaled)} rows)')
    
    save_csv(test_unscaled_path, X_test_unscaled, y_test, feature_cols)
    print(f'[+] Saved test_unscaled.csv ({len(X_test_unscaled)} rows)')
    
    print(f'\n[+] Done!')
    print(f'    train.csv - standardized training data')
    print(f'    test.csv - standardized testing data')
    print(f'    train_unscaled.csv - original training data')
    print(f'    test_unscaled.csv - original testing data')


if __name__ == '__main__':
    main()
